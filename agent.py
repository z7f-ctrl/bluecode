"""小蓝 Blue —— 基于 LangGraph 的本地个人 coding agent。v0.2 实现。

图：planner → agent → guard → reviewer →(pass/report | revise/agent)→ report → END

节点职责：
- planner   拆需求为 3~6 步计划
- agent     模型 + 工具循环；写/执行类工具只进 pending_changes，不真执行
- guard     有 pending_changes 时 interrupt() 等人审批；通过则真执行
- reviewer  毒舌自审，输出 pass / revise
- report    收尾汇总

v0.2 新增：
- SqliteSaver 持久化（~/.blue/checkpoints.sqlite）
- 多轮会话（同一 thread 内连续提交需求）
- 斜杠命令（/help /quit /clear /history /graph 等）
- --resume 恢复历史会话

CLI 交互在 __main__ 分支。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import traceback
import uuid
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 自动加载项目根目录下的 .env（含 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME）。
# .env 已被 .gitignore 排除；显式环境变量优先级高于 .env（load_dotenv 默认不覆盖已存在变量）。
load_dotenv()
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, Send, interrupt

from prompts import AGENT_PROMPT, PLANNER_PROMPT, REPORT_PROMPT, REVIEWER_PROMPT, WORKER_PROMPT
from tools import (
    ALL_TOOLS,
    FINAL_ANSWER_TOOL,
    PLAN_TOOL_NAMES,
    READ_ONLY_TOOLS,
    _resolve,
    check_command_safety,
    check_python_safety,
    execute_change,
)

MAX_REVIEW_ROUNDS = 3
MAX_TOOL_ITERATIONS = 15
MAX_WORKER_ITERATIONS = 8   # 并行 worker 的工具循环上限（比串行 agent 小，防限流下耗时失控）
MAX_PARALLEL_WORKERS = 4    # 并行 worker 数量上限（实测 API 限流，不宜更高）
BLUE_DIR = os.path.expanduser("~/.blue")
DB_PATH = os.path.join(BLUE_DIR, "checkpoints.sqlite")


def _resettable_add(old: list, new: list) -> list:
    """reducer：空列表 = 显式清空；非空 = 追加。

    并行 worker 的产出经它聚合（LangGraph 并行分支写同一 key 必须有 reducer），
    guard 执行后 / planner 新需求时返回 [] 完成重置。
    """
    return [] if new == [] else old + new


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    request: str
    plan: list[str]
    current_step: int
    pending_changes: Annotated[list[dict], _resettable_add]
    review_rounds: int
    verdict: str
    feedback: str
    parallel_tasks: list[str]        # planner 产出的可并行子任务（空 = 走串行 agent）
    current_subtask: str             # Send 注入给单个 worker 的子任务描述
    worker_notes: Annotated[list[str], _resettable_add]  # 各 worker 的一句话总结，聚合


def initial_state(request: str) -> AgentState:
    return {
        "messages": [],
        "request": request,
        "plan": [],
        "current_step": 0,
        "pending_changes": [],
        "review_rounds": 0,
        "verdict": "proceed",
        "feedback": "",
        "parallel_tasks": [],
        "current_subtask": "",
        "worker_notes": [],
    }


tool_node = ToolNode(READ_ONLY_TOOLS)


_model_cache: ChatOpenAI | None = None
_plain_model_cache: ChatOpenAI | None = None


def _base_kwargs() -> dict:
    kwargs: dict = {"model": os.environ.get("MODEL_NAME", "gpt-4o-mini")}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    return kwargs


def _make_model() -> ChatOpenAI:
    """带工具绑定的模型，供 agent / worker 的工具循环用。"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ChatOpenAI(**_base_kwargs()).bind_tools(ALL_TOOLS)
    return _model_cache


def _make_plain_model() -> ChatOpenAI:
    """不绑工具的模型，供 planner / reviewer / report 等纯文本节点用。

    绑工具的模型对纯文本 prompt 可能以 tool_calls 代替文本输出（K3 实测：
    finish_reason=tool_calls、content 为空），导致 planner JSON 解析退化为单步。
    """
    global _plain_model_cache
    if _plain_model_cache is None:
        _plain_model_cache = ChatOpenAI(**_base_kwargs())
    return _plain_model_cache


def should_skip_planner(request: str) -> bool:
    """判断需求是否简单到不需要 planner 拆解。"""
    # 一句话能说完、无多文件/多步骤迹象的需求直接进 agent
    if len(request) > 30:
        return False
    # 包含这些词说明可能有多步骤
    multi_step_indicators = ["和", "并", "同时", "然后", "接着", "再", "另外", "加上", "以及", "multiple", "and then"]
    if any(w in request for w in multi_step_indicators):
        return False
    return True


def planner(state: AgentState) -> dict:
    if should_skip_planner(state["request"]):
        return {"plan": [state["request"]], "current_step": 0, "verdict": "proceed",
                "parallel_tasks": [], "worker_notes": []}
    resp = _make_plain_model().invoke(
        [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["request"])]
    )
    content = resp.content
    plan: list | None = None
    parallel: list[str] = []
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if isinstance(parsed, list):
            # 旧格式兼容：纯数组 = 串行步骤
            plan = parsed
        elif isinstance(parsed, dict):
            plan = parsed.get("steps")
            raw_parallel = parsed.get("parallel_tasks") or []
            if isinstance(raw_parallel, list):
                seen: set[str] = set()
                for t in raw_parallel:
                    if isinstance(t, str) and t.strip() and t not in seen:
                        seen.add(t)
                        parallel.append(t)
                parallel = parallel[:MAX_PARALLEL_WORKERS]
    except Exception:
        plan = None
    if not isinstance(plan, list) or not all(isinstance(p, str) for p in plan):
        plan = [state["request"]]
    return {"plan": plan, "current_step": 0, "verdict": "proceed",
            "parallel_tasks": parallel, "worker_notes": []}


def agent(state: AgentState) -> dict:
    tip = f"计划：{json.dumps(state['plan'], ensure_ascii=False)}；当前第 {state['current_step'] + 1}/{len(state['plan'])} 步。"
    if state.get("verdict") in ("revise", "rejected") and state.get("feedback"):
        tip += f"\n上一轮评审/意见：{state['feedback']}"

    messages: list = [SystemMessage(content=AGENT_PROMPT), HumanMessage(content=state["request"])]
    # revise 回环时 state["messages"] 已被 reviewer 压缩成单条摘要，直接复用
    messages.extend(state["messages"])
    messages.append(HumanMessage(content=tip))

    updated: list = []
    # delta 语义：pending_changes 有 _resettable_add reducer，这里只收集本轮新增，
    # reducer 负责追加到 state；若从 state 复制再返回全量会导致重复累加。
    pending: list[dict] = []
    step = state.get("current_step", 0)
    iterations = 0

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1
        ai: AIMessage = _make_model().invoke(messages)
        messages.append(ai)
        updated.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            name, args = tc.get("name"), tc.get("args") or {}
            if name == FINAL_ANSWER_TOOL:
                # 显式终止：只读工具，直接执行并把 answer 作为最终答复
                tool_messages = tool_node.invoke(messages)
                tool_messages = tool_messages if isinstance(tool_messages, list) else tool_messages.get("messages", [])
                for tm in tool_messages:
                    messages.append(tm)
                    updated.append(tm)
                # 把 final_answer 内容提炼成一条 AIMessage，作为本轮收尾
                answer_text = args.get("answer", "")
                updated.append(AIMessage(content=answer_text))
                return {"messages": updated, "pending_changes": pending, "current_step": step}
            if name in PLAN_TOOL_NAMES:
                blocked = None
                if name == "plan_run_command":
                    try:
                        check_command_safety(args.get("command", ""))
                        _resolve(args.get("cwd", "."))
                    except ValueError as exc:
                        blocked = str(exc)
                elif name == "plan_run_python":
                    try:
                        check_python_safety(args.get("code", ""))
                    except ValueError as exc:
                        blocked = str(exc)
                if blocked:
                    tool_msg = ToolMessage(
                        content=f"被拦截：{blocked} 请修正后重试。",
                        tool_call_id=tc["id"],
                    )
                else:
                    pending.append({"action": name, **args})
                    # 只回摘要，不把完整 content/大段 args 再塞进上下文
                    summary = {k: v for k, v in args.items() if k in ("path", "command", "cwd")}
                    if "content" in args:
                        summary["content_len"] = len(args["content"])
                    if "old" in args:
                        summary["old_len"] = len(args["old"])
                    if "new" in args:
                        summary["new_len"] = len(args["new"])
                    if "code" in args:
                        summary["code_len"] = len(args["code"])
                    tool_msg = ToolMessage(
                        content=f"已暂存待审批：{name}({json.dumps(summary, ensure_ascii=False)})",
                        tool_call_id=tc["id"],
                    )
                messages.append(tool_msg)
                updated.append(tool_msg)
            else:
                tool_messages = tool_node.invoke(messages)
                tool_messages = tool_messages if isinstance(tool_messages, list) else tool_messages.get("messages", [])
                for tm in tool_messages:
                    messages.append(tm)
                    updated.append(tm)
        step = min(step + 1, max(len(state.get("plan", [])) or 1, 1))
        if pending:
            break
    else:
        # 达到工具迭代上限，强制结束并提示
        updated.append(AIMessage(content=f"（已达工具迭代上限 {MAX_TOOL_ITERATIONS}，强制收尾）"))

    return {"messages": updated, "pending_changes": pending, "current_step": step}


def worker(state: AgentState) -> dict:
    """并行 worker：处理 planner 派生的一个独立子任务，只读 + 暂存，不真执行。

    与 agent 节点同构的精简工具循环；不写 messages（并行分支消息会交错混乱），
    产出（pending_changes / worker_notes）经 _resettable_add reducer 聚合。
    """
    subtask = state["current_subtask"]
    messages: list = [
        SystemMessage(content=WORKER_PROMPT),
        HumanMessage(content=f"总需求：{state['request']}\n你负责的子任务：{subtask}"),
    ]
    pending: list[dict] = []

    for _ in range(MAX_WORKER_ITERATIONS):
        ai: AIMessage = _make_model().invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            name, args = tc.get("name"), tc.get("args") or {}
            if name == FINAL_ANSWER_TOOL:
                # 拿到总结即返回，不再 invoke，无需补 ToolMessage
                note = args.get("answer", "") or "完成"
                return {"pending_changes": pending,
                        "worker_notes": [f"「{subtask}」{note}"]}
            if name in PLAN_TOOL_NAMES:
                # 与 agent 节点一致的双路径安全校验（agent 直接搬 args，绕过 @tool 内检查）
                blocked = None
                if name == "plan_run_command":
                    try:
                        check_command_safety(args.get("command", ""))
                        _resolve(args.get("cwd", "."))
                    except ValueError as exc:
                        blocked = str(exc)
                elif name == "plan_run_python":
                    try:
                        check_python_safety(args.get("code", ""))
                    except ValueError as exc:
                        blocked = str(exc)
                if blocked:
                    messages.append(ToolMessage(
                        content=f"被拦截：{blocked} 请修正后重试。", tool_call_id=tc["id"]))
                    continue
                pending.append({"action": name, **args})
                summary = {k: v for k, v in args.items() if k in ("path", "command", "cwd")}
                messages.append(ToolMessage(
                    content=f"已暂存待审批：{name}({json.dumps(summary, ensure_ascii=False)})",
                    tool_call_id=tc["id"]))
            else:
                tool_messages = tool_node.invoke(messages)
                tool_messages = tool_messages if isinstance(tool_messages, list) else tool_messages.get("messages", [])
                messages.extend(tool_messages)
        if pending:
            break
    note = f"暂存 {len(pending)} 处改动" if pending else "未产生改动"
    return {"pending_changes": pending, "worker_notes": [f"「{subtask}」{note}"]}


def guard(state: AgentState) -> dict:
    if not state.get("pending_changes"):
        notes = "；".join(state.get("worker_notes", []))
        return {"verdict": "proceed", "pending_changes": [], "feedback": notes}
    answer = interrupt({"changes": state["pending_changes"], "question": "以上改动/命令是否允许执行？"})
    action = answer.get("action") if isinstance(answer, dict) else "approve"
    if action == "reject":
        return {"verdict": "rejected", "pending_changes": [], "feedback": answer.get("note", "用户拒绝")}
    if action == "modify":
        return {"verdict": "revise", "pending_changes": [], "feedback": answer.get("note", "用户要求修改")}
    # 记录改动文件列表，供 verifier 做语法检查
    changed_files = [c.get("path", "") for c in state["pending_changes"] if c.get("path")]
    summary = "\n".join(execute_change(c) for c in state["pending_changes"])
    if changed_files:
        summary += f"\n\n【改动文件】{'; '.join(changed_files)}"
    if state.get("worker_notes"):
        summary += "\n\n【并行子任务】\n" + "\n".join(state["worker_notes"])
    return {"verdict": "approved", "pending_changes": [], "feedback": summary}


def _compress_messages(messages: list, request: str) -> str:
    """把 agent 的完整工具历史压缩成一段人可读摘要，供 revise 回环时替代原始消息。

    不调用 LLM，用规则提取：读了哪些文件、暂存了哪些改动、执行结果。
    """
    read_files: list[str] = []
    planned_changes: list[str] = []
    executed_results: list[str] = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # 提取 read_file 结果
            if "共" in content and "行" in content and "显示" in content:
                first_line = content.split("\n")[0]
                read_files.append(first_line)
            # 提取暂存信息
            elif "已暂存待审批" in content:
                planned_changes.append(content.replace("已暂存待审批：", ""))
            # 提取命令执行结果
            elif "命令成功" in content or "命令退出码" in content:
                executed_results.append(content[:200] + ("…" if len(content) > 200 else ""))
        elif isinstance(msg, AIMessage):
            # 提取只读工具调用的参数摘要（plan_* 已由 ToolMessage 覆盖，避免重复）
            for tc in msg.tool_calls or []:
                name = tc.get("name", "")
                if name in ("read_file", "grep", "list_files"):
                    args = tc.get("args", {})
                    if name == "read_file":
                        read_files.append(f"read_file(path={args.get('path')})")
                    elif name == "grep":
                        read_files.append(f"grep(pattern={args.get('pattern')!r})")
                    elif name == "list_files":
                        read_files.append(f"list_files(dir={args.get('dir', '.')})")

    parts = [f"需求：{request}"]
    if read_files:
        parts.append(f"读取了 {len(read_files)} 个文件：{'; '.join(read_files[:5])}{'…' if len(read_files) > 5 else ''}")
    if planned_changes:
        parts.append(f"暂存了 {len(planned_changes)} 处改动：{'; '.join(planned_changes[:5])}{'…' if len(planned_changes) > 5 else ''}")
    if executed_results:
        parts.append(f"执行结果：{'; '.join(executed_results[:3])}")
    if not (read_files or planned_changes or executed_results):
        parts.append("（无工具调用历史）")
    return "\n".join(parts)


def verifier(state: AgentState) -> dict:
    """审批通过后自动验证：语法检查 + 尝试跑测试。结果供 reviewer 参考。"""
    verdict = state.get("verdict", "proceed")
    if verdict != "approved":
        # 只读/拒绝/修改 不需要验证
        return {"feedback": state.get("feedback", "")}

    feedback = state.get("feedback", "")
    results: list[str] = []

    # 从 feedback 解析改动文件列表（guard 执行后写入）
    changed_files: list[str] = []
    if "【改动文件】" in feedback:
        files_part = feedback.split("【改动文件】")[-1].strip()
        changed_files = [f.strip() for f in files_part.split(";") if f.strip()]

    # 1. 语法检查：对所有改动过的 .py 文件跑 py_compile
    import py_compile
    for path in changed_files:
        if path.endswith(".py"):
            try:
                p = _resolve(path)
                py_compile.compile(str(p), doraise=True)
                results.append(f"✓ {path} 语法检查通过")
            except py_compile.PyCompileError as exc:
                results.append(f"✗ {path} 语法错误：{exc.msg}")
            except Exception as exc:
                results.append(f"? {path} 语法检查异常：{exc}")

    # 2. 尝试发现测试并执行（如果项目里有测试文件）
    # 排除 benchmark 题库：那里是故意带 bug 的题目，跑它们只会得到必挂的噪音，
    # 还会误导 reviewer 打回（真机实测：项目根跑无关需求被 benchmarks/tasks 污染）
    EXCLUDED_TEST_PARTS = ("benchmarks/quixbugs/tasks", "__pycache__")
    test_files = []
    try:
        for pattern in ("test_*.py", "*_test.py"):
            for item in _resolve(".").rglob(pattern):
                rel = str(item.relative_to(_resolve(".")))
                if any(ex in rel for ex in EXCLUDED_TEST_PARTS):
                    continue
                test_files.append(rel)
    except Exception:
        pass

    if test_files:
        import subprocess
        # 最多跑 3 个测试文件，避免超时
        for tf in test_files[:3]:
            try:
                r = subprocess.run(
                    ["python3", "-m", "pytest", tf, "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=30, cwd=_resolve(".")
                )
                if r.returncode == 0:
                    results.append(f"✓ {tf} 测试通过")
                else:
                    # 只保留关键错误行
                    err_lines = [l for l in r.stdout.split("\n") if "FAILED" in l or "Error" in l or "assert" in l.lower()]
                    results.append(f"✗ {tf} 测试失败：{'; '.join(err_lines[:3])}")
            except subprocess.TimeoutExpired:
                results.append(f"? {tf} 测试超时（30s）")
            except Exception as exc:
                results.append(f"? {tf} 测试执行异常：{exc}")

    if results:
        feedback += "\n\n【自动验证结果】\n" + "\n".join(results)
    else:
        feedback += "\n\n【自动验证结果】无 .py 改动或未找到测试文件"

    return {"feedback": feedback}


def reviewer(state: AgentState) -> dict:
    rounds = state.get("review_rounds", 0)
    verdict = state.get("verdict", "proceed")
    if verdict == "rejected":
        return {"verdict": "pass", "feedback": f"（用户已拒绝）{state.get('feedback', '')}", "review_rounds": rounds + 1}
    if verdict == "proceed":
        return {"verdict": "pass", "feedback": "本次为只读任务，无需改动。", "review_rounds": rounds + 1}

    resp = _make_plain_model().invoke(
        [
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(
                content=(
                    f"用户需求：{state['request']}\n"
                    f"评审轮数（当前第 {rounds} 轮，上限 {MAX_REVIEW_ROUNDS}）\n"
                    f"本次执行结果：\n{state.get('feedback', '(无)')}"
                )
            ),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    m = re.search(r"\bverdict\s*[:：]?\s*(pass|revise)\b", text, re.IGNORECASE)
    if m:
        v = m.group(1).lower()
        new_verdict: Literal["pass", "revise"] = "revise" if v == "revise" else "pass"
    else:
        # 模型未按格式输出时 fail-closed：默认打回而非放行（实测模型常用自然语言
        # 回答"我先读一下代码"，无 verdict 关键字，fail-open 会导致 bug 未修就交付）。
        # 不会死循环：到 MAX_REVIEW_ROUNDS 上限下面会强制放行。
        new_verdict = "revise"
    new_rounds = rounds + 1
    if new_verdict == "revise" and new_rounds >= MAX_REVIEW_ROUNDS:
        new_verdict = "pass"
        text = f"（已达评审上限强制放行。历史意见：{text}）"

    # revise 回环时压缩消息历史，避免完整工具调用细节占 token
    update: dict = {"verdict": new_verdict, "feedback": text, "review_rounds": new_rounds}
    if new_verdict == "revise":
        compressed = _compress_messages(state.get("messages", []), state["request"])
        # 逐条 RemoveMessage 删除旧历史，再追加单条摘要
        removals = [RemoveMessage(id=m.id) for m in state.get("messages", []) if m.id]
        update["messages"] = removals + [HumanMessage(content=f"【上一轮执行摘要】\n{compressed}")]
    return update


def report(state: AgentState) -> dict:
    resp = _make_plain_model().invoke(
        [
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(
                content=(
                    f"用户需求：{state['request']}\n"
                    f"最终改动清单：{json.dumps(state.get('pending_changes', []), ensure_ascii=False)}\n"
                    f"评审轮数：{state.get('review_rounds', 0)}\n"
                    f"执行/测试结果：{state.get('feedback', '(无)')}"
                )
            ),
        ]
    )
    return {"feedback": resp.content if isinstance(resp.content, str) else str(resp.content)}


def route_by_verdict(state: AgentState) -> Literal["agent", "report"]:
    return "agent" if state.get("verdict") == "revise" else "report"


def route_after_planner(state: AgentState):
    """planner 之后：有可并行子任务（≥2）则 Send 扇出 worker，否则走串行 agent。"""
    tasks = state.get("parallel_tasks") or []
    if len(tasks) >= 2:
        return [Send("worker", {**state, "current_subtask": t}) for t in tasks]
    return "agent"


def _ensure_blue_dir() -> None:
    os.makedirs(BLUE_DIR, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    _ensure_blue_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = SqliteSaver(_get_conn())
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("worker", worker)
    builder.add_node("guard", guard)
    builder.add_node("verifier", verifier)
    builder.add_node("reviewer", reviewer)
    builder.add_node("report", report)
    builder.add_edge(START, "planner")
    # planner 后条件扇出：parallel_tasks ≥ 2 → 并行 worker，否则串行 agent
    builder.add_conditional_edges("planner", route_after_planner, ["agent", "worker"])
    builder.add_edge("agent", "guard")
    builder.add_edge("worker", "guard")  # join：所有并行 worker 完成后 guard 执行一次
    builder.add_edge("guard", "verifier")
    builder.add_edge("verifier", "reviewer")
    builder.add_conditional_edges(
        "reviewer", route_by_verdict, {"agent": "agent", "report": "report"}
    )
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────── 会话管理 ───────────────────────────

class Session:
    """一次交互式会话：维护 thread_id 与轮次，支持多轮需求。"""

    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id or f"blue-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.round = 0
        self.created_at = datetime.now().isoformat(timespec="seconds")

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def next_round(self) -> int:
        self.round += 1
        return self.round


def _save_session_meta(sess: Session) -> None:
    """把会话元信息写入 sqlite 辅助表，供 --resume 列表查询。"""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active TEXT,
                rounds INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (thread_id, created_at, last_active, rounds)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                last_active = excluded.last_active,
                rounds = excluded.rounds
            """,
            (sess.thread_id, sess.created_at, datetime.now().isoformat(timespec="seconds"), sess.round),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions() -> list[dict]:
    """从辅助表读历史会话列表。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT thread_id, created_at, last_active, rounds FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        return [
            {"thread_id": r[0], "created_at": r[1], "last_active": r[2], "rounds": r[3]}
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ─────────────────────────── 斜杠命令 ───────────────────────────

SLASH_HELP = """可用斜杠命令：
  /help          显示本帮助
  /quit, /exit   退出当前会话
  /clear         清空当前会话的上下文（开启新 thread）
  /history       查看本会话已完成的轮次与状态摘要
  /graph         打印图拓扑
  /resume        列出历史会话并恢复（等价于启动时 --resume）
  /new           强制开启新 thread（保留旧 checkpoint）
"""


def handle_slash(cmd: str, sess: Session, graph) -> tuple[bool, Session | None]:
    """处理斜杠命令。返回 (should_continue, new_session_or_none)。
    should_continue=False 表示退出主循环。
    """
    cmd = cmd.strip().lower()
    if cmd in ("/quit", "/exit"):
        print("[蓝] 👋 再见！")
        return False, None
    if cmd == "/help":
        print(SLASH_HELP)
        return True, None
    if cmd == "/clear":
        new_sess = Session()
        print(f"[蓝] 🧹 已开启新 thread：{new_sess.thread_id}")
        return True, new_sess
    if cmd == "/new":
        new_sess = Session()
        print(f"[蓝] 🆕 新 thread：{new_sess.thread_id}")
        return True, new_sess
    if cmd == "/history":
        cur = graph.get_state(sess.config)
        vals = cur.values if cur else {}
        print(f"[蓝] 当前 thread：{sess.thread_id}")
        print(f"     已进行 {sess.round} 轮需求")
        print(f"     图状态：next={list(cur.next) if cur and cur.next else '（已完成）'}")
        if vals.get("plan"):
            print(f"     最近计划：{json.dumps(vals['plan'], ensure_ascii=False)}")
        if vals.get("review_rounds"):
            print(f"     评审轮数：{vals['review_rounds']}")
        return True, None
    if cmd == "/graph":
        print(graph.get_graph().draw_ascii())
        return True, None
    if cmd == "/resume":
        sessions = list_sessions()
        if not sessions:
            print("[蓝] 暂无历史会话。")
            return True, None
        print("[蓝] 历史会话（最近在前）：")
        for i, s in enumerate(sessions[:10], 1):
            marker = " 👈 当前" if s["thread_id"] == sess.thread_id else ""
            print(f"  {i}. {s['thread_id']}  轮次={s['rounds']}  最后活动={s['last_active']}{marker}")
        choice = input("输入序号恢复，或回车取消 > ").strip()
        if not choice:
            return True, None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                new_sess = Session(thread_id=sessions[idx]["thread_id"])
                new_sess.round = sessions[idx]["rounds"]
                print(f"[蓝] 🔁 已恢复 thread：{new_sess.thread_id}")
                return True, new_sess
            print("[蓝] 序号无效。")
        except ValueError:
            print("[蓝] 请输入数字。")
        return True, None
    print(f"[蓝] 未知命令 {cmd}，输入 /help 查看可用命令。")
    return True, None


# ─────────────────────────── 主交互循环 ───────────────────────────


# ─────────────────────────── step 回调注册机制（借鉴 smolagents step_callbacks） ───────────────────────────
# 节点输出通过回调链处理，默认回调是 CLI 打印。
# 外部（TUI/Web UI/测试）可注册自己的回调，无需修改核心逻辑。
# 回调签名：fn(node_name: str, output: dict) -> None

from collections.abc import Callable

_step_callbacks: list[Callable[[str, dict], None]] = []


def register_step_callback(fn: Callable[[str, dict], None]) -> None:
    """注册一个节点输出回调。按注册顺序依次调用。"""
    _step_callbacks.append(fn)


def clear_step_callbacks() -> None:
    """清空所有回调（测试用）。"""
    _step_callbacks.clear()


def _emit_step(node_name: str, output: dict) -> None:
    for fn in _step_callbacks:
        try:
            fn(node_name, output)
        except Exception:  # noqa: BLE001 — 回调异常不阻断主流程
            pass


def _print_pending(prefix: str, changes: list[dict]) -> None:
    """打印暂存的改动清单（agent 与 worker 共用）。"""
    for c in changes:
        shown = {k: v for k, v in c.items() if k != "action"}
        # 大字段只显示长度，不打印完整内容
        for big in ("content", "old", "new"):
            if big in shown:
                shown[f"{big}_len"] = len(shown.pop(big))
        print(f"{prefix} 已暂存待审批 → {c['action']}({json.dumps(shown, ensure_ascii=False)})")


def _print_node(node_name: str, output: dict) -> None:
    if node_name == "planner":
        plan = output.get("plan", [])
        parallel = output.get("parallel_tasks") or []
        if len(parallel) >= 2:
            print(f"[蓝] 拆出 {len(parallel)} 个独立子任务，并行 worker 处理：{json.dumps(parallel, ensure_ascii=False)}")
        # planner 条件化：单步且与需求原文一致 → 简单需求直接执行
        elif len(plan) == 1:
            print("[蓝] 简单需求，直接执行")
        else:
            print(f"[蓝] 计划：{json.dumps(plan, ensure_ascii=False)}")
    elif node_name == "agent" and output.get("pending_changes"):
        _print_pending("[蓝]", output["pending_changes"])
    elif node_name == "worker":
        _print_pending("[蓝·worker]", output.get("pending_changes", []))
        for note in output.get("worker_notes", []):
            print(f"[蓝·worker] {note}")
    elif node_name == "guard":
        print(f"[蓝] 审批结果：{output.get('verdict')}")
    elif node_name == "verifier":
        fb = output.get("feedback", "")
        if "【自动验证结果】" in fb:
            # 只打印验证部分，不重复执行结果
            verify_part = fb.split("【自动验证结果】")[-1].strip()
            print(f"[蓝] 🔍 自动验证：{verify_part}")
    elif node_name == "reviewer":
        mark = "✅ 放行" if output.get("verdict") == "pass" else "🔪 打回"
        print(f"[评审] {mark}｜{output.get('feedback', '')}")
    elif node_name == "report":
        print(f"\n[蓝] {output.get('feedback', '')}")


def run_round(graph, sess: Session, request: str) -> None:
    """执行一轮需求：stream 图执行，处理 interrupt 审批。"""
    config = sess.config
    state = initial_state(request)
    print(f"[蓝] ★ 第 {sess.next_round()} 轮收到！开始干活。")
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)
    except Exception:
        traceback.print_exc()

    while True:
        cur = graph.get_state(config)
        if not cur.next:
            break
        for task in cur.tasks:
            if task.interrupts:
                payload = task.interrupts[0].value
                print("\n[蓝] ⏸ 等待你审批：")
                for ci, ch in enumerate(payload.get("changes", []), 1):
                    shown = {k: v for k, v in ch.items() if k != "action"}
                    print(f"  {ci}. [{ch['action']}] {json.dumps(shown, ensure_ascii=False)}")

                def ask(prompt: str) -> str:
                    try:
                        return input(prompt).strip().lower()
                    except EOFError:
                        print("\n[蓝] ⏹ 输入流已关闭，按「拒绝」安全中止。")
                        return "n"

                choice = ask("[y]允许 [n]拒绝 [m]修改意见 > ")
                resume_val = {"action": "approve"}
                if choice == "n":
                    resume_val = {"action": "reject", "note": ask("  拒绝原因(可空) > ") or "用户拒绝"}
                elif choice == "m":
                    resume_val = {"action": "modify", "note": ask("  修改意见 > ")}
                print()
                for chunk in graph.stream(Command(resume=resume_val), config=config, stream_mode="updates"):
                    for node_name, output in chunk.items():
                        if node_name == "guard" and output.get("verdict") == "approved":
                            print(f"[蓝] ✅ 已执行：\n{output.get('feedback', '')}")
                        else:
                            _emit_step(node_name, output)
    _save_session_meta(sess)


def run_interactive(graph, request: str | None = None) -> None:
    """多轮交互主循环：支持连续提需求 + 斜杠命令。"""
    register_step_callback(_print_node)  # 默认回调：CLI 打印
    sess = Session()
    if request:
        run_round(graph, sess, request)
    print("\n[蓝] 进入多轮模式。输入 /help 查看命令，直接输入需求继续干活。")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            print("\n[蓝] 👋 输入流关闭，退出。")
            break
        if not line:
            continue
        if line.startswith("/"):
            cont, new_sess = handle_slash(line, sess, graph)
            if not cont:
                break
            if new_sess is not None:
                sess = new_sess
            continue
        run_round(graph, sess, line)


def _resume_picker() -> str | None:
    """启动时的 --resume 会话选择器。返回选中的 thread_id 或 None。"""
    sessions = list_sessions()
    if not sessions:
        print("[蓝] 暂无历史会话可恢复。")
        return None
    print("[蓝] 历史会话（最近在前）：")
    for i, s in enumerate(sessions[:10], 1):
        print(f"  {i}. {s['thread_id']}  轮次={s['rounds']}  最后活动={s['last_active']}")
    choice = input("输入序号恢复，或回车开启新会话 > ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]["thread_id"]
        print("[蓝] 序号无效，开启新会话。")
    except ValueError:
        print("[蓝] 输入无效，开启新会话。")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="小蓝 Blue —— 本地个人 coding agent")
    parser.add_argument("request", nargs="?", default=None, help='要做的事，例如 "给 hello.py 加错误处理并写测试"')
    parser.add_argument("--show-graph", action="store_true", help="打印图拓扑后退出")
    parser.add_argument("--resume", action="store_true", help="恢复历史会话")
    parser.add_argument("--auto-approve", action="store_true", help="benchmark 模式：guard 自动审批通过，不中断等待人工")
    args = parser.parse_args()

    graph = build_graph()
    if args.show_graph:
        print(graph.get_graph().draw_ascii())
        return
    if args.resume:
        tid = _resume_picker()
        if tid:
            sess = Session(thread_id=tid)
            # 恢复后先展示当前状态
            cur = graph.get_state(sess.config)
            if cur and cur.next:
                print(f"[蓝] 恢复 thread {tid}，图处于等待状态：{list(cur.next)}")
                # 如果有 pending interrupt，继续走审批循环
                run_round(graph, sess, "")  # 空请求不会触发新 planner，直接检查 state
            else:
                print(f"[蓝] 恢复 thread {tid}，上轮已完成。输入新需求继续。")
                run_interactive(graph)
            return
        # 用户取消或无效 → 落入新会话
    if args.auto_approve:
        # benchmark 模式：非交互，单轮，guard 自动通过
        register_step_callback(_print_node)
        sess = Session()
        run_round_auto(graph, sess, args.request or "")
        return
    run_interactive(graph, args.request)


def run_round_auto(graph, sess: Session, request: str) -> None:
    """benchmark 模式：单轮执行，guard 自动 approve 不中断。"""
    config = sess.config
    state = initial_state(request)
    print(f"[蓝] ★ benchmark 模式收到：{request}")
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)
    except Exception:
        traceback.print_exc()

    # 自动审批所有 interrupt
    while True:
        cur = graph.get_state(config)
        if not cur.next:
            break
        for task in cur.tasks:
            if task.interrupts:
                print("[蓝] ⏸ 自动审批通过")
                for chunk in graph.stream(Command(resume={"action": "approve"}), config=config, stream_mode="updates"):
                    for node_name, output in chunk.items():
                        _emit_step(node_name, output)
    _save_session_meta(sess)


if __name__ == "__main__":
    main()