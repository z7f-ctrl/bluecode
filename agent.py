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

v0.7 阶段一新增：
- /retry 断点续跑（run_round 的审批循环抽为 _drain 公共函数，续跑共用）
- _logged_invoke 对瞬时错误（429/5xx/超时/连接错误）指数退避自动重试
- .blue.toml 权限分级（allow/ask/deny 三档，两层配置逐键合并，见 tools.py）

CLI 交互在 __main__ 分支。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
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
    ACTION_CATEGORY,
    ALL_TOOLS,
    FINAL_ANSWER_TOOL,
    PLAN_TOOL_NAMES,
    READ_ONLY_TOOLS,
    _resolve,
    check_command_safety,
    check_python_safety,
    execute_change,
    permission_for_action,
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
    thread_id: str                     # 会话标识（guard 快照/审计等节点内需要）
    plan: list[str]
    current_step: int
    pending_changes: Annotated[list[dict], _resettable_add]
    review_rounds: int
    verdict: str
    feedback: str
    parallel_tasks: list[str]        # planner 产出的可并行子任务（空 = 走串行 agent）
    current_subtask: str             # Send 注入给单个 worker 的子任务描述
    worker_notes: Annotated[list[str], _resettable_add]  # 各 worker 的一句话总结，聚合
    executed_changes: list[dict]     # guard 执行通过后留存的完整改动（reviewer 看 diff、report 列清单；pending_changes 审批后即清空，不能用）
    changed_files: list[str]         # guard 写入的改动文件列表（verifier 读，替代【改动文件】文本协议）


def initial_state(request: str) -> AgentState:
    return {
        "messages": [],
        "request": request,
        "thread_id": "",
        "plan": [],
        "current_step": 0,
        "pending_changes": [],
        "review_rounds": 0,
        "verdict": "proceed",
        "feedback": "",
        "parallel_tasks": [],
        "current_subtask": "",
        "worker_notes": [],
        "executed_changes": [],
        "changed_files": [],
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


# ─────────────────────────── 文件日志（节点事件 + 可选 LLM 全文） ───────────────────────────
# 两层，都落 ~/.blue/logs/（本地，不入库）：
# - 节点日志 blue-<date>.log：节点输出摘要 + 异常，CLI 启动时经 step 回调挂载
# - LLM 日志 blue-llm-<date>.log：BLUE_LOG_LLM=1 时记录每次调用的请求摘要、
#   响应全文、finish_reason、token usage、耗时（功能调试 + 性能分析用）

LOG_DIR = os.path.join(BLUE_DIR, "logs")


def _get_logger(name: str, filename: str) -> logging.Logger:
    """懒创建文件 logger：首次使用才建目录/文件。多进程并发写同一文件时
    长行可能交错（benchmark 并行子进程场景），可读性受损但不丢数据。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(os.path.join(LOG_DIR, filename), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _node_logger() -> logging.Logger:
    return _get_logger("blue.node", f"blue-{datetime.now():%Y%m%d}.log")


def _llm_logger() -> logging.Logger:
    return _get_logger("blue.llm", f"blue-llm-{datetime.now():%Y%m%d}.log")


def _llm_log_enabled() -> bool:
    return os.environ.get("BLUE_LOG_LLM", "").lower() in ("1", "true", "yes", "on")


# ─────────────────────────── token 用量追踪 ───────────────────────────
# _logged_invoke 统一入口无条件累加（带锁，并行 worker 共用）；每轮需求结束
# 打印本轮消耗并计入 Session。单次调用明细见 BLUE_LOG_LLM=1 的 LLM 全文日志。

_TOKEN_LOCK = threading.Lock()
_TOKEN_COLLECTOR = {"prompt": 0, "completion": 0, "calls": 0}


def _extract_usage(resp) -> tuple[int, int]:
    """从模型响应提取 (prompt_tokens, completion_tokens)。
    优先 LangChain 标准 usage_metadata，fallback OpenAI 原生 response_metadata。"""
    um = getattr(resp, "usage_metadata", None)
    if um:
        return um.get("input_tokens", 0) or 0, um.get("output_tokens", 0) or 0
    meta = getattr(resp, "response_metadata", None) or {}
    tu = meta.get("token_usage") or {}
    return tu.get("prompt_tokens", 0) or 0, tu.get("completion_tokens", 0) or 0


def _record_usage(resp) -> None:
    p, c = _extract_usage(resp)
    with _TOKEN_LOCK:
        _TOKEN_COLLECTOR["prompt"] += p
        _TOKEN_COLLECTOR["completion"] += c
        _TOKEN_COLLECTOR["calls"] += 1


def _reset_token_usage() -> None:
    with _TOKEN_LOCK:
        _TOKEN_COLLECTOR.update(prompt=0, completion=0, calls=0)


def _token_usage_snapshot() -> dict:
    with _TOKEN_LOCK:
        return dict(_TOKEN_COLLECTOR)


# ─────────────────────────── 瞬时错误自动重试（v0.7） ───────────────────────────
# HTTP 429 / 5xx / 超时 / 连接错误按指数退避自动重试（2s → 8s → 20s + 随机抖动），
# 重试耗尽才把异常抛给上层（/retry 手工兜底）；非瞬时错误（401/400 等）直接抛。

_RETRY_DELAYS = (2, 8, 20)  # 最多 3 次重试的基准间隔（秒）
# openai SDK 异常类型名兜底（防御性：拿不到 status_code 时按类型名匹配）
_TRANSIENT_EXC_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
}


def _is_transient_error(exc: Exception) -> bool:
    """判断是否瞬时错误（值得自动重试）。优先 status_code，拿不到时按异常类型名匹配。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        try:
            code = int(status)
            return code == 429 or code >= 500 or code in (408, 409)
        except (TypeError, ValueError):
            pass
    if type(exc).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _logged_invoke(model: ChatOpenAI, messages: list, caller: str):
    """model.invoke 的观测包装：瞬时错误自动重试；累加 token 用量；
    BLUE_LOG_LLM=1 时落 LLM 全文日志（失败尝试也记日志，usage 只累加成功的）。"""
    attempt = 0
    while True:
        t0 = time.monotonic()
        try:
            resp = model.invoke(messages)
            break
        except Exception as exc:
            if _llm_log_enabled():
                _llm_logger().info(
                    "=== %s (failed %.1fs, attempt %d) ===\n%s: %s",
                    caller, time.monotonic() - t0, attempt + 1, type(exc).__name__, exc,
                )
            if attempt < len(_RETRY_DELAYS) and _is_transient_error(exc):
                delay = _RETRY_DELAYS[attempt] + random.uniform(0, 1.5)  # 随机抖动防惊群
                attempt += 1
                print(_c(f"[蓝] ⏳ 限流/网络波动，{delay:.0f}s 后自动重试 ({attempt}/{len(_RETRY_DELAYS)})", _C.YELLOW))
                time.sleep(delay)
                continue
            raise
    _record_usage(resp)
    if _llm_log_enabled():
        duration = time.monotonic() - t0
        lines = [f"=== {caller} ({duration:.1f}s) ==="]
        for m in messages:
            role = getattr(m, "type", "?")
            content = m.content if isinstance(m.content, str) else str(m.content)
            line = f"<{role}> {content[:300]}"
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                line += f" tool_calls={json.dumps(tcs, ensure_ascii=False)[:300]}"
            lines.append(line)
        out = resp.content if isinstance(resp.content, str) else str(resp.content)
        lines.append(f"--> {out}")
        if getattr(resp, "tool_calls", None):
            lines.append(f"--> tool_calls={json.dumps(resp.tool_calls, ensure_ascii=False)}")
        meta = getattr(resp, "response_metadata", None) or {}
        p_tok, c_tok = _extract_usage(resp)  # 与收集器同口径（双来源），不写 None/None
        lines.append(f"--> finish={meta.get('finish_reason')} tokens={p_tok}/{c_tok}")
        _llm_logger().info("\n".join(lines))
    return resp


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
                "parallel_tasks": [], "worker_notes": [],
                "executed_changes": [], "changed_files": []}
    resp = _logged_invoke(
        _make_plain_model(),
        [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["request"])],
        "planner",
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
            "parallel_tasks": parallel, "worker_notes": [],
            "executed_changes": [], "changed_files": []}


def _precheck_plan_tool(name: str, args: dict) -> str | None:
    """plan_* 暂存前校验（agent 与 worker 共用，返回拦截原因；None = 通过）。

    - plan_run_command：危险关键词/白名单 + cwd 沙箱
    - plan_run_python：ast 静态检查
    - .blue.toml deny：暂存即拒，反馈模型换方案（execute_change 还有最后防线）
    """
    try:
        if name == "plan_run_command":
            check_command_safety(args.get("command", ""))
            _resolve(args.get("cwd", "."))
        elif name == "plan_run_python":
            check_python_safety(args.get("code", ""))
    except ValueError as exc:
        return str(exc)
    if permission_for_action(name) == "deny":
        return f"此类操作被 .blue.toml 配置禁止（{ACTION_CATEGORY.get(name, '')}=deny），请换方案"
    return None


def agent(state: AgentState) -> dict:
    tip = f"计划：{json.dumps(state['plan'], ensure_ascii=False)}；当前第 {state['current_step'] + 1}/{len(state['plan'])} 步。"
    if state.get("verdict") in ("revise", "rejected") and state.get("feedback"):
        tip += f"\n上一轮评审/意见：{state['feedback']}"

    messages: list = [SystemMessage(content=AGENT_PROMPT), HumanMessage(content=state["request"])]
    # revise 回环时 state["messages"] 已被 reviewer 压缩成单条摘要，直接复用
    messages.extend(state["messages"])
    messages.append(HumanMessage(content=tip))
    head_len = len(messages)  # 头部固定不动；循环内滑动窗口只压缩之后的工具交互

    updated: list = []
    # delta 语义：pending_changes 有 _resettable_add reducer，这里只收集本轮新增，
    # reducer 负责追加到 state；若从 state 复制再返回全量会导致重复累加。
    pending: list[dict] = []
    step = state.get("current_step", 0)
    iterations = 0

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1
        messages = _sliding_compress(messages, head_len, state["request"])
        ai: AIMessage = _logged_invoke(_make_model(), messages, "agent")
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
                blocked = _precheck_plan_tool(name, args)
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
        # 计划步只在一批改动攒出（即将进入审批）时推进：
        # 此前每次工具迭代都自增，step 语义实际是「迭代计数」，
        # 与 tip 里「当前第 N/M 步」的计划步含义不符
        if pending:
            step = min(step + 1, max(len(state.get("plan", [])) or 1, 1))
            break
    else:
        # 达到工具迭代上限，强制结束并提示
        updated.append(AIMessage(content=f"（已达工具迭代上限 {MAX_TOOL_ITERATIONS}，强制收尾）"))

    return {"messages": updated, "pending_changes": pending, "current_step": step}


def _worker_result(subtask: str, note: str, pending: list[dict]) -> dict:
    """worker 返回值组装。

    pending 为空时**省略该 key**：_resettable_add 语义里空列表 = 清空，
    并行分支更新应用顺序不确定，返回 [] 会抹掉兄弟 worker 已聚合的改动。
    """
    r: dict = {"worker_notes": [f"「{subtask}」{note}"]}
    if pending:
        r["pending_changes"] = pending
    return r


def worker(state: AgentState) -> dict:
    """并行 worker：处理 planner 派生的一个独立子任务，只读 + 暂存，不真执行。

    与 agent 节点同构的精简工具循环；不写 messages（并行分支消息会交错混乱），
    产出（pending_changes / worker_notes）经 _resettable_add reducer 聚合。
    单点失败（API 超时/限流/异常）降级为失败 note，不拖垮整图。
    """
    subtask = state["current_subtask"]
    try:
        messages: list = [
            SystemMessage(content=WORKER_PROMPT),
            HumanMessage(content=f"总需求：{state['request']}\n你负责的子任务：{subtask}"),
        ]
        pending: list[dict] = []

        for _ in range(MAX_WORKER_ITERATIONS):
            messages = _sliding_compress(messages, 2, state["request"])
            ai: AIMessage = _logged_invoke(_make_model(), messages, "worker")
            messages.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                name, args = tc.get("name"), tc.get("args") or {}
                if name == FINAL_ANSWER_TOOL:
                    # 拿到总结即返回，不再 invoke，无需补 ToolMessage
                    note = args.get("answer", "") or "完成"
                    return _worker_result(subtask, note, pending)
                if name in PLAN_TOOL_NAMES:
                    # 与 agent 节点一致的双路径安全校验（agent 直接搬 args，绕过 @tool 内检查）
                    blocked = _precheck_plan_tool(name, args)
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
        return _worker_result(subtask, note, pending)
    except Exception as exc:  # noqa: BLE001 — 并行 worker 单点失败不应拖垮整图
        return _worker_result(subtask, f"子任务失败：{type(exc).__name__}: {exc}", [])


# guard 执行 pending_changes 的固定顺序：写文件类（幂等）在前，命令/Python 在后。
# /retry 断点续跑是 at-least-once：崩溃续跑会重进 guard，已执行的改动可能再执行一次，
# 幂等的写文件先跑可最大限度降低重复执行的副作用。
_EXEC_ORDER = {"plan_write_file": 0, "plan_patch": 0, "plan_run_command": 1, "plan_run_python": 1}


def _execution_order(changes: list[dict]) -> list[dict]:
    """稳定排序：写文件类在前、命令/Python 在后，同组内保持原有顺序。"""
    return sorted(changes, key=lambda c: _EXEC_ORDER.get(c.get("action", ""), 2))


def _execute_approved(changes: list[dict], state: AgentState) -> tuple[str, list[str]]:
    """执行已批准（人工审批或配置放行）的改动：固定顺序 + 执行前快照。
    返回 (summary, changed_files)。"""
    ordered = _execution_order(changes)
    changed_files = [c.get("path", "") for c in ordered if c.get("path")]
    # /undo 快照：必须在 execute_change 之前（备份的是改动前的内容）
    snap_note = ""
    if changed_files and _snapshot_files(changed_files, state.get("thread_id", ""), state["request"]):
        snap_note = "\n\n【快照】改动前文件已备份（/undo 可回退；命令/Python 副作用不可撤）"
    summary = "\n".join(execute_change(c) for c in ordered) + snap_note
    return summary, changed_files


def guard(state: AgentState) -> dict:
    if not state.get("pending_changes"):
        notes = "；".join(state.get("worker_notes", []))
        return {"verdict": "proceed", "pending_changes": [], "feedback": notes,
                "executed_changes": [], "changed_files": []}
    pending = list(state["pending_changes"])
    # v0.7 权限分级：整批都是配置 allow 的类别时跳过 interrupt 直接执行，审计记 auto_allow；
    # 混合批次（含 ask 类别）保守起见整批走人工审批。deny 在暂存层/execute_change 已拦。
    if all(permission_for_action(c.get("action", "")) == "allow" for c in pending):
        cats = "、".join(sorted({ACTION_CATEGORY.get(c.get("action", ""), "?") for c in pending}))
        print(_c(f"[蓝] ⚡ 配置放行（{cats}=allow）：直接执行 {len(pending)} 项改动", _C.DIM))
        _audit_log(state.get("thread_id", ""), {"action": "auto_allow"}, pending)
        ordered = _execution_order(pending)
        summary, changed_files = _execute_approved(ordered, state)
        if changed_files:
            summary += f"\n\n【改动文件】{'; '.join(changed_files)}"
        if state.get("worker_notes"):
            summary += "\n\n【并行子任务】\n" + "\n".join(state["worker_notes"])
        return {"verdict": "approved", "pending_changes": [], "feedback": summary,
                "executed_changes": ordered, "changed_files": changed_files}
    answer = interrupt({"changes": pending, "question": "以上改动/命令是否允许执行？"})
    action = answer.get("action") if isinstance(answer, dict) else "approve"
    if action == "reject":
        return {"verdict": "rejected", "pending_changes": [], "feedback": answer.get("note", "用户拒绝"),
                "executed_changes": [], "changed_files": []}
    if action == "modify":
        return {"verdict": "revise", "pending_changes": [], "feedback": answer.get("note", "用户要求修改"),
                "executed_changes": [], "changed_files": []}
    # 执行前留存完整改动清单：reviewer 看 diff、report 列清单、verifier 读 changed_files。
    # pending_changes 随后清空，不存一份的话下游全拿不到改了什么（P1 实测坑）。
    # 逐条审批：resume 带 indices（0 起）时只执行批准的条目，跳过的记入 feedback。
    indices = answer.get("indices")
    if isinstance(indices, list) and indices:
        idx_set = {int(i) for i in indices}
        executed = [c for i, c in enumerate(pending) if i in idx_set]
        skipped = [c for i, c in enumerate(pending) if i not in idx_set]
    else:
        executed, skipped = pending, []
    ordered = _execution_order(executed)
    summary, changed_files = _execute_approved(ordered, state)
    if skipped:
        skipped_desc = "; ".join(
            f"{c.get('action')}({c.get('path') or c.get('command', '')})" for c in skipped
        )
        summary += f"\n\n【跳过】{len(skipped)} 条未获批准：{skipped_desc}"
    if changed_files:
        summary += f"\n\n【改动文件】{'; '.join(changed_files)}"
    if state.get("worker_notes"):
        summary += "\n\n【并行子任务】\n" + "\n".join(state["worker_notes"])
    return {"verdict": "approved", "pending_changes": [], "feedback": summary,
            "executed_changes": ordered, "changed_files": changed_files}


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


AGENT_MSG_WINDOW = 20  # agent/worker 循环内保留的最近工具交互条数（超出部分压缩成摘要）


def _sliding_compress(messages: list, head_len: int, request: str) -> list:
    """工具循环内的滑动窗口：消息膨胀时把较早轮次压缩成一条摘要。

    revise 回环的压缩（reviewer）管跨轮次，这里管单次循环内：模型连调
    N 次 read_file/grep 后，全量历史会让每次 invoke 的 token 线性膨胀。

    head_len：头部固定不动的条数（system + request + 历史摘要 + tip）。
    完整性约束：按「一轮 = AIMessage + 其 ToolMessage 序列」整组切除，
    不留孤儿 ToolMessage（API 会 400）。旧摘要组并入新摘要，不丢历史。
    """
    overflow = len(messages) - head_len - AGENT_MSG_WINDOW
    if overflow <= 0:
        return messages
    head, body = messages[:head_len], messages[head_len:]
    # 按轮分组：AIMessage 开新组，其余（ToolMessage/旧摘要 HumanMessage）并入当前组
    groups: list[list] = []
    for m in body:
        if isinstance(m, AIMessage):
            groups.append([m])
        elif groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    # 切掉最早若干组，直到覆盖 overflow
    cut, gi = 0, 0
    while gi < len(groups) and cut < overflow:
        cut += len(groups[gi])
        gi += 1
    if gi == 0:
        return messages
    dropped = [m for g in groups[:gi] for m in g]
    # _compress_messages 只提取 AI/Tool 信息；旧摘要（HumanMessage）需手动衔接保留
    old_summary = "\n".join(
        str(m.content) for m in dropped
        if isinstance(m, HumanMessage) and str(m.content).startswith("【早前操作摘要】")
    )
    new_summary = _compress_messages(dropped, request)
    merged = "\n".join(s for s in (old_summary, new_summary) if s)
    kept = [m for g in groups[gi:] for m in g]
    return head + [HumanMessage(content=f"【早前操作摘要】\n{merged}")] + kept


def verifier(state: AgentState) -> dict:
    """审批通过后自动验证：语法检查 + 尝试跑测试。结果供 reviewer 参考。"""
    verdict = state.get("verdict", "proceed")
    if verdict != "approved":
        # 只读/拒绝/修改 不需要验证
        return {"feedback": state.get("feedback", "")}

    feedback = state.get("feedback", "")
    results: list[str] = []

    # 改动文件列表：优先读 guard 写入的 state 字段（v0.5.x 起）；
    # fallback 解析 feedback 文本协议（兼容旧 checkpoint 恢复的场景）
    changed_files: list[str] = list(state.get("changed_files", []))
    if not changed_files and "【改动文件】" in feedback:
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
                    [sys.executable, "-m", "pytest", tf, "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=30, cwd=_resolve(".")
                )
                if r.returncode == 0:
                    results.append(f"✓ {tf} 测试通过")
                else:
                    # 只保留关键错误行（单行截 200 字符防长 assert 展开行）；
                    # 无关键行时 fallback 到输出尾部——否则 reviewer 看到空失败信息
                    err_lines = [l[:200] for l in r.stdout.split("\n")
                                 if "FAILED" in l or "Error" in l or "assert" in l.lower()]
                    if not err_lines:
                        err_lines = [l[:200] for l in r.stdout.strip().split("\n")[-5:]]
                    results.append(f"✗ {tf} 测试失败：{'; '.join(err_lines[:5])}")
            except subprocess.TimeoutExpired:
                results.append(f"? {tf} 测试超时（30s）")
            except Exception as exc:
                results.append(f"? {tf} 测试执行异常：{exc}")

    if results:
        feedback += "\n\n【自动验证结果】\n" + "\n".join(results)
    else:
        feedback += "\n\n【自动验证结果】无 .py 改动或未找到测试文件"

    return {"feedback": feedback}


def _summarize_changes_for_review(changes: list[dict]) -> str:
    """给 reviewer 看的实际改动内容（diff 摘要，总上限 2000 字符）。

    此前 reviewer 只能看 feedback 里的执行结果文本——「测试通过但语义错误」
    的改动拦不住（无测试覆盖的真机场景只剩 py_compile 兜底）。
    """
    if not changes:
        return "（无）"
    parts: list[str] = []
    budget = 2000
    for c in changes:
        head = f"[{c.get('action', '?')}] {c.get('path') or c.get('command') or ''}"
        detail = ""
        if "old" in c or "new" in c:
            detail = f"\n  - old: {str(c.get('old', ''))[:300]}\n  + new: {str(c.get('new', ''))[:300]}"
        elif "content" in c:
            text = c["content"]
            detail = f"\n  content: {text[:300]}{'…' if len(text) > 300 else ''}"
        elif "code" in c:
            code = c["code"]
            detail = f"\n  code: {code[:300]}{'…' if len(code) > 300 else ''}"
        part = head + detail
        if budget - len(part) < 0:
            parts.append("…（其余改动截断）")
            break
        parts.append(part)
        budget -= len(part)
    return "\n".join(parts)


def reviewer(state: AgentState) -> dict:
    rounds = state.get("review_rounds", 0)
    verdict = state.get("verdict", "proceed")
    if verdict == "rejected":
        return {"verdict": "pass", "feedback": f"（用户已拒绝）{state.get('feedback', '')}", "review_rounds": rounds + 1}
    if verdict == "proceed":
        return {"verdict": "pass", "feedback": "本次为只读任务，无需改动。", "review_rounds": rounds + 1}

    resp = _logged_invoke(
        _make_plain_model(),
        [
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(
                content=(
                    f"用户需求：{state['request']}\n"
                    f"评审轮数（当前第 {rounds} 轮，上限 {MAX_REVIEW_ROUNDS}）\n"
                    f"本次执行结果：\n{state.get('feedback', '(无)')}\n"
                    f"实际改动内容：\n{_summarize_changes_for_review(state.get('executed_changes', []))}"
                )
            ),
        ],
        "reviewer",
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
    fb = state.get("feedback", "")
    # 只读 / 拒绝场景模板化，省一次 LLM 调用（~2000 token）。
    # 特征串与 reviewer 固定文案耦合（guard-verifier 的【改动文件】同类约定），改文案两边同步。
    if "本次为只读任务" in fb:
        last_ai = next(
            (m.content for m in reversed(state.get("messages", []))
             if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()),
            "",
        )
        body = f"本次为只读任务，未做任何改动。\n\n**答复**：{last_ai}" if last_ai else "本次为只读任务，未做任何改动。"
        return {"feedback": f"# 交付报告\n\n{body}"}
    if "（用户已拒绝）" in fb:
        return {"feedback": "# 交付报告\n\n改动未通过审批，未执行任何操作。"}
    resp = _logged_invoke(
        _make_plain_model(),
        [
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(
                content=(
                    f"用户需求：{state['request']}\n"
                    f"最终改动清单：\n{_summarize_changes_for_review(state.get('executed_changes', []))}\n"
                    f"评审轮数：{state.get('review_rounds', 0)}\n"
                    f"执行/测试结果：{state.get('feedback', '(无)')}"
                )
            ),
        ],
        "report",
    )
    return {"feedback": resp.content if isinstance(resp.content, str) else str(resp.content)}


def route_by_verdict(state: AgentState) -> Literal["agent", "report"]:
    return "agent" if state.get("verdict") == "revise" else "report"


def route_after_planner(state: AgentState):
    """planner 之后：有可并行子任务（≥2）则 Send 扇出 worker，否则走串行 agent。"""
    tasks = state.get("parallel_tasks") or []
    if len(tasks) >= 2:
        # payload 只传 worker 需要的两个 key：完整 state（含 messages/plan）会放大
        # 每个分支的 checkpoint 序列化开销；worker 的产出经 reducer 聚合不依赖其余字段
        return [
            Send("worker", {"request": state["request"], "current_subtask": t})
            for t in tasks
        ]
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
        # 会话级 token 累计（内存，不落库；重启清零）
        self.token_usage = {"prompt": 0, "completion": 0, "calls": 0}

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
  /undo          回退最近一次审批通过的文件改动（命令/Python 副作用不可撤）
  /retry         从断点续跑上一轮未完成的执行（异常中断/审批点均可续）
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
        if sess.token_usage["calls"]:
            t = sess.token_usage
            print(f"     token 累计：{t['prompt']} + {t['completion']} = {t['prompt'] + t['completion']}（{t['calls']} 次调用）")
        print(f"     图状态：next={list(cur.next) if cur and cur.next else '（已完成）'}")
        if vals.get("plan"):
            print(f"     最近计划：{json.dumps(vals['plan'], ensure_ascii=False)}")
        if vals.get("review_rounds"):
            print(f"     评审轮数：{vals['review_rounds']}")
        return True, None
    if cmd == "/graph":
        print(graph.get_graph().draw_ascii())
        return True, None
    if cmd == "/undo":
        print(_c(f"[蓝] ↩ {_undo_latest(sess.thread_id)}", _C.YELLOW))
        return True, None
    if cmd == "/retry":
        if not resume_pending(graph, sess):
            print("[蓝] 没有可续的断点（上一轮已正常结束）。")
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


def _summarize_output(output: dict) -> str:
    """节点输出压缩成单行摘要，供节点文件日志。"""
    parts: list[str] = []
    if output.get("verdict"):
        parts.append(f"verdict={output['verdict']}")
    if output.get("pending_changes"):
        actions = ",".join(c.get("action", "?") for c in output["pending_changes"])
        parts.append(f"pending={len(output['pending_changes'])}({actions})")
    if output.get("parallel_tasks"):
        parts.append(f"parallel={len(output['parallel_tasks'])}")
    if output.get("worker_notes"):
        parts.append(f"notes={len(output['worker_notes'])}")
    if "review_rounds" in output:
        parts.append(f"rounds={output['review_rounds']}")
    if output.get("feedback"):
        fb = str(output["feedback"]).replace("\n", " ")[:200]
        parts.append(f"feedback={fb!r}")
    return " ".join(parts) or "(empty)"


def _file_log_callback(node_name: str, output: dict) -> None:
    """step 回调：节点输出写文件日志（CLI 启动时挂载）。"""
    _node_logger().info("[%s] %s", node_name, _summarize_output(output))


def _setup_file_logging() -> None:
    """CLI 入口调用：注册节点文件日志回调。测试不调 main()，不写文件。"""
    register_step_callback(_file_log_callback)


# ─────────────────────────── 终端颜色（ANSI，无依赖） ───────────────────────────
# 仅交互 TTY 启用：管道/重定向（benchmark 子进程 capture_output、results/*.log）自动无色。
# 尊重 NO_COLOR 惯例（https://no-color.org/）：设置即禁用。

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class _C:
    """ANSI 颜色码；无颜色环境全部是空串。"""
    BLUE = "\033[94m" if _USE_COLOR else ""        # [蓝] 普通播报
    CYAN = "\033[36m" if _USE_COLOR else ""        # worker / 次级播报
    GREEN = "\033[32m" if _USE_COLOR else ""       # 放行 / 成功
    YELLOW = "\033[33m" if _USE_COLOR else ""      # 待审批 / 打回（需要用户注意）
    RED = "\033[31m" if _USE_COLOR else ""         # 错误
    BRIGHT_CYAN = "\033[96m" if _USE_COLOR else ""  # 用户输入提示符 / 交付报告
    DIM = "\033[2m" if _USE_COLOR else ""          # 自动模式播报
    RESET = "\033[0m" if _USE_COLOR else ""


def _c(text: str, color: str) -> str:
    """着色；无颜色环境原样返回。"""
    return f"{color}{text}{_C.RESET}" if _USE_COLOR else text


# input() 提示符着色：GNU readline / libedit 需要 \001\002 包围不可见字符，
# 否则长输入换行时光标位置算错。无 readline 的环境裸用 ANSI 即可。
try:
    import readline as _readline  # noqa: F401 — 顺带启用行编辑/历史（易用性）
    _P1, _P2 = "\001", "\002"
except ImportError:
    _P1, _P2 = "", ""


def _prompt(text: str, color: str) -> str:
    """input() 用的着色提示符（readline 安全）。"""
    return f"{_P1}{color}{_P2}{text}{_P1}{_C.RESET}{_P2}" if _USE_COLOR else text


def _shown_change(c: dict) -> dict:
    """改动摘要：大字段（content/old/new/code）替换为长度，防长内容刷爆终端。
    _print_pending 播报与 run_round 审批展示共用。"""
    shown = {k: v for k, v in c.items() if k != "action"}
    for big in ("content", "old", "new", "code"):
        if big in shown:
            shown[f"{big}_len"] = len(shown.pop(big))
    return shown


def _preview_lines(text: str, n: int = 5) -> str:
    """长文本预览：前 n 行 + 总行数提示（审批场景：看得见概要，不被刷屏）。"""
    lines = text.split("\n")
    if len(lines) <= n:
        return text
    return "\n".join(lines[:n]) + f"\n  …（共 {len(lines)} 行，按 d 查看全文）"


def _print_change_approval(ci: int, ch: dict) -> None:
    """审批列表的单条改动：给内容预览而非只有长度——审批是安全底线，
    只看 content_len=N 就按 y 等于闭眼放行。"""
    action = ch["action"]
    if action == "plan_run_command":
        # 命令本来就不长，完整显示
        print(_c(f"  {ci}. [{action}] {ch.get('command', '')}", _C.YELLOW))
        return
    print(_c(f"  {ci}. [{action}] {ch.get('path', '')}", _C.YELLOW))
    if action == "plan_patch":
        print(_c(f"     --- old\n{_preview_lines(ch.get('old', ''), 3)}", _C.DIM))
        print(_c(f"     +++ new\n{_preview_lines(ch.get('new', ''), 3)}", _C.DIM))
    elif action == "plan_write_file":
        print(_c(_preview_lines(ch.get("content", "")), _C.DIM))
    elif action == "plan_run_python":
        print(_c(_preview_lines(ch.get("code", ""), 10), _C.DIM))


def _print_changes_full(changes: list[dict]) -> None:
    """[d] 详情：完整打印每条改动的全部内容（用户主动要求，不再截断）。
    rich 可用时升级渲染：patch 红绿 unified diff、写文件/代码语法高亮（超 100 行分页）。"""
    for ci, ch in enumerate(changes, 1):
        print(_c(f"── 改动 {ci} [{ch['action']}] {'─' * 30}", _C.YELLOW))
        if _RICH_CONSOLE is not None and _print_change_rich(ch):
            print()
            continue
        for k, v in ch.items():
            if k != "action":
                print(f"{k}: {v}")
        print()


try:  # rich 是软依赖：缺失时 fallback 纯文本，不影响主流程
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.text import Text
    _RICH_CONSOLE: "Console | None" = Console()
except ImportError:
    _RICH_CONSOLE = None


def _lex_for_path(path: str) -> str:
    """按文件扩展名猜 pygments lexer（语法高亮用）。"""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python", "js": "javascript", "ts": "typescript", "css": "css",
        "html": "html", "json": "json", "md": "markdown", "sh": "bash",
        "toml": "toml", "yaml": "yaml", "yml": "yaml",
    }.get(ext, "text")


def _print_change_rich(ch: dict) -> bool:
    """用 rich 渲染单条改动详情。返回是否已渲染（False 时调用方 fallback 纯文本）。"""
    console = _RICH_CONSOLE
    action = ch["action"]
    if action == "plan_patch":
        import difflib
        old_lines = str(ch.get("old", "")).splitlines(keepends=True)
        new_lines = str(ch.get("new", "")).splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new")
        text = Text()
        for line in diff:
            line = line.rstrip("\n")
            if line.startswith("+") and not line.startswith("+++"):
                text.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                text.append(line + "\n", style="red")
            elif line.startswith("@@"):
                text.append(line + "\n", style="cyan")
            else:
                text.append(line + "\n")
        console.print(f"path: {ch.get('path', '')}")
        console.print(text)
        return True
    if action in ("plan_write_file", "plan_run_python"):
        code = ch.get("content") if action == "plan_write_file" else ch.get("code")
        lexer = _lex_for_path(ch.get("path", "")) if action == "plan_write_file" else "python"
        syntax = Syntax(str(code), lexer, line_numbers=True, word_wrap=True)
        if ch.get("path"):
            console.print(f"path: {ch['path']}")
        if str(code).count("\n") + 1 > 100:
            with console.pager():  # 超 100 行分页（非 tty 时 rich 直接顺序输出，安全）
                console.print(syntax)
        else:
            console.print(syntax)
        return True
    return False


def _print_pending(prefix: str, changes: list[dict]) -> None:
    """打印暂存的改动清单（agent 与 worker 共用）。"""
    for c in changes:
        print(f"{prefix} 已暂存待审批 → {c['action']}({json.dumps(_shown_change(c), ensure_ascii=False)})")


def _print_node(node_name: str, output: dict) -> None:
    if node_name == "planner":
        plan = output.get("plan", [])
        parallel = output.get("parallel_tasks") or []
        if len(parallel) >= 2:
            print(_c(f"[蓝] 拆出 {len(parallel)} 个独立子任务，并行 worker 处理：{json.dumps(parallel, ensure_ascii=False)}", _C.BLUE))
        # planner 条件化：单步且与需求原文一致 → 简单需求直接执行
        elif len(plan) == 1:
            print(_c("[蓝] 简单需求，直接执行", _C.BLUE))
        else:
            print(_c(f"[蓝] 计划：{json.dumps(plan, ensure_ascii=False)}", _C.BLUE))
    elif node_name == "agent" and output.get("pending_changes"):
        _print_pending(_c("[蓝]", _C.BLUE), output["pending_changes"])
    elif node_name == "worker":
        _print_pending(_c("[蓝·worker]", _C.CYAN), output.get("pending_changes", []))
        for note in output.get("worker_notes", []):
            print(_c(f"[蓝·worker] {note}", _C.CYAN))
    elif node_name == "guard":
        print(_c(f"[蓝] 审批结果：{output.get('verdict')}", _C.BLUE))
    elif node_name == "verifier":
        fb = output.get("feedback", "")
        if "【自动验证结果】" in fb:
            # 只打印验证部分，不重复执行结果；✗ 红 ✓ 绿，扫一眼即知
            verify_part = fb.split("【自动验证结果】")[-1].strip()
            colored = [
                _c(ln, _C.RED) if "✗" in ln else _c(ln, _C.GREEN) if "✓" in ln else ln
                for ln in verify_part.split("\n")
            ]
            print(_c("[蓝] 🔍 自动验证：", _C.BLUE) + "\n".join(colored))
    elif node_name == "reviewer":
        passed = output.get("verdict") == "pass"
        mark = "✅ 放行" if passed else "🔪 打回"
        print(_c(f"[评审] {mark}｜{output.get('feedback', '')}", _C.GREEN if passed else _C.YELLOW))
    elif node_name == "report":
        print(_c(f"\n[蓝] {output.get('feedback', '')}", _C.BRIGHT_CYAN))


def _round_cost_str(usage: dict) -> str:
    """按单价配置算本轮成本串；未配置或配置非法返回空串（不影响播报）。"""
    try:
        pi = float(os.environ.get("PRICE_PER_1M_INPUT", "") or 0)
        po = float(os.environ.get("PRICE_PER_1M_OUTPUT", "") or 0)
    except ValueError:
        return ""
    if not (pi or po):
        return ""
    cost = usage["prompt"] * pi / 1e6 + usage["completion"] * po / 1e6
    return f"｜≈ ${cost:.4f}"


AUDIT_LOG = os.path.join(BLUE_DIR, "audit.jsonl")
BACKUP_ROOT = os.path.join(BLUE_DIR, "backups")


def _snapshot_files(files: list[str], thread_id: str, request: str) -> str | None:
    """guard 执行前对改动文件做快照，供 /undo 恢复。返回快照时间戳（无文件则 None）。

    已存在的文件复制内容（undo 写回）；不存在的新文件只记路径（undo 删除）。
    边界：仅 plan_write_file/plan_patch 的目标文件在快照内；
    plan_run_command / plan_run_python 的副作用不可撤。
    """
    files = [f for f in files if f]
    if not files:
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tid = thread_id or "unknown"
    snap_dir = os.path.join(BACKUP_ROOT, tid, ts)
    saved, new_files = [], []
    for rel in files:
        try:
            src = _resolve(rel)
        except ValueError:
            continue
        if src.is_file():
            dst = os.path.join(snap_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            saved.append(rel)
        else:
            new_files.append(rel)
    os.makedirs(snap_dir, exist_ok=True)
    meta = {"ts": ts, "request": request, "files": saved, "new_files": new_files,
            "note": "仅文件改动可撤；plan_run_command/plan_run_python 的副作用不可撤"}
    with open(os.path.join(snap_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BACKUP_ROOT, tid, "latest"), "w", encoding="utf-8") as f:
        f.write(ts)
    return ts


def _undo_latest(thread_id: str) -> str:
    """恢复最近一次快照（单轮 latest 指针）。返回人可读的恢复报告。"""
    tdir = os.path.join(BACKUP_ROOT, thread_id or "unknown")
    latest_file = os.path.join(tdir, "latest")
    if not os.path.isfile(latest_file):
        return "没有可回退的快照（审批通过文件改动时会自动备份）。"
    with open(latest_file, encoding="utf-8") as f:
        ts = f.read().strip()
    snap_dir = os.path.join(tdir, ts)
    meta_path = os.path.join(snap_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return f"快照 {ts} 不完整（缺 meta.json），无法回退。"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    restored, deleted, errors = [], [], []
    for rel in meta.get("files", []):
        try:
            dst = _resolve(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(snap_dir, rel), dst)
            restored.append(rel)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rel}: {exc}")
    for rel in meta.get("new_files", []):
        try:
            p = _resolve(rel)
            if p.exists():
                p.unlink()
                deleted.append(rel)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rel}: {exc}")
    os.remove(latest_file)  # 只撤一轮；快照目录保留在磁盘上可手动翻
    lines = [f"已回退 {ts} 的改动（需求：{meta.get('request', '')[:50]}）"]
    if restored:
        lines.append(f"恢复 {len(restored)} 个文件：{'; '.join(restored)}")
    if deleted:
        lines.append(f"删除 {len(deleted)} 个新建文件：{'; '.join(deleted)}")
    if errors:
        lines.append(f"部分失败：{'; '.join(errors)}")
    lines.append("注意：plan_run_command / plan_run_python 的副作用不在快照内，未回退。")
    return "\n".join(lines)


def _audit_log(thread_id: str, decision: dict, changes: list[dict]) -> None:
    """审批决定追加写审计日志（含拒绝/修改/配置放行；只追加，一行一条 jsonl）。

    人工审批挂在 run_round / run_round_auto 的 resume 处；配置放行（auto_allow）
    由 guard 节点直接写（thread_id 已在 state 里）。写入失败不阻断主流程。
    """
    try:
        record: dict = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "thread": thread_id,
            "action": decision.get("action"),
            "changes": [_shown_change(c) for c in changes],
        }
        if decision.get("indices") is not None:
            record["indices"] = decision["indices"]
        if decision.get("note"):
            record["note"] = decision["note"]
        os.makedirs(BLUE_DIR, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — 审计是旁路，绝不阻断主流程
        pass


def _finish_round_usage(sess: Session) -> None:
    """一轮需求结束：汇总本轮 token 消耗，计入 Session 并播报。"""
    usage = _token_usage_snapshot()
    for k in sess.token_usage:
        sess.token_usage[k] += usage.get(k, 0)
    if usage["calls"]:
        total = usage["prompt"] + usage["completion"]
        sess_total = sess.token_usage["prompt"] + sess.token_usage["completion"]
        print(_c(
            f"[蓝] 📊 本轮 token：{usage['prompt']} + {usage['completion']} = {total}"
            f"（{usage['calls']} 次调用）{_round_cost_str(usage)}｜会话累计 {sess_total}",
            _C.DIM,
        ))


def _drain(graph, config: dict, sess: Session) -> None:
    """审批 drain：处理 guard interrupt 直到图没有后续节点。

    run_round 正常路径与 /retry 断点续跑（resume_pending）共用。
    断在非审批点（异常中断）时直接返回——留给 /retry 续跑，不在此空转
    （此前这里会对无 interrupt 的 cur.next 死循环，v0.7 顺手修掉）。
    """
    while True:
        cur = graph.get_state(config)
        if not cur.next:
            break
        interrupted = [t for t in cur.tasks if t.interrupts]
        if not interrupted:
            break
        for task in interrupted:
            payload = task.interrupts[0].value
            print(_c("\n[蓝] ⏸ 等待你审批：", _C.YELLOW))
            changes = payload.get("changes", [])
            for ci, ch in enumerate(changes, 1):
                _print_change_approval(ci, ch)

            def ask(prompt: str) -> str:
                try:
                    return input(prompt).strip().lower()
                except EOFError:
                    print(_c("\n[蓝] ⏹ 输入流已关闭，按「拒绝」安全中止。", _C.RED))
                    return "n"

            # [d] 详情：看完全文后回到审批提示，不计为决策
            # 序号选批：1,3 → 只批这几条（resume 带 indices，guard 跳过其余）
            resume_val = {"action": "approve"}
            while True:
                choice = ask(_prompt("[y]全批 [n]全拒 [m]意见 [d]详情 [序号]选批 > ", _C.BRIGHT_CYAN))
                if choice == "d":
                    _print_changes_full(changes)
                    continue
                if re.fullmatch(r"[\d,\s]+", choice or "") and choice.strip(" ,"):
                    nums = sorted({int(t) for t in re.split(r"[,\s]+", choice.strip()) if t})
                    if nums and min(nums) >= 1 and max(nums) <= len(changes):
                        resume_val = {"action": "approve", "indices": [n - 1 for n in nums]}
                        break
                    print(_c(f"  序号需在 1~{len(changes)} 之间，请重输。", _C.RED))
                    continue
                break
            if choice == "n":
                resume_val = {"action": "reject", "note": ask(_prompt("  拒绝原因(可空) > ", _C.BRIGHT_CYAN)) or "用户拒绝"}
            elif choice == "m":
                resume_val = {"action": "modify", "note": ask(_prompt("  修改意见 > ", _C.BRIGHT_CYAN))}
            _audit_log(sess.thread_id, resume_val, changes)
            print()
            for chunk in graph.stream(Command(resume=resume_val), config=config, stream_mode="updates"):
                for node_name, output in chunk.items():
                    if node_name == "guard" and output.get("verdict") == "approved":
                        print(_c(f"[蓝] ✅ 已执行：\n{output.get('feedback', '')}", _C.GREEN))
                    else:
                        _emit_step(node_name, output)


def resume_pending(graph, sess: Session) -> bool:
    """/retry 断点续跑：当前 thread 的 checkpoint 有未完成的业务，就从停下的地方继续。

    统一覆盖三场景：①进程内图执行异常中断；②进程重启后 --resume 找回会话发现
    有一轮没跑完；③死在 guard interrupt 审批点（续跑 = 重新弹出审批提示）。
    返回是否有断点可续（False = 上一轮已正常跑完，空操作，绝不默默重跑）。

    at-least-once 语义（已接受）：guard 节点内无中间 checkpoint，崩溃续跑会重进
    guard，已执行的改动可能再执行一次——靠执行顺序（幂等的写文件先跑）+ 审计日志兜底。
    """
    cur = graph.get_state(sess.config)
    if not cur or not cur.next:
        return False
    # at-least-once 提示：崩在 guard 执行途中（guard 待重跑且无 interrupt 挂起）时，
    # 已执行的改动会再执行一次，提醒查审计日志；停在审批点（有 interrupt）时改动
    # 尚未执行，无重复风险不提示。
    if "guard" in cur.next and not any(t.interrupts for t in cur.tasks):
        print(_c("[蓝] ⚠ 上次执行可能已部分完成，重复执行的改动以审计日志为准。", _C.YELLOW))
    print(_c(f"[蓝] 🔁 从断点继续（待执行节点：{list(cur.next)}）…", _C.BLUE))
    try:
        for chunk in graph.stream(None, config=sess.config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)
    except Exception:
        traceback.print_exc()
        _node_logger().exception("resume_pending 续跑异常（thread=%s）", sess.thread_id)
    _drain(graph, sess.config, sess)
    _finish_round_usage(sess)
    _save_session_meta(sess)
    return True


def run_round(graph, sess: Session, request: str) -> None:
    """执行一轮需求：stream 图执行，处理 interrupt 审批。"""
    config = sess.config
    state = initial_state(request)
    state["thread_id"] = sess.thread_id
    _reset_token_usage()
    print(_c(f"[蓝] ★ 第 {sess.next_round()} 轮收到！开始干活。", _C.BLUE))
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)
    except Exception:
        traceback.print_exc()
        _node_logger().exception("run_round 图执行异常（thread=%s）", sess.thread_id)
        print(_c("[蓝] ⚠ 本轮执行中断，可用 /retry 从断点继续。", _C.RED))
    _drain(graph, config, sess)
    _finish_round_usage(sess)
    _save_session_meta(sess)


def run_interactive(graph, request: str | None = None, sess: Session | None = None) -> None:
    """多轮交互主循环：支持连续提需求 + 斜杠命令。"""
    register_step_callback(_print_node)  # 默认回调：CLI 打印
    sess = sess or Session()
    if request:
        run_round(graph, sess, request)
    print(_c("\n[蓝] 进入多轮模式。输入 /help 查看命令，直接输入需求继续干活。", _C.BLUE))
    while True:
        try:
            line = input(_prompt("\n> ", _C.BRIGHT_CYAN)).strip()
        except EOFError:
            print(_c("\n[蓝] 👋 输入流关闭，退出。", _C.BLUE))
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

    _setup_file_logging()
    graph = build_graph()
    if args.show_graph:
        print(graph.get_graph().draw_ascii())
        return
    if args.resume:
        tid = _resume_picker()
        if tid:
            sess = Session(thread_id=tid)
            cur = graph.get_state(sess.config)
            if cur and cur.next:
                # 找回的会话有一轮没跑完：断点续跑（等价 /retry），而非开新一轮
                resume_pending(graph, sess)
            else:
                print(f"[蓝] 恢复 thread {tid}，上轮已完成。输入新需求继续。")
            run_interactive(graph, sess=sess)
            return
        # 用户取消或无效 → 落入新会话
    if args.auto_approve:
        # benchmark 模式：非交互，单轮，guard 自动通过
        if not (args.request or "").strip():
            parser.error("--auto-approve 需要同时提供需求（request），空需求没有可执行内容")
        register_step_callback(_print_node)
        sess = Session()
        final = run_round_auto(graph, sess, args.request or "")
        # CI 退出码：verdict 非 pass，或 verifier 报过 ✗（上限强制放行时 verdict
        # 也是 pass，靠失败标记兜底）。用户主动拒绝=rejected→reviewer pass=exit 0。
        failed = final.get("verdict") != "pass" or "✗" in str(final.get("feedback", ""))
        sys.exit(1 if failed else 0)
    run_interactive(graph, args.request)


def run_round_auto(graph, sess: Session, request: str) -> dict:
    """benchmark 模式：单轮执行，guard 自动 approve 不中断。返回 final state values（CI 退出码判断用）。"""
    config = sess.config
    state = initial_state(request)
    state["thread_id"] = sess.thread_id
    _reset_token_usage()
    print(_c(f"[蓝] ★ benchmark 模式收到：{request}", _C.BLUE))
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)
    except Exception:
        traceback.print_exc()
        _node_logger().exception("run_round_auto 图执行异常（thread=%s）", sess.thread_id)

    # 自动审批所有 interrupt
    while True:
        cur = graph.get_state(config)
        if not cur.next:
            break
        for task in cur.tasks:
            if task.interrupts:
                print(_c("[蓝] ⏸ 自动审批通过", _C.DIM))
                _audit_log(sess.thread_id, {"action": "auto-approve"},
                           task.interrupts[0].value.get("changes", []))
                for chunk in graph.stream(Command(resume={"action": "approve"}), config=config, stream_mode="updates"):
                    for node_name, output in chunk.items():
                        _emit_step(node_name, output)
    _finish_round_usage(sess)
    _save_session_meta(sess)
    return graph.get_state(config).values


if __name__ == "__main__":
    main()