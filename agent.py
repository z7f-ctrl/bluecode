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
from langgraph.types import Command, interrupt

from prompts import AGENT_PROMPT, PLANNER_PROMPT, REPORT_PROMPT, REVIEWER_PROMPT
from tools import (
    ALL_TOOLS,
    PLAN_TOOL_NAMES,
    READ_ONLY_TOOLS,
    _resolve,
    check_command_safety,
    execute_change,
)

MAX_REVIEW_ROUNDS = 3
MAX_TOOL_ITERATIONS = 15
BLUE_DIR = os.path.expanduser("~/.blue")
DB_PATH = os.path.join(BLUE_DIR, "checkpoints.sqlite")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    request: str
    plan: list[str]
    current_step: int
    pending_changes: list[dict]
    review_rounds: int
    verdict: str
    feedback: str


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
    }


tool_node = ToolNode(READ_ONLY_TOOLS)


_model_cache: ChatOpenAI | None = None


def _make_model() -> ChatOpenAI:
    global _model_cache
    if _model_cache is None:
        kwargs: dict = {"model": os.environ.get("MODEL_NAME", "gpt-4o-mini")}
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        if os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        _model_cache = ChatOpenAI(**kwargs).bind_tools(ALL_TOOLS)
    return _model_cache


def planner(state: AgentState) -> dict:
    resp = _make_model().invoke(
        [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["request"])]
    )
    content = resp.content
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        plan = parsed if isinstance(parsed, list) else parsed.get("steps")
    except Exception:
        plan = None
    if not isinstance(plan, list) or not all(isinstance(p, str) for p in plan):
        plan = [state["request"]]
    return {"plan": plan, "current_step": 0, "verdict": "proceed"}


def agent(state: AgentState) -> dict:
    tip = f"计划：{json.dumps(state['plan'], ensure_ascii=False)}；当前第 {state['current_step'] + 1}/{len(state['plan'])} 步。"
    if state.get("verdict") in ("revise", "rejected") and state.get("feedback"):
        tip += f"\n上一轮评审/意见：{state['feedback']}"

    messages: list = [SystemMessage(content=AGENT_PROMPT), HumanMessage(content=state["request"])]
    # revise 回环时 state["messages"] 已被 reviewer 压缩成单条摘要，直接复用
    messages.extend(state["messages"])
    messages.append(HumanMessage(content=tip))

    updated: list = []
    pending: list[dict] = list(state.get("pending_changes", []))
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
            if name in PLAN_TOOL_NAMES:
                blocked = None
                if name == "plan_run_command":
                    try:
                        check_command_safety(args.get("command", ""))
                        _resolve(args.get("cwd", "."))
                    except ValueError as exc:
                        blocked = str(exc)
                if blocked:
                    tool_msg = ToolMessage(
                        content=f"命令被拦截：{blocked} 请换一条安全的命令。",
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


def guard(state: AgentState) -> dict:
    if not state.get("pending_changes"):
        return {"verdict": "proceed", "pending_changes": [], "feedback": ""}
    answer = interrupt({"changes": state["pending_changes"], "question": "以上改动/命令是否允许执行？"})
    action = answer.get("action") if isinstance(answer, dict) else "approve"
    if action == "reject":
        return {"verdict": "rejected", "pending_changes": [], "feedback": answer.get("note", "用户拒绝")}
    if action == "modify":
        return {"verdict": "revise", "pending_changes": [], "feedback": answer.get("note", "用户要求修改")}
    summary = "\n".join(execute_change(c) for c in state["pending_changes"])
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
            # 提取 plan_* 工具调用的参数摘要
            for tc in msg.tool_calls or []:
                name = tc.get("name", "")
                if name in ("plan_write_file", "plan_patch", "plan_run_command"):
                    args = tc.get("args", {})
                    if name == "plan_write_file":
                        planned_changes.append(f"plan_write_file(path={args.get('path')}, content_len={len(args.get('content', ''))})")
                    elif name == "plan_patch":
                        planned_changes.append(f"plan_patch(path={args.get('path')}, old_len={len(args.get('old', ''))}, new_len={len(args.get('new', ''))})")
                    elif name == "plan_run_command":
                        planned_changes.append(f"plan_run_command(command={args.get('command')!r})")

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


def reviewer(state: AgentState) -> dict:
    rounds = state.get("review_rounds", 0)
    verdict = state.get("verdict", "proceed")
    if verdict == "rejected":
        return {"verdict": "pass", "feedback": f"（用户已拒绝）{state.get('feedback', '')}", "review_rounds": rounds + 1}
    if verdict == "proceed":
        return {"verdict": "pass", "feedback": "本次为只读任务，无需改动。", "review_rounds": rounds + 1}

    resp = _make_model().invoke(
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
        new_verdict = "revise" if re.search(r"\brevise\b", text) else "pass"
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
    resp = _make_model().invoke(
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
    builder.add_node("guard", guard)
    builder.add_node("reviewer", reviewer)
    builder.add_node("report", report)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("agent", "guard")
    builder.add_edge("guard", "reviewer")
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


def _print_node(node_name: str, output: dict) -> None:
    if node_name == "planner":
        print(f"[蓝] 计划：{json.dumps(output.get('plan', []), ensure_ascii=False)}")
    elif node_name == "agent" and output.get("pending_changes"):
        for c in output["pending_changes"]:
            shown = {k: v for k, v in c.items() if k != "action"}
            # 大字段只显示长度，不打印完整内容
            for big in ("content", "old", "new"):
                if big in shown:
                    shown[f"{big}_len"] = len(shown.pop(big))
            print(f"[蓝] 已暂存待审批 → {c['action']}({json.dumps(shown, ensure_ascii=False)})")
    elif node_name == "guard":
        print(f"[蓝] 审批结果：{output.get('verdict')}")
    elif node_name == "reviewer":
        mark = "✅ 放行" if output.get("verdict") == "pass" else "🔪 打回"
        print(f"[毒舌评审] {mark}｜{output.get('feedback', '')}")
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
                _print_node(node_name, output)
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
                            _print_node(node_name, output)
    _save_session_meta(sess)


def run_interactive(graph, request: str | None = None) -> None:
    """多轮交互主循环：支持连续提需求 + 斜杠命令。"""
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
    run_interactive(graph, args.request)


if __name__ == "__main__":
    main()