"""小蓝 Blue —— 基于 LangGraph 的本地个人 coding agent。v0.1 实现。

图：planner → agent → guard → reviewer →(pass/report | revise/agent)→ report → END

节点职责：
- planner   拆需求为 3~6 步计划
- agent     模型 + 工具循环；写/执行类工具只进 pending_changes，不真执行
- guard     有 pending_changes 时 interrupt() 等人审批；通过则真执行
- reviewer  毒舌自审，输出 pass / revise
- report    收尾汇总

CLI 交互在 __main__ 分支。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 自动加载项目根目录下的 .env（含 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME）。
# .env 已被 .gitignore 排除；显式环境变量优先级高于 .env（load_dotenv 默认不覆盖已存在变量）。
load_dotenv()
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
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


def _make_model():
    kwargs: dict = {"model": os.environ.get("MODEL_NAME", "gpt-4o-mini")}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    return ChatOpenAI(**kwargs).bind_tools(ALL_TOOLS)


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
    messages.extend(state["messages"])
    messages.append(HumanMessage(content=tip))

    updated: list = []
    pending: list[dict] = list(state.get("pending_changes", []))
    step = state.get("current_step", 0)

    while True:
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
                    tool_msg = ToolMessage(
                        content=f"已暂存待审批：{name}({json.dumps(args, ensure_ascii=False)})",
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
    return {"verdict": new_verdict, "feedback": text, "review_rounds": new_rounds}


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


def build_graph():
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
    return builder.compile(checkpointer=MemorySaver())


def _print_node(node_name: str, output: dict) -> None:
    if node_name == "planner":
        print(f"[蓝] 计划：{json.dumps(output.get('plan', []), ensure_ascii=False)}")
    elif node_name == "agent" and output.get("pending_changes"):
        for c in output["pending_changes"]:
            shown = {k: v for k, v in c.items() if k != "action"}
            print(f"[蓝] 已暂存待审批 → {c['action']}({json.dumps(shown, ensure_ascii=False)})")
    elif node_name == "guard":
        print(f"[蓝] 审批结果：{output.get('verdict')}")
    elif node_name == "reviewer":
        mark = "✅ 放行" if output.get("verdict") == "pass" else "🔪 打回"
        print(f"[毒舌评审] {mark}｜{output.get('feedback', '')}")
    elif node_name == "report":
        print(f"\n[蓝] {output.get('feedback', '')}")


def run_interactive(graph, request: str) -> None:
    config = {"configurable": {"thread_id": "blue-single"}}
    state = initial_state(request)
    print("[蓝] ★ 收到！开始干活。")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="小蓝 Blue —— 本地个人 coding agent")
    parser.add_argument("request", nargs="?", default=None, help='要做的事，例如 "给 hello.py 加错误处理并写测试"')
    parser.add_argument("--show-graph", action="store_true", help="打印图拓扑后退出")
    parser.add_argument("--resume", action="store_true", help="恢复上次会话（v0.2 实现）")
    args = parser.parse_args()

    graph = build_graph()
    if args.show_graph:
        print(graph.get_graph().draw_ascii())
        return
    if args.resume:
        print("--resume 为 v0.2 功能，暂未实现。")
        return
    request = args.request or input("> 你要我做什么？\n> ")
    run_interactive(graph, request)


if __name__ == "__main__":
    main()