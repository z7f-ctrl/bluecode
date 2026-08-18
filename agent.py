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

# 版本号单一来源：与 pyproject.toml 的 version 对齐；pipx 安装后优先读包元数据覆盖，
# 源码直跑（bluecode 未作为包安装）时回落到此常量。
try:
    from importlib.metadata import version as _pkg_version
    BLUE_VERSION = _pkg_version("bluecode")
except Exception:
    BLUE_VERSION = "0.8.4"

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 配置分层（v0.7 阶段二）：显式 shell 环境变量 > 项目 .env（cwd）> 全局 ~/.blue/.env
# （blue init 写入处，pipx 安装后在任意目录跑都能读到）。load_dotenv 默认不覆盖
# 已存在变量，先加载的优先。.env 已被 .gitignore 排除，绝不入库。
load_dotenv()
load_dotenv(os.path.join(os.path.expanduser("~/.blue"), ".env"))
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
    load_permissions,
    permission_for_action,
)
# 目录常量来自 session（#7 模块拆分）；这里尽早导入供本模块模块级常量定义使用。
from session import (
    BLUE_DIR, DB_PATH, AUDIT_LOG, BACKUP_ROOT, ENV_GLOBAL_PATH, ARCHIVE_DIR,
)

MAX_REVIEW_ROUNDS = 3
MAX_TOOL_ITERATIONS = 15
MAX_WORKER_ITERATIONS = 8   # 并行 worker 的工具循环上限（比串行 agent 小，防限流下耗时失控）
MAX_PARALLEL_WORKERS = 4    # 并行 worker 数量上限（实测 API 限流，不宜更高）
RESUME_STREAM_TIMEOUT = 60.0  # /retry 断点续跑的墙钟超时（秒）：节点崩溃后续跑可能永不返回（忙等占满 CPU），靠超时熔断
ARCHIVE_SNAPSHOT_EVERY = 20  # archive.jsonl 每追加 N 条做一次快照（轻量版本化）
ARCHIVE_KEEP_SNAPSHOTS = 5   # 快照保留份数（超出删最旧，防无限膨胀）
AGENT_PROTECT_ROUNDS = 3     # 滑动窗口活动任务保护：最近 N 轮工具证据永不压缩
QUIET_CONSOLE = False        # blue web REPL 模式置 True：服务端静默，播报由客户端 REPL 从 SSE 打印（防双份）
# 目录常量（BLUE_DIR/DB_PATH/AUDIT_LOG/BACKUP_ROOT/ENV_GLOBAL_PATH/ARCHIVE_DIR）已迁至 session.py，
# 经文件末尾 `from session import *` 重导出，保持 agent.X 可访问（测试 patch 用）。


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
    final_answer: str                # agent 的最终答复全文（纯信息/文档类任务的交付正文，report 直出）


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
        "final_answer": "",
    }


tool_node = ToolNode(READ_ONLY_TOOLS)


_model_cache: ChatOpenAI | None = None
_plain_model_cache: ChatOpenAI | None = None


def _base_kwargs() -> dict:
    """构造 ChatOpenAI 的 kwargs：经多模型注册表解析（models.py），
    未注册时回落纯环境变量旧行为（MODEL_NAME/OPENAI_BASE_URL/OPENAI_API_KEY）。"""
    from models import model_kwargs
    return model_kwargs()


def _reset_model_cache() -> None:
    """清空模型实例缓存：/model 切换激活模型后调用，下次 _make_model 用新配置重建。"""
    global _model_cache, _plain_model_cache
    _model_cache = None
    _plain_model_cache = None


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


# ─────────────────────────── 多模型管理（v0.8） ───────────────────────────
# 注册表 ~/.blue/models.toml + /model 命令（cli 侧）切换激活模型。
# 切换后必须 _reset_model_cache()：模型实例是模块级单例，不重建则切了白切。
# 纯配置逻辑在 models.py；这里只提供带缓存失效的包装（测试 patch "agent.X" 契约）。


def list_models() -> list[dict]:
    """有序模型列表（/model 展示用）。"""
    from models import list_models as _list
    return _list()


def current_model_name() -> str:
    """当前激活模型名。"""
    from models import active_model_name
    return active_model_name()


def set_active_model(name: str) -> str:
    """切换激活模型并清空模型缓存。返回提示文本（成功/失败均人可读）。"""
    from models import set_active_model as _set
    ok, msg = _set(name)
    if ok:
        _reset_model_cache()
        return f"已切换模型 → {msg}"
    return f"切换失败：{msg}"


def active_context_window() -> int:
    """当前激活模型的上下文窗口大小（token）。"""
    from models import context_window
    return context_window()


# ─────────────────────────── token 用量追踪 ───────────────────────────
# _logged_invoke 统一入口无条件累加（带锁，并行 worker 共用）；每轮需求结束
# 打印本轮消耗并计入 Session。单次调用明细见 BLUE_LOG_LLM=1 的 LLM 全文日志。

_TOKEN_LOCK = threading.Lock()
# context = 本轮峰值单次调用 prompt tokens（近似"上下文占用"：一次调用发给
# 模型的输入即当时上下文大小，取峰值反映最接近窗口上限的时刻）
_TOKEN_COLLECTOR = {"prompt": 0, "completion": 0, "calls": 0, "context": 0}


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
        _TOKEN_COLLECTOR["context"] = max(_TOKEN_COLLECTOR["context"], p)


def _reset_token_usage() -> None:
    with _TOKEN_LOCK:
        _TOKEN_COLLECTOR.update(prompt=0, completion=0, calls=0, context=0)


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


HISTORY_MAX_CHARS = 6000  # 跨轮历史（当前需求之前）总字符超此值时压缩成一条摘要


def planner(state: AgentState) -> dict:
    update: dict = {}
    history = list(state.get("messages", []))
    # 跨轮历史压缩（revise 压缩管单轮内，这里管跨轮）：多轮会话下历史无限累积，
    # planner/agent 每次 invoke 都带全量，token 线性膨胀。规则压缩，不调 LLM；
    # 保留末尾的当前需求消息不动，只压它之前的。
    if len(history) > 1 and sum(
        len(str(getattr(m, "content", ""))) for m in history[:-1]
    ) > HISTORY_MAX_CHARS:
        old = history[:-1]
        removals = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
        compressed = _compress_messages(old, state["request"])
        # 摘要落盘归档：跨轮压缩是"不可逆"的隐形操作，落盘后可追溯/回看（/context）
        _append_archive(state.get("thread_id", ""), "cross_round", compressed)
        summary = HumanMessage(content=f"【历史会话摘要】\n{compressed}")
        update["messages"] = removals + [summary]
        history = [summary] + history[-1:]
    # 有历史时 planner 带上历史段（指代连贯性："它/那个文件"靠这个解析）；
    # 无历史（单轮/新会话）prompt 与之前完全一致
    request_text = state["request"]
    if history[:-1]:
        hist_text = "\n".join(
            f"- {str(getattr(m, 'content', ''))[:200]}" for m in history[:-1])
        request_text = f"【历史对话】\n{hist_text}\n\n【当前需求】\n{state['request']}"
    if should_skip_planner(state["request"]):
        return {**update, "plan": [state["request"]], "current_step": 0, "verdict": "proceed",
                "parallel_tasks": [], "worker_notes": [],
                "executed_changes": [], "changed_files": []}
    resp = _logged_invoke(
        _make_plain_model(),
        [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=request_text)],
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
    return {**update, "plan": plan, "current_step": 0, "verdict": "proceed",
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


def _tool_loop_core(messages, *, max_iter, head_len, request,
                    emit, finalize, on_final_answer, caller):
    """agent / worker 共享的工具循环内核（消除两处重复维护，N² 修复只此一份）。

    持有：迭代循环 + _sliding_compress + _logged_invoke + tool_calls 分发
    （final_answer / 计划工具双路径校验 _precheck_plan_tool / 只读 N² 安全取回）
    + 首批 pending 即 break。差异通过回调外置：
      emit(msg)            —— 把消息登记进返回的 state 增量（agent 写 updated；
                              worker 传 None，messages 仅内部累积不返回）。
      finalize(pending, hit_cap) —— 构造返回值 dict（step 推进 / worker note /
                              迭代上限提示等差异）。
      on_final_answer(args, pending) —— final_answer 处理；返回 dict 即提前结束
                              整轮（agent 需 invoke tool_node 取真实结果并写
                              updated；worker 仅取 note），返回 None 表示不提前结束。

    messages 原地累积（不返回）；只读结果的 N² 安全取回（单次 ToolNode.invoke
    后按 tool_call_id 建 map）只在此处维护。
    """
    if emit is None:
        emit = lambda m: None
    pending: list[dict] = []
    iterations = 0
    hit_cap = False
    while iterations < max_iter:
        iterations += 1
        messages[:] = _sliding_compress(messages, head_len, request)
        ai: AIMessage = _logged_invoke(_make_model(), messages, caller)
        messages.append(ai)
        emit(ai)
        if not ai.tool_calls:
            break
        # 只读工具结果：单次 ToolNode.invoke 执行本 AI 消息的全部只读调用，再按
        # tool_call_id 取回——此前在循环里每个 tc 都 invoke 一次，而 ToolNode 每次
        # 都执行 AI 消息里的全部调用，N 个并行调用产生 N² 条重复 ToolMessage（同
        # tool_call_id 重复），被网关按非法请求 400 拒掉（glm 实测）。
        ro_results: dict | None = None
        for tc in ai.tool_calls:
            name, args = tc.get("name"), tc.get("args") or {}
            if name == FINAL_ANSWER_TOOL:
                early = on_final_answer(args, pending)
                if early is not None:
                    return early
                continue
            if name in PLAN_TOOL_NAMES:
                # 与 worker 一致的双路径安全校验（agent 直接搬 args，绕过 @tool 内检查）
                blocked = _precheck_plan_tool(name, args)
                if blocked:
                    tool_msg = ToolMessage(
                        content=f"被拦截：{blocked} 请修正后重试。", tool_call_id=tc["id"])
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
                emit(tool_msg)
            else:
                if ro_results is None:
                    tms = tool_node.invoke(messages)
                    tms = tms if isinstance(tms, list) else tms.get("messages", [])
                    ro_results = {tm.tool_call_id: tm for tm in tms}
                tool_msg = ro_results.get(tc["id"])
                if tool_msg is not None:
                    messages.append(tool_msg)
                    emit(tool_msg)
        # 计划步只在一批改动攒出（即将进入审批）时推进
        if pending:
            break
    else:
        hit_cap = True
    return finalize(pending, hit_cap)


def agent(state: AgentState) -> dict:
    tip = f"计划：{json.dumps(state['plan'], ensure_ascii=False)}；当前第 {state['current_step'] + 1}/{len(state['plan'])} 步。"
    if state.get("verdict") in ("revise", "rejected") and state.get("feedback"):
        tip += f"\n上一轮评审/意见：{state['feedback']}"
    # 当前需求并进 tip——除非 state messages 末尾已是它（run_round 注入的本轮
    # HumanMessage(request)），避免重复；revise 回环时 messages 被压成摘要，
    # 需求只在 tip 里，必须带上
    history = state["messages"]
    tail_is_request = bool(history) and isinstance(history[-1], HumanMessage) \
        and history[-1].content == state["request"]
    if not tail_is_request:
        tip = f"当前需求：{state['request']}\n{tip}"

    # 多轮连贯：state messages 含历史各轮的 Human(需求)+AI/工具交互（跨轮压缩
    # 在 planner 入口做），revise 回环时为 reviewer 压的单条摘要，均直接复用
    messages: list = [SystemMessage(content=AGENT_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=tip))
    head_len = len(messages)  # 头部固定不动；循环内滑动窗口只压缩之后的工具交互

    updated: list = []
    # delta 语义：pending_changes 有 _resettable_add reducer，这里只收集本轮新增，
    # reducer 负责追加到 state；若从 state 复制再返回全量会导致重复累加。
    step = state.get("current_step", 0)

    def _emit(msg) -> None:
        updated.append(msg)

    def _on_final_answer(args, pending) -> dict:
        # 显式终止：只读工具，直接执行并把 answer 作为最终答复
        tool_messages = tool_node.invoke(messages)
        tool_messages = tool_messages if isinstance(tool_messages, list) else tool_messages.get("messages", [])
        for tm in tool_messages:
            messages.append(tm)
            updated.append(tm)
        # 把 final_answer 内容提炼成一条 AIMessage，作为本轮收尾；全文进 state 供 report 直出
        updated.append(AIMessage(content=args.get("answer", "")))
        return {"messages": updated, "pending_changes": pending, "current_step": step,
                "final_answer": args.get("answer", "")}

    def _finalize(pending, hit_cap) -> dict:
        # 计划步只在一批改动攒出（即将进入审批）时推进一次：
        # 此前每次工具迭代都自增，step 语义实际是「迭代计数」，
        # 与 tip 里「当前第 N/M 步」的计划步含义不符
        if pending:
            nonlocal step
            step = min(step + 1, max(len(state.get("plan", [])) or 1, 1))
        if hit_cap:
            # 达到工具迭代上限，强制结束并提示
            updated.append(AIMessage(content=f"（已达工具迭代上限 {MAX_TOOL_ITERATIONS}，强制收尾）"))
        return {"messages": updated, "pending_changes": pending, "current_step": step}

    return _tool_loop_core(
        messages, max_iter=MAX_TOOL_ITERATIONS, head_len=head_len,
        request=state["request"], emit=_emit, finalize=_finalize,
        on_final_answer=_on_final_answer, caller="agent")


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

        def _on_final_answer(args, pending) -> dict:
            # 拿到总结即返回，不再 invoke，无需补 ToolMessage
            note = args.get("answer", "") or "完成"
            return _worker_result(subtask, note, pending)

        def _finalize(pending, hit_cap) -> dict:
            note = f"暂存 {len(pending)} 处改动" if pending else "未产生改动"
            return _worker_result(subtask, note, pending)

        # emit=None：worker 不把消息登记进 state 增量（并行分支消息会交错混乱，
        # 仅内部 messages 累积，产出经 _resettable_add reducer 聚合）
        return _tool_loop_core(
            messages, max_iter=MAX_WORKER_ITERATIONS, head_len=2,
            request=state["request"], emit=None, finalize=_finalize,
            on_final_answer=_on_final_answer, caller="worker")
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
    # 批次入口读一次权限配置，批内复用（避免逐条重读 TOML）；每轮 guard 仍现读，
    # 「运行中改配置即时生效」语义不变。
    perms = load_permissions()
    if all(permission_for_action(c.get("action", ""), perms) == "allow" for c in pending):
        cats = "、".join(sorted({ACTION_CATEGORY.get(c.get("action", ""), "?") for c in pending}))
        if not QUIET_CONSOLE:
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


_archive_cursor: dict[str, int] = {}  # 进程内归档游标缓存（thread_id -> 下一条游标）


def _append_archive(thread_id: str, source: str, summary: str) -> None:
    """压缩摘要落盘 archive.jsonl（append-only、游标递增）+ 每 N 条快照轮转。

    定位：跨轮压缩/revise 摘要从"只活在 messages 里、不可追溯"升级为"落盘可查"
    （nanobot 的 Consolidator 思想）。best-effort：任何失败都吞掉，绝不阻断主流程。
    """
    try:
        if not thread_id or not summary:
            return
        cursor = _archive_cursor.get(thread_id)
        if cursor is None:
            path = os.path.join(ARCHIVE_DIR, f"{thread_id}.jsonl")
            cursor = 1
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                cursor = int(json.loads(line)["cursor"]) + 1
                except Exception:
                    cursor = 1
            _archive_cursor[thread_id] = cursor
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        record = {
            "cursor": cursor, "ts": datetime.now().isoformat(timespec="seconds"),
            "thread_id": thread_id, "source": source, "summary": summary,
        }
        with open(os.path.join(ARCHIVE_DIR, f"{thread_id}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _archive_cursor[thread_id] = cursor + 1
        if cursor % ARCHIVE_SNAPSHOT_EVERY == 0:
            _snapshot_archive(thread_id, cursor)
    except Exception:
        pass  # 归档是辅助设施，失败不影响主流程


def _snapshot_archive(thread_id: str, cursor: int) -> None:
    """快照轮转：每 ARCHIVE_SNAPSHOT_EVERY 条把当前 archive 复制到 backups/，
    保留最近 ARCHIVE_KEEP_SNAPSHOTS 份（nanobot GitStore 的免 git 轻量替代）。"""
    try:
        snap_dir = os.path.join(ARCHIVE_DIR, "backups")
        os.makedirs(snap_dir, exist_ok=True)
        src = os.path.join(ARCHIVE_DIR, f"{thread_id}.jsonl")
        dst = os.path.join(snap_dir, f"{thread_id}-{cursor}.jsonl")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        snaps = sorted(
            (p for p in os.listdir(snap_dir)
             if p.startswith(thread_id + "-") and p.endswith(".jsonl")),
            key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]),
        )
        for old in snaps[:-ARCHIVE_KEEP_SNAPSHOTS]:
            os.unlink(os.path.join(snap_dir, old))
    except Exception:
        pass


def read_archive(thread_id: str, n: int = 5) -> list[dict]:
    """读最近 n 条 archive 记录（/context 展示用；文件缺失/损坏返回空列表）。"""
    path = os.path.join(ARCHIVE_DIR, f"{thread_id}.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        return lines[-n:]
    except Exception:
        return []


def _compress_messages(messages: list, request: str) -> str:
    """把 agent 的完整工具历史压缩成一段人可读摘要，供 revise 回环时替代原始消息。

    不调用 LLM，用规则提取：读了哪些文件、暂存了哪些改动、执行结果。
    """
    read_files: list[str] = []
    planned_changes: list[str] = []
    executed_results: list[str] = []
    user_turns: list[str] = []  # 历史 HumanMessage（各轮需求原文）

    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not text.startswith("【"):  # 跳过摘要/tip 类合成消息
                user_turns.append(text[:100])
        elif isinstance(msg, ToolMessage):
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
    if user_turns:
        # 历史轮次的用户需求（指代连贯性的关键："它/那个文件"指向的对象在这里）
        prior = [t for t in user_turns if t != request]
        if prior:
            parts.append(f"历史需求：{'；'.join(prior[-5:])}")
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
    # 切掉最早若干组，直到覆盖 overflow；活动任务保护：最近 AGENT_PROTECT_ROUNDS
    # 轮（当前工具突发的证据）永不压缩——并行只读爆发（一条 AI 带多个 tool_calls）
    # 会让单轮消息数膨胀、窗口内保留轮数变少，把正在使用的最近几轮也压掉会丢
    # "刚读到的文件内容"（nanobot auto-compact 跳过活动任务的同款思路）。
    max_cut = max(len(groups) - AGENT_PROTECT_ROUNDS, 0)
    cut, gi = 0, 0
    while gi < max_cut and cut < overflow:
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
        # 摘要落盘归档（revise 回环压缩同样不可逆，归档后 /context 可查）
        _append_archive(state.get("thread_id", ""), "revise", compressed)
        # 逐条 RemoveMessage 删除旧历史，再追加单条摘要
        removals = [RemoveMessage(id=m.id) for m in state.get("messages", []) if m.id]
        update["messages"] = removals + [HumanMessage(content=f"【上一轮执行摘要】\n{compressed}")]
    return update


def report(state: AgentState) -> dict:
    fb = state.get("feedback", "")
    # 只读 / 拒绝场景模板化，省一次 LLM 调用（~2000 token）。
    # 特征串与 reviewer 固定文案耦合（guard-verifier 的【改动文件】同类约定），改文案两边同步。
    if "本次为只读任务" in fb:
        # 纯信息/文档类任务的交付正文：优先 final_answer 全文（提示词已要求正文不截断），
        # 兜底取 messages 尾部 last_ai（旧 checkpoint 无 final_answer 字段时兼容）。
        answer = (state.get("final_answer") or "").strip()
        if not answer:
            answer = next(
                (m.content for m in reversed(state.get("messages", []))
                 if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()),
                "",
            )
        body = f"本次为只读任务，未做任何改动。\n\n**答复**：{answer}" if answer else "本次为只读任务，未做任何改动。"
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
                    f"执行/测试结果：{state.get('feedback', '(无)')}\n"
                    f"agent 最终答复：{state.get('final_answer') or '(无)'}"
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


# ── 以下定义已迁至子模块（#7 模块拆分）──
#   Session / _ensure_blue_dir / _get_conn / _save_session_meta / list_sessions / 目录常量 → session.py
#   SLASH_HELP / handle_slash / 渲染与交互 / step 回调机制 → cli.py
#   经文件末尾显式重导出，agent.X 仍可访问（patch("agent.X") 亦生效，validate_graph.py 依赖）。
#
#   注意：step 回调的注册表 _step_callbacks 只在 cli.py 一份（register/clear/_emit_step
#   全由 cli 提供）；agent 的 _drain/_run_graph_core/run_round_auto 调用的 _emit_step
#   即 cli 那份，回调注册（register_step_callback(_print_node)）与触发天然同一份列表。


# ─────────────────────────── 主交互循环 ───────────────────────────


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


def _safe_tid(thread_id: str) -> str | None:
    """校验 thread_id 可安全用于拼备份路径；不安全返回 None（fail-closed）。

    当前 tid 均为 Session 内部生成（blue-<ts>-<hex>）、--resume 走序号选择，
    无外部输入路径；此校验防未来外部来源（如可指定 tid 的入口）带 ../
    或路径分隔符逃逸 BACKUP_ROOT。

    用 Path.resolve + relative_to 校验：解析后的路径必须仍落在 BACKUP_ROOT
    内，既拦 ../ 也拦绝对路径等任何越界写法（比单纯字符过滤更强）。
    """
    tid = thread_id or "unknown"
    try:
        root = Path(BACKUP_ROOT).resolve()
        resolved = (root / tid).resolve()
        resolved.relative_to(root)  # 越界（含 ../、绝对路径）抛 ValueError
    except (ValueError, OSError):
        return None
    return tid


def _snapshot_files(files: list[str], thread_id: str, request: str) -> str | None:
    """guard 执行前对改动文件做快照，供 /undo 恢复。返回快照时间戳（无文件则 None）。

    已存在的文件复制内容（undo 写回）；不存在的新文件只记路径（undo 删除）。
    边界：仅 plan_write_file/plan_patch 的目标文件在快照内；
    plan_run_command / plan_run_python 的副作用不可撤。
    """
    files = [f for f in files if f]
    if not files:
        return None
    tid = _safe_tid(thread_id)
    if tid is None:
        print(_c(f"[蓝] ⚠ thread_id 含非法字符（{thread_id!r}），跳过快照——本轮改动不可 /undo。", _C.RED))
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
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
    tid = _safe_tid(thread_id)
    if tid is None:
        return f"thread_id 含非法字符（{thread_id!r}），拒绝回退。"
    tdir = os.path.join(BACKUP_ROOT, tid)
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


def _context_usage_str(context: int) -> str:
    """上下文占用串：峰值 prompt / 窗口大小（百分比）。窗口取当前激活模型配置，
    未配置回落默认 128k；context 为 0（无调用）时返回空串。"""
    if not context:
        return ""
    window = active_context_window()
    pct = context * 100.0 / window if window else 0.0
    return f"｜上下文 {context:,}/{window:,} ({pct:.1f}%)"


def _finish_round_usage(sess: Session) -> None:
    """一轮需求结束：汇总本轮 token 消耗，计入 Session 并播报。
    context 是峰值状态量：跨轮取 max（历史累积只增），不做加法累计。"""
    usage = _token_usage_snapshot()
    for k in sess.token_usage:
        if k == "context":
            continue
        sess.token_usage[k] += usage.get(k, 0)
    sess.token_usage["context"] = max(sess.token_usage.get("context", 0), usage.get("context", 0))
    if usage["calls"]:
        total = usage["prompt"] + usage["completion"]
        sess_total = sess.token_usage["prompt"] + sess.token_usage["completion"]
        if not QUIET_CONSOLE:
            print(_c(
                f"[蓝] 📊 本轮 token（{current_model_name()}）：输入 {usage['prompt']} + 输出 {usage['completion']} = {total}"
                f"（{usage['calls']} 次调用）{_round_cost_str(usage)}"
                f"{_context_usage_str(usage.get('context', 0))}｜会话累计 {sess_total}",
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


def resume_pending(graph, sess: Session, drain=_drain, holder: str = "cli") -> bool:
    """/retry 断点续跑：当前 thread 的 checkpoint 有未完成的业务，就从停下的地方继续。

    drain 可注入（默认 CLI 的 _drain）：Web 控制台传 web_drain 让审批走浏览器。

    统一覆盖三场景：①进程内图执行异常中断；②进程重启后 --resume 找回会话发现
    有一轮没跑完；③死在 guard interrupt 审批点（续跑 = 重新弹出审批提示）。
    返回是否有断点可续（False = 上一轮已正常跑完，空操作，绝不默默重跑）。

    at-least-once 语义（已接受）：guard 节点内无中间 checkpoint，崩溃续跑会重进
    guard，已执行的改动可能再执行一次——靠执行顺序（幂等的写文件先跑）+ 审计日志兜底。

    熔断（RESUME_STREAM_TIMEOUT）：续跑若在同一节点反复失败（如 planner 持续抛错、
    API 长时间 5xx），graph.stream(None) 可能无进展地重入该节点甚至**永不返回**
    （实测：节点崩溃后续跑，LangGraph 从 checkpoint 反复重驱失败任务、既不抛异常
    也不推进，单次 stream 调用卡死数小时占满 CPU）。故单次续跑套墙钟超时，超时即
    熔断返回、绝不忙等；正常续跑停在 guard interrupt 后交给 _drain 审批。
    """
    # M1 跨进程执行锁（v0.8.x）：续跑同样占用会话（含审批等待期）。跨进程同会话
    # 并发续跑会互相覆盖 checkpoint——锁内串行；锁冲突抛 SessionBusyError 给调用方。
    with exec_lock(sess.thread_id, holder):
        return _resume_pending_locked(graph, sess, drain)


def _resume_pending_locked(graph, sess: Session, drain=_drain) -> bool:
    """resume_pending 的锁内主体（原逻辑不变）。"""
    cur = graph.get_state(sess.config)
    if not cur or not cur.next:
        return False
    # at-least-once 提示：崩在 guard 执行途中（guard 待重跑且无 interrupt 挂起）时，
    # 已执行的改动会再执行一次，提醒查审计日志；停在审批点（有 interrupt）时改动
    # 尚未执行，无重复风险不提示。
    if "guard" in cur.next and not any(t.interrupts for t in cur.tasks):
        print(_c("[蓝] ⚠ 上次执行可能已部分完成，重复执行的改动以审计日志为准。", _C.YELLOW))
    print(_c(f"[蓝] 🔁 从断点继续（待执行节点：{list(cur.next)}）…", _C.BLUE))

    # 单次 stream 续跑 + 熔断超时：正常续跑会停在 guard interrupt（等 _drain 审批），
    # 所以**不能**用「失败重试」循环——在 interrupt 上重复 stream(None) 会挂死。
    # 这里只在「单次 stream 超时不返回」时熔断：节点崩溃后续跑，LangGraph 可能从
    # checkpoint 反复重驱失败任务而永不返回（实测占满 CPU 数小时），信号无法打断，
    # 只能放后台线程跑 + join 超时检测。
    box: dict = {}

    def _run_stream():
        chunks: list = []
        err: list = []
        try:
            for chunk in graph.stream(None, config=sess.config, stream_mode="updates"):
                for node_name, output in chunk.items():
                    chunks.append((node_name, output))
        except BaseException as e:
            err.append(e)
        box["chunks"], box["err"] = chunks, err

    th = threading.Thread(target=_run_stream, daemon=True)
    th.start()
    th.join(timeout=RESUME_STREAM_TIMEOUT)
    timed_out = th.is_alive()
    if timed_out:
        # 不强行杀线程（Python 无法安全强杀），直接熔断返回，让主线程释放；
        # 残余线程随进程退出回收。用户可稍后重试 /retry。
        print(_c(
            f"[蓝] ⏱ 断点续跑 {RESUME_STREAM_TIMEOUT:.0f}s 无进展，已熔断停止（避免忙等占满 CPU）。"
            f"可稍后重试 /retry，或检查 API/网络后重跑。",
            _C.RED,
        ))
        _node_logger().warning("resume_pending 续跑超时熔断（thread=%s）", sess.thread_id)
    else:
        for node_name, output in box.get("chunks", []):
            _emit_step(node_name, output)
        err = box.get("err", [])
        if err:
            traceback.print_exception(type(err[0]), err[0], err[0].__traceback__)
            _node_logger().exception("resume_pending 续跑异常（thread=%s）", sess.thread_id)
    drain(graph, sess.config, sess)
    _finish_round_usage(sess)
    _save_session_meta(sess)
    return True


def _run_graph_core(graph, sess: Session, request: str, *, banner: str, drain,
                    holder: str = "cli") -> None:
    """run_round / run_round_auto 共享的一轮执行骨架（消除两处重复的前 10 行）。

    持有：跨进程执行锁（M1，v0.8.x）→ initial_state 组装 → thread_id → messages
    注入需求（多轮连贯基础）→ token 重置 → 打印 → graph.stream（异常兜底）
    → drain（审批策略注入）→ token 播报 → 会话元信息落库。差异仅审批环节：
    drain 回调封装不同审批策略（人工 input / 自动 approve / Web 桥）。

    holder 区分执行来源（cli / bench / web）：同进程内重入无害（锁可重入），
    跨进程同会话并发执行由锁拦下（SessionBusyError 抛给调用方处理）。
    """
    config = sess.config
    state = initial_state(request)
    state["thread_id"] = sess.thread_id
    # 本轮需求写入 messages（add_messages 追加到历史尾部）——多轮连贯的基础：
    # 此前 request 从不进 messages，新一轮看不到上一轮问过什么，指代全断
    state["messages"] = [HumanMessage(content=request)]
    _reset_token_usage()
    if not QUIET_CONSOLE:
        print(banner)
    with exec_lock(sess.thread_id, holder):
        try:
            for chunk in graph.stream(state, config=config, stream_mode="updates"):
                for node_name, output in chunk.items():
                    _emit_step(node_name, output)
        except Exception:
            traceback.print_exc()
            _node_logger().exception("图执行异常（thread=%s）", sess.thread_id)
            if not QUIET_CONSOLE:
                print(_c("[蓝] ⚠ 本轮执行中断，可用 /retry 从断点继续。", _C.RED))
        drain(graph, config, sess)
        _finish_round_usage(sess)
        _save_session_meta(sess)


def run_round(graph, sess: Session, request: str) -> None:
    """执行一轮需求：stream 图执行，处理 interrupt 审批。"""
    banner = _c(f"[蓝] ★ 第 {sess.next_round()} 轮收到！开始干活。", _C.BLUE)
    _run_graph_core(graph, sess, request, banner=banner, drain=_drain)


# ─────────────────────────── blue init / doctor（v0.7 阶段二） ───────────────────────────
# init：交互式写全局 ~/.blue/.env（key 用 getpass 不回显），写完即跑 doctor 验证。
# doctor：环境/依赖/配置/API 可达/模型存在/tool calling 六项自检——「配错端点/模型
# 裸 traceback」的实测坑（v1h typo、模型名不存在）都在启动时拦下。退出码 0/1，CI 可用。


# ── doctor/init 定义已迁至 doctor.py（见文件末尾 `from doctor import *`）──

def main() -> None:
    # 子命令先行（v0.7 阶段二）：blue init 交互配配置 / blue doctor 自检
    if sys.argv[1:2] == ["init"]:
        sys.exit(cmd_init())
    if sys.argv[1:2] == ["doctor"]:
        sys.exit(cmd_doctor())
    if sys.argv[1:2] == ["web"]:
        # v0.8 Web 控制台：FastAPI + SSE（依赖为可选 extras：pip install bluecode[web]）
        try:
            from web.server import main as web_main
        except ImportError:
            print("[蓝] Web 控制台依赖未安装：pip install bluecode[web]"
                  "（或 pip install fastapi uvicorn）")
            sys.exit(1)
        sys.exit(web_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description="小蓝 Blue —— 本地个人 coding agent")
    parser.add_argument("--version", action="version",
                        version=f"小蓝 Blue {BLUE_VERSION}", help="显示版本号并退出")
    parser.add_argument("request", nargs="?", default=None, help='要做的事，例如 "给 hello.py 加错误处理并写测试"；子命令：init / doctor / web')
    parser.add_argument("--show-graph", action="store_true", help="打印图拓扑后退出")
    parser.add_argument("--resume", action="store_true", help="恢复历史会话")
    parser.add_argument("--auto-approve", action="store_true", help="benchmark 模式：guard 自动审批通过，不中断等待人工")
    parser.add_argument("--connect", default=None, metavar="URL",
                        help="连接运行中的 Web 控制台执行（客户端模式，执行权在 Web 引擎进程；"
                             "默认自动探测本机 8765，--local 跳过）")
    parser.add_argument("--local", action="store_true",
                        help="强制本地直连执行（跳过本机 Web 引擎探测）")
    parser.add_argument("--token", default=None,
                        help="客户端模式访问 Web 的 Bearer token（或环境变量 BLUE_WEB_TOKEN；loopback 模式不需要）")
    args = parser.parse_args()

    # 客户端模式（M2，v0.8.x）：执行权在 Web 引擎进程，CLI 只做输入/事件/审批桥，
    # 两边天然同步（同一事件源）。触发：显式 --connect；或交互式（无 request/
    # resume/auto-approve）且本机 web 在跑且未 --local——「打开 Web 页面后继续在
    # CLI 操作」的默认行为。探测失败静默回落直连，体验与旧版一致。
    web_base = args.connect
    if web_base is None and not (args.local or args.request or args.resume
                                 or args.auto_approve or args.show_graph):
        from webclient import probe_web
        health = probe_web()
        if health is not None:
            web_base = "http://127.0.0.1:8765"
            print(_c(f"[蓝] 检测到 Web 控制台（v{health.get('version', '')}），进入客户端模式；"
                     f"用 --local 可强制直连。", _C.BLUE))
    if web_base is not None:
        if args.auto_approve:
            parser.error("--connect 与 --auto-approve 不能同时使用（客户端模式执行权在 Web 引擎）")
        if args.resume:
            print(_c("[蓝] --resume 在客户端模式不适用：用 /sessions 查看、/use 切换会话。", _C.YELLOW))
        from webclient import run_client
        sys.exit(run_client(web_base, args.token, request=args.request))

    _setup_file_logging()
    graph = build_graph()
    if args.show_graph:
        print(graph.get_graph().draw_ascii())
        return
    if args.resume:
        tid = _resume_picker(graph)
        if tid:
            sess = Session(thread_id=tid)
            cur = graph.get_state(sess.config)
            if cur and cur.next:
                # 找回的会话有一轮没跑完：断点续跑（等价 /retry），而非开新一轮
                try:
                    resume_pending(graph, sess)
                except SessionBusyError as exc:
                    print(_c(f"[蓝] ⏳ {exc}", _C.YELLOW))
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

    def _auto_drain(graph, config, sess) -> None:
        # 自动审批所有 interrupt（benchmark/CI 用；生产代码库勿无人值守跑）
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

    banner = _c(f"[蓝] ★ benchmark 模式收到：{request}", _C.BLUE)
    _run_graph_core(graph, sess, request, banner=banner, drain=_auto_drain, holder="bench")
    return graph.get_state(sess.config).values


# ─────────────────────────── #7 模块拆分：重导出（facade） ───────────────────────────
# session / cli / doctor 的定义迁到子模块；此处显式重导出，使所有 `agent.X` 引用与
# `patch("agent.X")` 继续生效（validate_graph.py 等依赖），零测试改动。
# 注意：必须显式列出（含下划线前缀名），`from x import *` 不会引入 _ 前缀符号。
from session import (  # noqa: E402,F401
    BLUE_DIR, DB_PATH, AUDIT_LOG, BACKUP_ROOT, ENV_GLOBAL_PATH,
    Session, _ensure_blue_dir, _get_conn, _save_session_meta, list_sessions,
    SessionBusyError, peek_exec_lock, acquire_exec_lock, heartbeat_exec_lock,
    release_exec_lock, exec_lock, LOCK_STALE_SECONDS, LOCK_HEARTBEAT_INTERVAL,
)
from cli import (  # noqa: E402,F401
    _C, _c, _prompt, SLASH_HELP, handle_slash, register_step_callback,
    clear_step_callbacks, _emit_step, _summarize_output, _file_log_callback,
    _setup_file_logging, _shown_change, _preview_lines, _print_change_approval,
    _print_changes_full, _RICH_CONSOLE, _lex_for_path, _print_change_rich,
    _print_pending, _print_node, run_interactive, _resume_picker,
)
from doctor import (  # noqa: E402,F401
    _check_python, _check_deps, _check_config, _check_blue_dir, _fetch_model_ids,
    _check_api_and_model, _check_tool_calling, cmd_doctor, _write_env_file, cmd_init,
)
from models import (  # noqa: E402,F401
    MODELS_PATH, DEFAULT_CONTEXT_WINDOW, load_models, active_model_name,
    context_window, list_models as _registry_list_models,
)


if __name__ == "__main__":
    main()