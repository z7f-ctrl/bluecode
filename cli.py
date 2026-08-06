"""小蓝 Blue —— CLI 交互 + 渲染（斜杠命令 / step 回调 / 颜色 / 审批展示 / 主循环）。

从 agent.py 拆出（#7 模块拆分）。运行时依赖 agent 的节点/辅助函数（_undo_latest /
resume_pending / run_round / _node_logger），通过 `import agent` 在函数内或模块级
惰性取用——agent 在文件末尾才 `from cli import *`，此时 agent 已完整定义，无加载期环。
Session / list_sessions 直接来自 session（无环）。
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from collections.abc import Callable

from session import Session, list_sessions

# 注意：不在模块级 `import agent`，否则 `python agent.py`（此时 agent 是 __main__，
# sys.modules['agent'] 不存在）会让 cli 的 import 加载第二份 agent，触发循环导入。
# 需要 agent 节点/辅助函数的地方改用惰性 `import agent`（函数内），运行时两者均已完整加载。

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
    import agent  # 惰性取用：避免与 agent 的循环导入（见模块 docstring）
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
        print(agent._c(f"[蓝] ↩ {agent._undo_latest(sess.thread_id)}", agent._C.YELLOW))
        return True, None
    if cmd == "/retry":
        if not agent.resume_pending(graph, sess):
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


# ─────────────────────────── step 回调注册机制（借鉴 smolagents step_callbacks） ───────────────────────────
# 节点输出通过回调链处理，默认回调是 CLI 打印。
# 外部（TUI/Web UI/测试）可注册自己的回调，无需修改核心逻辑。
# 回调签名：fn(node_name: str, output: dict) -> None

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
    import agent  # 惰性取用（避免循环导入）
    agent._node_logger().info("[%s] %s", node_name, _summarize_output(output))


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


# ─────────────────────────── 主交互循环 ───────────────────────────

def run_interactive(graph, request: str | None = None, sess: Session | None = None) -> None:
    """多轮交互主循环：支持连续提需求 + 斜杠命令。"""
    import agent  # 惰性取用（避免循环导入）
    register_step_callback(_print_node)  # 默认回调：CLI 打印
    sess = sess or Session()
    if request:
        agent.run_round(graph, sess, request)
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
        agent.run_round(graph, sess, line)


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
