"""工具集 + 安全校验。

只读工具（list_files / read_file / grep）直接执行，免审批；
写/执行工具（plan_write_file / plan_patch / plan_run_command）——
它们被调用时**不真正执行**，只把结构化参数返回给 agent 节点，
由 agent 节点攒进 `pending_changes`，最终在 guard 审批通过后才执行。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

# 工作目录沙箱：工具只能在这个目录内活动
WORKDIR = Path.cwd()

# 命令执行的危险关键词/组合，命中即拒
_BLOCKED_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+-rf\b"),      # rm -rf
    re.compile(r"\bmkfs\b"),           # 格式化
    re.compile(r">\s*/dev/sd"),        # 直接写裸设备
    re.compile(r"\bshutdown\b"),       # 关机
    re.compile(r"\breboot\b"),         # 重启
    re.compile(r"\brm\s+/\b"),         # 删根
    re.compile(r"\bsudo\b"),           # 提权
    re.compile(r"\|"),                 # shell 管道（高危组合）
    # 复合命令 / 命令替换：ls && rm x、echo $(rm x)、cat `rm x` 会把第二段命令
    # 藏进一条"看起来安全"的命令里（黑名单只查整串关键词、白名单只查命令头，
    # 第二段完全失控）。与管道同级处理：禁复合命令，多步操作拆多次暂存。
    re.compile(r"&&"),
    re.compile(r";"),
    re.compile(r"\$\("),
    re.compile(r"`"),
]

_DEFAULT_TIMEOUT = 60  # 命令默认超时（秒）

# ─────────────────────────── 命令白名单（借鉴 smolagents 白名单安全模型） ───────────
# BLUE_COMMAND_WHITELIST 为空/未设置 → 仅黑名单模式（现状，宽松）
# 设置为逗号分隔的命令名后 → 白名单模式：命令的第一个 token 必须在名单内，否则拦截
# 例：BLUE_COMMAND_WHITELIST="python3,pytest,ls,cat"
_COMMAND_WHITELIST: set[str] = {
    c.strip() for c in os.environ.get("BLUE_COMMAND_WHITELIST", "").split(",") if c.strip()
}


def _command_head(command: str) -> str:
    """取命令第一个 token（去掉常见 env 前缀）。"""
    tokens = command.strip().split()
    # 跳过 VAR=value 前缀和 env 命令
    i = 0
    while i < len(tokens) and ("=" in tokens[i] and not tokens[i].startswith("-")) or tokens[i] == "env":
        i += 1
    return tokens[i] if i < len(tokens) else ""


def _resolve(path: str) -> Path:
    """把相对路径解析到工作目录内，越界直接报错。"""
    p = (WORKDIR / path).resolve()
    if not p.is_relative_to(WORKDIR.resolve()):
        raise ValueError(f"路径越界：{path}（工具只能访问工作目录内）")
    return p


# ─────────────────────────── 只读工具（免审批） ───────────────────────────


@tool
def list_files(dir: str = ".") -> str:
    """列出工作目录内某目录下的条目（目录带 / 后缀）。dir 默认为当前工作目录。"""
    p = _resolve(dir)
    if not p.is_dir():
        return f"错误：{dir} 不是目录"
    entries = []
    for child in sorted(p.iterdir()):
        entries.append(child.name + ("/" if child.is_dir() else ""))
    return "\n".join(entries) if entries else "（空目录）"


@tool
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """带行号读取文件内容。start_line/end_line 限定行区间（1 起），省略则读全部。
    单次最多读取 500 行；如文件更长，请用 start_line/end_line 分段读取。"""
    p = _resolve(path)
    if not p.is_file():
        return f"错误：{path} 不是文件或不存在"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, start_line)
    end = total if end_line is None else min(total, end_line)
    # 默认上限 500 行，防止大文件一次撑爆上下文
    if end_line is None and end - start + 1 > 500:
        end = start + 499
    if start > total:
        return f"（文件共 {total} 行，起始行超出范围）"
    body = "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))
    suffix = f"\n…（共 {total} 行，仅显示 {start}~{end}；请用 start_line/end_line 继续）" if end < total else ""
    return f"{path}（共 {total} 行，显示 {start}~{end} 行）：\n{body}{suffix}"


@tool
def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
    """在工作目录内按正则搜索文本。返回匹配的 文件名:行号:内容，最多 50 条。
    glob 可选，用于限定文件（如 '*.py'）。"""
    p = _resolve(path)
    if not p.exists():
        return f"错误：{path} 不存在"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"错误：无效的正则 {pattern!r}：{exc}"
    results: list[str] = []
    paths = [p] if p.is_file() else sorted(p.rglob("*") if glob is None else p.glob(glob))
    for fp in paths:
        if fp.is_file():
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    results.append(f"{fp.relative_to(WORKDIR)}:{i}:{line.strip()}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n…（已截断，最多显示 50 条）"
    return "\n".join(results) if results else "（无匹配）"


# ─────────────────────────── 写/执行工具（只暂存，不执行） ───────────────────────────


@tool
def plan_write_file(path: str, content: str) -> str:
    """【暂存】计划写入/覆盖一个文件。不会真的执行，只会加入待审批列表。
    content 为完整的新内容。审批通过与后会真正写入。
    注意：如果文件很长（>100 行），优先考虑用 plan_patch 做局部修改，避免重复传输全量内容。"""
    _resolve(path)  # 仅校验路径合法性，不执行
    return f"plan_write_file(path={path}, content_len={len(content)}) 已暂存——等待审批。"


@tool
def plan_patch(path: str, old: str, new: str) -> str:
    """【暂存】计划替换文件中某段文本（old→new）。不会真的执行。
    审批通过后会先在文件中找到唯一的 old 再替换。
    适用于局部修改，比 plan_write_file 更节省 token。"""
    _resolve(path)
    return f"plan_patch(path={path}, old_len={len(old)}, new_len={len(new)}) 已暂存——等待审批。"


def check_command_safety(command: str) -> None:
    """命令安全校验：先黑名单关键词，再（若配置了）白名单。两条路径都必须调用：
    ① agent 节点暂存 plan_run_command 时（提前拦下）
    ② execute_change 审批后执行前（最后防线）"""
    for pat in _BLOCKED_COMMAND_PATTERNS:
        if pat.search(command):
            raise ValueError(f"命令含危险关键词 {pat.pattern}，已被拦截：{command}")
    if _COMMAND_WHITELIST:
        head = _command_head(command)
        if head not in _COMMAND_WHITELIST:
            raise ValueError(
                f"命令 {head!r} 不在白名单内（BLUE_COMMAND_WHITELIST={sorted(_COMMAND_WHITELIST)}），已被拦截：{command}"
            )


@tool
def plan_run_command(command: str) -> str:
    """【暂存】计划执行一条 shell 命令。不会真的执行，需要审批。
    有危险关键词（rm -rf / 管道 sudo 等）会在审批前就被拦截；
    若配置了 BLUE_COMMAND_WHITELIST，则命令头必须在白名单内。"""
    check_command_safety(command)
    return f"plan_run_command(command={command!r}) 已暂存——等待审批。"


# ─────────────────────────── plan_run_python（借鉴 smolagents CodeAgent） ───────────
# 模型直接写 Python 代码作为动作，一次可组合多个工具调用 + 控制流。
# 安全：ast 静态检查（禁 import 白名单外模块 / 禁 dunder 属性访问）+
#       受限 builtins + 操作计数上限，execute_change 执行前二次校验。

import ast
import io
import contextlib

_PYTHON_ALLOWED_IMPORTS = {
    "re", "json", "math", "os", "pathlib", "collections", "itertools",
    "functools", "datetime",
    # subprocess 已移除：subprocess.run("rm x") 可完全绕过 check_command_safety，
    # 要跑命令必须走 plan_run_command 接受完整校验链。os 保留（os.path 等刚需），
    # os.system 理论上同类风险——沙箱是纵深防御一层，最终门槛是 guard 人工审批。
}
_PYTHON_MAX_OPERATIONS = 100_000  # ast 节点数上限，防巨型代码
_PYTHON_TIMEOUT = 30  # 秒


class _OperationCounter(ast.NodeVisitor):
    """统计 ast 节点数，超 _PYTHON_MAX_OPERATIONS 拒绝。"""

    def __init__(self) -> None:
        self.count = 0

    def generic_visit(self, node):
        self.count += 1
        if self.count > _PYTHON_MAX_OPERATIONS:
            raise ValueError(f"代码过大（ast 节点数 > {_PYTHON_MAX_OPERATIONS}）")
        super().generic_visit(node)


def check_python_safety(code: str) -> None:
    """Python 代码静态安全检查：
    - 仅允许 import _PYTHON_ALLOWED_IMPORTS 内的模块
    - 禁止访问 __xxx__ dunder 属性（防沙箱逃逸）
    - ast 节点数上限
    两条路径都必须调用：① agent 暂存时 ② execute_change 执行前。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Python 语法错误：{exc}") from exc
    _OperationCounter().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name.split(".")[0] for a in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            for mod in names:
                if mod and mod not in _PYTHON_ALLOWED_IMPORTS:
                    raise ValueError(f"禁止导入模块 {mod!r}（白名单：{sorted(_PYTHON_ALLOWED_IMPORTS)}）")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError(f"禁止访问 dunder 属性 {node.attr!r}")


@tool
def plan_run_python(code: str) -> str:
    """【暂存】计划在工作目录内执行一段 Python 代码。不会真的执行，需要审批。
    适用：一次组合多个文件操作/数据处理/控制流，比多次 plan_run_command 省轮次。
    限制：仅可导入 re/json/math/os/pathlib/collections/itertools/functools/datetime，
    禁止 __dunder__ 属性访问（含 getattr/setattr 的字符串形式），30s 超时，输出截断 3000 字符。"""
    check_python_safety(code)
    return f"plan_run_python(code_len={len(code)}) 已暂存——等待审批。"


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """受限 __import__：只允许 _PYTHON_ALLOWED_IMPORTS 内的模块。"""
    root = name.split(".")[0]
    if root not in _PYTHON_ALLOWED_IMPORTS:
        raise ImportError(f"禁止导入模块 {root!r}（白名单：{sorted(_PYTHON_ALLOWED_IMPORTS)}）")
    import builtins as _b
    return _b.__import__(name, globals, locals, fromlist, level)


def _safe_getattr(obj, name, *default):
    """getattr 包装：拦截 dunder 字符串参数。

    ast 检查只看 ast.Attribute 节点的属性名，看不到 getattr(obj, '__class__')
    这种字符串参数——经典逃逸链 getattr(getattr(o,'__class__'),'__subclasses__')()
    全靠它。必须在运行时拦。
    """
    if isinstance(name, str) and name.startswith("__"):
        raise ValueError(f"禁止 getattr 访问 dunder 属性 {name!r}")
    import builtins as _b
    return _b.getattr(obj, name, *default)


def _safe_setattr(obj, name, value):
    """setattr 包装：同理拦截 dunder 字符串参数。"""
    if isinstance(name, str) and name.startswith("__"):
        raise ValueError(f"禁止 setattr 访问 dunder 属性 {name!r}")
    import builtins as _b
    return _b.setattr(obj, name, value)


def _restricted_builtins() -> dict:
    """受限 builtins：保留常用安全函数 + 受限 __import__，剔除 open/eval/exec。"""
    import builtins as _b

    safe_names = [
        "abs", "all", "any", "bin", "bool", "chr", "dict", "dir", "divmod",
        "enumerate", "filter", "float", "format", "frozenset",
        "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip", "Exception",
        "ValueError", "TypeError", "KeyError", "IndexError", "StopIteration",
        "True", "False", "None",
    ]
    d = {n: getattr(_b, n) for n in safe_names if hasattr(_b, n)}
    d["__import__"] = _restricted_import
    # getattr/setattr 换成包装版：拦 dunder 字符串参数（ast 检查看不到字符串）
    d["getattr"], d["setattr"] = _safe_getattr, _safe_setattr
    return d


def _execute_python(code: str) -> str:
    """在受限命名空间执行已审批的 Python 代码，捕获 stdout。"""
    check_python_safety(code)  # 最后防线
    namespace: dict = {"__builtins__": _restricted_builtins()}
    # 注入受限的工作目录上下文
    namespace["WORKDIR"] = WORKDIR
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compile(code, "<plan_run_python>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 — 沙箱内异常全部回吐给模型
        return f"执行失败：{type(exc).__name__}: {exc}\nstdout:\n{stdout.getvalue()[-3000:]}"
    out = stdout.getvalue()
    return f"执行成功\n{out[-3000:]}" if out else "执行成功（无输出）"


@tool
def final_answer(answer: str) -> str:
    """任务完成时调用，给出最终答复摘要。调用后本轮 agent 循环立即终止。
    只读工具，免审批。必须在确认任务完成后调用，不要在还有工具未执行时使用。"""
    return f"final_answer: {answer}"


# 只读工具：给 ToolNode 真正执行（final_answer 虽触发终止，但本身无副作用，归只读）
READ_ONLY_TOOLS = [list_files, read_file, grep, final_answer]

# 全部工具：供模型绑定（绑定后才知道有哪些工具可调用）
ALL_TOOLS = READ_ONLY_TOOLS + [plan_write_file, plan_patch, plan_run_command, plan_run_python]

# 暂存类工具名集合：agent 节点据此把它们从 pending_changes
PLAN_TOOL_NAMES = {plan_write_file.name, plan_patch.name, plan_run_command.name, plan_run_python.name}

# 终止类工具名：agent 循环见到即 break
FINAL_ANSWER_TOOL = final_answer.name


# ─────────────────────────── 审批通过后的真正执行 ───────────────────────────


def execute_change(change: dict) -> str:
    """真正执行一条已审批的 pending_change。返回执行结果描述。"""
    action: str = change["action"]
    path: str | None = change.get("path")
    try:
        if action == "plan_write_file":
            if not path:
                raise ValueError("plan_write_file 缺少 path")
            p = _resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(change["content"], encoding="utf-8")
            return f"已写入 {path}（{len(change['content'])} 字符）"

        if action == "plan_patch":
            if not path:
                raise ValueError("plan_patch 缺少 path")
            p = _resolve(path)
            text = p.read_text(encoding="utf-8")
            old, new = change["old"], change["new"]
            if old not in text:
                return f"失败：{path} 中未找到待替换文本"
            if text.count(old) > 1:
                return f"失败：{path} 中待替换文本出现 {text.count(old)} 次，需用 plan_write_file"
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            return f"已补丁 {path}"

        if action == "plan_run_command":
            cmd, _cwd = change["command"], change.get("cwd", ".")
            check_command_safety(cmd)
            run_dir = _resolve(_cwd)  # cwd 同样受沙箱约束，越界直接拒绝
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=_DEFAULT_TIMEOUT, cwd=run_dir,
            )
            out = result.stdout[-3000:] if result.stdout else ""
            err = result.stderr[-3000:] if result.stderr else ""
            if result.returncode != 0:
                return f"命令退出码 {result.returncode}\nstdout:\n{out}\nstderr:\n{err}"
            return f"命令成功（退出码 0）\n{out}\n{err}".strip()

        if action == "plan_run_python":
            return _execute_python(change["code"])

        return f"未知动作 {action}"
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return f"执行失败：{exc}"