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
]

_DEFAULT_TIMEOUT = 60  # 命令默认超时（秒）


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
    """危险命令校验：命中关键词/管道即抛 ValueError。两条路径都必须调用：
    ① agent 节点暂存 plan_run_command 时（提前拦下）
    ② execute_change 审批后执行前（最后防线）"""
    for pat in _BLOCKED_COMMAND_PATTERNS:
        if pat.search(command):
            raise ValueError(f"命令含危险关键词 {pat.pattern}，已被拦截：{command}")


@tool
def plan_run_command(command: str) -> str:
    """【暂存】计划执行一条 shell 命令。不会真的执行，需要审批。
    有危险关键词（rm -rf / 管道 sudo 等）会在审批前就被拦截。"""
    check_command_safety(command)
    return f"plan_run_command(command={command!r}) 已暂存——等待审批。"


# 只读工具：给 ToolNode 真正执行
READ_ONLY_TOOLS = [list_files, read_file, grep]

# 全部工具：供模型绑定（绑定后才知道有哪些工具可调用）
ALL_TOOLS = READ_ONLY_TOOLS + [plan_write_file, plan_patch, plan_run_command]

# 暂存类工具名集合：agent 节点据此把它们从 pending_changes
PLAN_TOOL_NAMES = {plan_write_file.name, plan_patch.name, plan_run_command.name}


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

        return f"未知动作 {action}"
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return f"执行失败：{exc}"