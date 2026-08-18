"""web/events.py — SSE 事件层：封装/裁剪、环形缓冲、线程→asyncio 桥、审批卡构造。

协议合同见 design-web.md §4.2 / §5：
- 6 种事件：round_start / node / approval_required / round_end / error / info；
- 每条带递增 id，断线用 Last-Event-ID 从每会话环形缓冲（deque maxlen=500）重放；
- node.data 裁剪：pending_changes 每条只留 shown_change 摘要、feedback 截前 500 字符、
  report 全文放行（与 CLI「防长内容刷屏」同一哲学，全文只经审批卡与懒加载详情端点出）；
- 工作线程 publish → ring buffer 同步追加 + loop.call_soon_threadsafe 推送给 SSE 订阅者。

设计稿中的 approve.py（审批卡构造/预览/diff hunks/决策校验）并入本文件（已定偏差）。
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import threading
from collections import deque
from typing import Any

import cli
import tools

BUFFER_MAXLEN = 500          # 每会话环形缓冲容量（design §8）
FEEDBACK_PREVIEW_CHARS = 500  # node.data 里 feedback 的截断长度（design §4.2）

VALID_ACTIONS = ("approve", "reject", "modify")  # 审批决策合法动作（fail-closed 校验用）


# ─────────────────────────── SSE 封装 ───────────────────────────


def format_sse(ev: dict) -> str:
    """{id, event, data} → SSE 线格式（id:/event:/data:，空行收尾）。"""
    return (
        f"id: {ev['id']}\n"
        f"event: {ev['event']}\n"
        # default=str 兜底：裁剪遗漏的非 JSON 对象降级为 repr，绝不让 SSE 端点 500
        f"data: {json.dumps(ev['data'], ensure_ascii=False, default=str)}\n\n"
    )


def _summarize_messages(messages: list) -> list[dict]:
    """messages 列表（含 AIMessage 等对象，不可 JSON 序列化）→ 紧凑摘要。"""
    out = []
    for m in messages:
        content = getattr(m, "content", m)
        if not isinstance(content, str):
            content = str(content)
        out.append({"type": type(m).__name__, "preview": content[:200]})
    return out


def redact_node_output(node: str, output: Any) -> dict:
    """node 事件数据裁剪（design §4.2）。非 dict 输出按 {text: 截 500} 发。"""
    if not isinstance(output, dict):
        return {"text": str(output)[:FEEDBACK_PREVIEW_CHARS]}
    data = dict(output)
    if data.get("messages"):
        data["messages"] = _summarize_messages(data["messages"])
    if data.get("pending_changes"):
        data["pending_changes"] = [cli.shown_change(c) for c in data["pending_changes"]]
    fb = data.get("feedback")
    if isinstance(fb, str) and len(fb) > FEEDBACK_PREVIEW_CHARS and node != "report":
        # report 全文放行（交付报告是给用户看的正文）；其余节点只给预览
        data["feedback"] = fb[:FEEDBACK_PREVIEW_CHARS]
        data["feedback_len"] = len(fb)
    return data


# ─────────────────────────── 环形缓冲 + asyncio 桥 ───────────────────────────


class SessionBus:
    """单会话事件总线：环形缓冲（重放）+ 订阅者队列（实时推送）。

    publish 可被任意线程调用（工作线程）；推送给 asyncio.Queue 走
    loop.call_soon_threadsafe。loop 在 app startup 时注入（set_loop）；
    未注入时只写缓冲（重放仍可用），不推送。
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: deque[dict] = deque(maxlen=BUFFER_MAXLEN)
        self._next_id = 0
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def publish(self, event: str, data: dict) -> dict:
        """分配递增 id、追加环形缓冲、推送给全部订阅者。返回完整事件。"""
        with self._lock:
            self._next_id += 1
            ev = {"id": self._next_id, "event": event, "data": data}
            self.events.append(ev)
            subs = list(self._subscribers)
            loop = self._loop
        if loop is not None:
            for q in subs:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, ev)
                except Exception:  # noqa: BLE001 — loop 已关闭等，推送失败不阻断执行
                    pass
        return ev

    def subscribe_replay(self, q: asyncio.Queue, last_id: int = 0) -> list[dict]:
        """原子完成「注册订阅 + 取回 last_id 之后的存量」，避免缝隙丢事件。"""
        with self._lock:
            self._subscribers.add(q)
            return [ev for ev in self.events if ev["id"] > last_id]

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def replay(self, last_id: int = 0) -> list[dict]:
        with self._lock:
            return [ev for ev in self.events if ev["id"] > last_id]


# ─────────────────────────── 审批卡构造（预览规则与 CLI 逐条对齐） ───────────────────────────


def _bytes_hint(text: str) -> int:
    return len(text.encode("utf-8"))


def _preview_for_change(c: dict) -> tuple[dict, int]:
    """单条改动的预览块 + bytes_hint。规则逐条对齐 cli._print_change_approval：
    write_file 前 5 行 / patch old·new 各前 3 行 / python 前 10 行 / command 全文。"""
    action = c.get("action", "")
    if action == "plan_run_command":
        cmd = str(c.get("command", ""))
        return {"rule": "command_full", "text": cmd}, _bytes_hint(cmd)
    if action == "plan_patch":
        old, new = str(c.get("old", "")), str(c.get("new", ""))
        return {
            "rule": "patch_3lines",
            "old_head": old.split("\n")[:3],
            "new_head": new.split("\n")[:3],
            "old_lines": old.count("\n") + (1 if old else 0),
            "new_lines": new.count("\n") + (1 if new else 0),
        }, _bytes_hint(old + new)
    if action == "plan_write_file":
        content = str(c.get("content", ""))
        return {
            "rule": "write_5lines",
            "content_head": content.split("\n")[:5],
            "total_lines": content.count("\n") + (1 if content else 0),
        }, _bytes_hint(content)
    if action == "plan_run_python":
        code = str(c.get("code", ""))
        return {
            "rule": "python_10lines",
            "code_head": code.split("\n")[:10],
            "total_lines": code.count("\n") + (1 if code else 0),
        }, _bytes_hint(code)
    return {"rule": "unknown"}, 0


def build_approval_card(round_no: int, approval_id: str, changes: list[dict],
                        perms: dict | None = None) -> dict:
    """interrupt payload 的 changes → approval_required 事件的审批卡（design §4.2 示例）。

    perms 可选传入 load_permissions() 结果（批次内复用，不逐条重读 TOML）。
    """
    if perms is None:
        perms = tools.load_permissions()
    items = []
    for i, c in enumerate(changes):
        action = c.get("action", "")
        preview, bytes_hint = _preview_for_change(c)
        item = {
            "index": i,
            "action": action,
            "category": tools.ACTION_CATEGORY.get(action, "?"),
            "permission": tools.permission_for_action(action, perms),
            "preview": preview,
            "bytes_hint": bytes_hint,
        }
        if c.get("path"):
            item["path"] = c["path"]
        if c.get("command"):
            item["command"] = c["command"]
        items.append(item)
    return {"round": round_no, "approval_id": approval_id, "changes": items}


def validate_decision(body: dict) -> dict:
    """REST 决策体 → guard resume 协议。非法动作抛 ValueError（fail-closed：端点 422）。

    输出照抄 guard 的 resume 协议：{"action": "approve"/"reject"/"modify",
    "indices": [0基, ...]?, "note": str?}（design §5.2 对齐表）。
    """
    action = (body.get("action") or "").strip()
    if action not in VALID_ACTIONS:
        raise ValueError(f"非法审批动作 {action!r}（合法：{'/'.join(VALID_ACTIONS)}）")
    decision: dict[str, Any] = {"action": action}
    indices = body.get("indices")
    if indices is not None:
        if not isinstance(indices, list) or not all(isinstance(i, int) and i >= 0 for i in indices):
            raise ValueError("indices 必须是 0 基非负整数数组")
        decision["indices"] = sorted(set(indices))
    note = body.get("note")
    if note is not None:
        decision["note"] = str(note)
    return decision


# ─────────────────────────── 改动详情（changes/{i} 懒加载，design §6.3） ───────────────────────────


def structured_diff(old: str, new: str, context: int = 3) -> dict:
    """difflib.unified_diff（与 cli._print_change_rich 同算法）→ 结构化 hunks（带行号）。"""
    old_lines, new_lines = str(old).splitlines(), str(new).splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new",
                                lineterm="", n=context)
    hunks: list[dict] = []
    hunk: dict | None = None
    old_no = new_no = 0
    for line in diff:
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            hunk = {
                "header": line,
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or 1),
                "lines": [],
            }
            hunks.append(hunk)
            old_no, new_no = hunk["old_start"], hunk["new_start"]
        elif hunk is not None:
            if line.startswith("+"):
                hunk["lines"].append({"type": "add", "old": None, "new": new_no, "text": line[1:]})
                new_no += 1
            elif line.startswith("-"):
                hunk["lines"].append({"type": "del", "old": old_no, "new": None, "text": line[1:]})
                old_no += 1
            elif line.startswith("\\"):
                hunk["lines"].append({"type": "meta", "old": None, "new": None, "text": line})
            else:
                text = line[1:] if line.startswith(" ") else line
                hunk["lines"].append({"type": "ctx", "old": old_no, "new": new_no, "text": text})
                old_no += 1
                new_no += 1
    return {"hunks": hunks, "old_lines": len(old_lines), "new_lines": len(new_lines)}


def _numbered_lines(text: str) -> list[dict]:
    return [{"no": i, "text": t} for i, t in enumerate(str(text).split("\n"), 1)]


def change_detail(index: int, change: dict) -> dict:
    """单条改动全文 + diff hunks / 带行号代码块（「展开全文」懒加载端点的数据）。

    全文只经审批卡预览与本端点出（design §4.2 的裁剪哲学）。
    """
    action = change.get("action", "")
    detail: dict[str, Any] = {
        "index": index,
        "action": action,
        "category": tools.ACTION_CATEGORY.get(action, "?"),
        "permission": tools.permission_for_action(action),
    }
    if change.get("path"):
        detail["path"] = change["path"]
    if action == "plan_patch":
        detail["old"] = str(change.get("old", ""))
        detail["new"] = str(change.get("new", ""))
        detail["diff"] = structured_diff(detail["old"], detail["new"])
    elif action == "plan_write_file":
        detail["content"] = str(change.get("content", ""))
        detail["lines"] = _numbered_lines(detail["content"])
        detail["total_lines"] = len(detail["lines"])
    elif action == "plan_run_python":
        detail["code"] = str(change.get("code", ""))
        detail["lines"] = _numbered_lines(detail["code"])
        detail["total_lines"] = len(detail["lines"])
    elif action == "plan_run_command":
        detail["command"] = str(change.get("command", ""))
    else:
        detail["change"] = change
    return detail
