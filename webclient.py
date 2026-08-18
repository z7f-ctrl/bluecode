"""webclient.py — CLI 客户端模式（M2，v0.8.x）：把终端变成 Web 控制台的第二个视图。

架构：CLI 直连模式与 Web 直连执行并存时是「双写者」，强一致要求单写者——客户端
模式下**执行权在 Web 引擎进程**（blue web），CLI 只做三件事：
- 输入 → POST /api/sessions/{tid}/messages（发需求）；
- 事件 → GET /api/sessions/{tid}/events（SSE 节点级播报，与浏览器同源同序）；
- 审批 → POST /api/sessions/{tid}/approvals（y/n/m/d 决策桥到 Web）。

两端信息天然同步：同一执行器、同一事件源、同一 checkpoints.sqlite（design-web.md
§8 的「CLI 与 Web 同会话并发禁止」在此升级为「单写者强一致」）。仅依赖 stdlib
（urllib），与项目零第三方依赖哲学一致；Web 未启动时探测失败，CLI 回落直连模式
（见 agent.main 的 --connect / --local / 自动探测逻辑）。
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8765"  # 与 blue web 默认端口一致


class WebUnavailable(RuntimeError):
    """Web 引擎不可达（未启动/已退出/连接被拒）。"""


# ─────────────────────────── HTTP 薄壳（纯 stdlib） ───────────────────────────


def _open(method: str, base: str, path: str, body: dict | None = None,
          token: str | None = None, timeout: float = 10.0):
    url = base.rstrip("/") + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        raise WebUnavailable(str(exc.reason)) from exc


def _http(method: str, base: str, path: str, body: dict | None = None,
          token: str | None = None, timeout: float = 10.0) -> tuple[int, dict | None]:
    """请求并解析 JSON 响应。HTTP 错误码也返回（body 是错误体），只有网络层失败抛异常。"""
    try:
        resp = _open(method, base, path, body, token, timeout)
        try:
            raw = resp.read().decode("utf-8")
        finally:
            resp.close()
        try:
            return resp.status, json.loads(raw) if raw else None
        except ValueError:
            return resp.status, {"raw": raw[:300]}
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            raw = ""
        try:
            return exc.code, json.loads(raw) if raw else None
        except ValueError:
            return exc.code, {"error": "http_error", "message": raw[:300]}


def probe_web(base: str = DEFAULT_BASE, timeout: float = 0.4) -> dict | None:
    """探测本机 Web 引擎（/api/health）。不可达或非 Blue 服务 → None（静默回落直连）。"""
    try:
        status, data = _http("GET", base, "/api/health", timeout=timeout)
        if status == 200 and isinstance(data, dict) and data.get("version"):
            return data
    except (WebUnavailable, OSError, ValueError):
        pass
    return None


class WebClient:
    """对 design-web.md §4.1 REST + §4.2 SSE 的薄封装（客户端模式专用）。"""

    def __init__(self, base: str, token: str | None = None):
        self.base = base.rstrip("/")
        self.token = token

    def health(self) -> dict:
        status, data = _http("GET", self.base, "/api/health")
        if status != 200 or not isinstance(data, dict):
            raise WebUnavailable(f"health 异常：HTTP {status}")
        return data

    def create_session(self, request: str = "") -> str:
        status, data = _http("POST", self.base, "/api/sessions",
                             {"request": request})
        if status != 200 or not data or not data.get("thread_id"):
            raise WebUnavailable(f"建会话失败：HTTP {status} {data}")
        return data["thread_id"]

    def send(self, tid: str, text: str) -> tuple[int, dict | None]:
        return _http("POST", self.base, f"/api/sessions/{tid}/messages", {"text": text})

    def snapshot(self, tid: str) -> dict:
        status, data = _http("GET", self.base, f"/api/sessions/{tid}")
        return data if isinstance(data, dict) else {}

    def context(self, tid: str) -> dict:
        status, data = _http("GET", self.base, f"/api/sessions/{tid}/context")
        return data if isinstance(data, dict) else {}

    def list_sessions(self) -> list[dict]:
        status, data = _http("GET", self.base, "/api/sessions")
        if status == 200 and isinstance(data, dict):
            return data.get("sessions") or []
        return []

    def approve(self, tid: str, approval_id: str, decision: dict) -> tuple[int, dict | None]:
        return _http("POST", self.base, f"/api/sessions/{tid}/approvals",
                     {"approval_id": approval_id, **decision})

    def undo(self, tid: str) -> tuple[int, dict | None]:
        return _http("POST", self.base, f"/api/sessions/{tid}/undo", {})

    def retry(self, tid: str) -> tuple[int, dict | None]:
        return _http("POST", self.base, f"/api/sessions/{tid}/retry", {})

    def changes(self, tid: str, index: int) -> tuple[int, dict | None]:
        return _http("GET", self.base, f"/api/sessions/{tid}/changes/{index}")

    def events(self, tid: str, last_id: int = 0):
        """SSE 事件流生成器：{id, event, data}。连接被服务端关闭时正常结束（外层重连）。

        行协议解析：SSE 帧 = 若干 `key: value` 行 + 空行。用 readline 而非 read(N)
        ——urllib 的 read(N) 在流式响应上要等满 N 字节才返回（keepalive 每 15s 才
        十几字节，read(4096) 等于挂死），readline 每行一到即回。
        """
        resp = _open("GET", self.base, f"/api/sessions/{tid}/events",
                     token=self.token, timeout=None)
        fields: dict[str, str] = {}
        try:
            while True:
                line = resp.readline()
                if not line:
                    return  # 服务端关闭连接（外层重连）
                line = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:  # 空行 = 一帧结束
                    if fields.get("event") and fields.get("data"):
                        try:
                            yield {
                                "id": int(fields.get("id") or 0),
                                "event": fields["event"],
                                "data": json.loads(fields["data"]),
                            }
                        except (ValueError, json.JSONDecodeError):
                            pass
                    fields = {}
                elif line.startswith(":"):
                    continue  # keepalive 注释行
                elif ":" in line:
                    k, v = line.split(":", 1)
                    fields[k] = v.strip()
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass


# ─────────────────────────── 终端渲染（与 cli._print_node 对齐的瘦身版） ───────────────────────────


def _print_node_client(node: str, data: dict) -> None:
    import cli  # 惰性取用（避免循环导入）
    if node == "planner":
        plan = data.get("plan") or []
        if len(plan) > 1:
            print(cli._c(f"[蓝] 计划：{json.dumps(plan, ensure_ascii=False)}", cli._C.BLUE))
        else:
            print(cli._c("[蓝] 简单需求，直接执行", cli._C.BLUE))
    elif node == "agent":
        pend = data.get("pending_changes") or []
        if pend:
            print(cli._c(f"[蓝] 已暂存 {len(pend)} 条待审批", cli._C.BLUE))
    elif node == "worker":
        pend = data.get("pending_changes") or []
        print(cli._c(f"[蓝·worker] 子任务产出 {len(pend)} 条待审批", cli._C.CYAN))
        for note in data.get("worker_notes") or []:
            print(cli._c(f"[蓝·worker] {note}", cli._C.CYAN))
    elif node == "guard":
        print(cli._c(f"[蓝] 审批结果：{data.get('verdict')}", cli._C.BLUE))
    elif node == "verifier":
        fb = str(data.get("feedback") or "")
        if "【自动验证结果】" in fb:
            verify = fb.split("【自动验证结果】")[-1].strip()
            colored = [
                cli._c(ln, cli._C.RED) if "✗" in ln else cli._c(ln, cli._C.GREEN) if "✓" in ln else ln
                for ln in verify.split("\n")
            ]
            print(cli._c("[蓝] 🔍 自动验证：", cli._C.BLUE) + "\n".join(colored))
    elif node == "reviewer":
        passed = data.get("verdict") == "pass"
        print(cli._c(f"[评审] {'✅ 放行' if passed else '🔪 打回'}｜{str(data.get('feedback', ''))[:200]}",
                     cli._C.GREEN if passed else cli._C.YELLOW))
    elif node == "report":
        print(cli._c(f"\n[蓝] {data.get('feedback', '')}", cli._C.BRIGHT_CYAN))


def _print_approval_card(card: dict) -> None:
    """审批卡渲染（数据来自 SSE approval_required 的 card 预览，规则与 CLI 对齐）。"""
    import cli
    changes = card.get("changes") or []
    print(cli._c(f"\n[蓝] ⏸ 等待你审批（round {card.get('round')}，{card.get('approval_id')}）：",
                 cli._C.YELLOW))
    for i, item in enumerate(changes, 1):
        act = item.get("action", "?")
        print(cli._c(f"  {i}. [{act}] {item.get('path', '')}", cli._C.YELLOW))
        prev = item.get("preview") or {}
        rule = prev.get("rule")
        if rule == "command_full":
            print(cli._c(f"     {prev.get('text', '')}", cli._C.DIM))
        elif rule == "patch_3lines":
            print(cli._c("     --- old\n" + "\n".join(prev.get("old_head") or []), cli._C.DIM))
            print(cli._c("     +++ new\n" + "\n".join(prev.get("new_head") or []), cli._C.DIM))
        elif rule == "write_5lines":
            print(cli._c("\n".join(prev.get("content_head") or []), cli._C.DIM))
        elif rule == "python_10lines":
            print(cli._c("\n".join(prev.get("code_head") or []), cli._C.DIM))


def _print_change_detail(client: WebClient, tid: str, index: int) -> None:
    """[d] 详情：走懒加载端点 changes/{index} 取全文（与 Web 前端同源）。"""
    import cli
    status, data = client.changes(tid, index)
    if status != 200 or not isinstance(data, dict):
        print(cli._c(f"  ⚠ 详情获取失败：{data}", cli._C.RED))
        return
    print(cli._c(f"── 改动 {index + 1} [{data.get('action')}] {'─' * 30}", cli._C.YELLOW))
    if data.get("path"):
        print(f"path: {data['path']}")
    diff = data.get("diff")
    if isinstance(diff, dict) and diff.get("hunks"):
        for h in diff["hunks"]:
            print(h["header"])
            for ln in h["lines"]:
                mark = {"add": "+", "del": "-", "ctx": " ", "meta": "\\"}.get(ln["type"], " ")
                print(f"{mark}{ln['text']}")
    elif data.get("content") is not None:
        print(data["content"])
    elif data.get("old") is not None:
        print("--- old")
        print(data["old"])
        print("+++ new")
        print(data["new"])
    elif data.get("command"):
        print(data["command"])
    print()


def _ask_decision(client: WebClient, tid: str, card: dict) -> dict:
    """审批决策交互（语义与 agent._drain 对齐）：y/n/m/d/序号，返回 guard resume 协议体。"""
    import cli
    changes = card.get("changes") or []
    while True:
        try:
            choice = input(
                cli._prompt("[y]全批 [n]全拒 [m]意见 [d]详情 [序号]选批 > ", cli._C.BRIGHT_CYAN)
            ).strip().lower()
        except EOFError:
            print(cli._c("\n[蓝] ⏹ 输入流已关闭，按「拒绝」安全中止。", cli._C.RED))
            return {"action": "reject", "note": "（输入流关闭，CLI 默认拒绝）"}
        if choice == "d":
            for i in range(len(changes)):
                _print_change_detail(client, tid, i)
            continue
        if re.fullmatch(r"[\d,\s]+", choice or "") and choice.strip(" ,"):
            nums = sorted({int(t) for t in re.split(r"[,\s]+", choice.strip()) if t})
            if nums and min(nums) >= 1 and max(nums) <= len(changes):
                return {"action": "approve", "indices": [n - 1 for n in nums]}
            print(cli._c(f"  序号需在 1~{len(changes)} 之间，请重输。", cli._C.RED))
            continue
        if choice in ("y", ""):
            return {"action": "approve"}
        if choice == "n":
            try:
                note = input(cli._prompt("  拒绝原因(可空) > ", cli._C.BRIGHT_CYAN)).strip()
            except EOFError:
                note = ""
            return {"action": "reject", "note": note or "用户拒绝"}
        if choice == "m":
            try:
                note = input(cli._prompt("  修改意见 > ", cli._C.BRIGHT_CYAN)).strip()
            except EOFError:
                note = ""
            return {"action": "modify", "note": note}
        if choice == "q":
            return {"action": "reject", "note": "用户拒绝"}
        print("  输入 y / n / m / d / 序号（q 拒绝并退出当前审批）")


# ─────────────────────────── 斜杠命令桥（客户端模式） ───────────────────────────

CLIENT_SLASH_HELP = """[蓝] 客户端模式命令（执行权在 Web 引擎）：
  /help           本帮助
  /quit /exit     退出
  /new            开启新会话（Web 端会话列表同步可见）
  /sessions       列出 Web 端会话（用 /use 切换）
  /use N          切换到第 N 个会话
  /history        当前会话快照（轮次/状态/报告/token）
  /context        当前会话上下文构成（压缩摘要/消息尾部/归档记录）
  /status         当前会话状态（busy/待执行节点）
  /undo           撤销上一轮改动
  /retry          断点续跑当前会话
  /model          查看 Web 引擎当前模型（切换请改 ~/.blue/models.toml 或直连模式）
"""


def _handle_client_slash(line: str, client: WebClient, tid_box: dict,
                         consumers: "_Consumers") -> str:
    """处理斜杠命令。返回 "cont"（继续）或 "quit"（退出）。"""
    import cli
    cmd = line.strip().lower()
    if cmd == "/help":
        print(CLIENT_SLASH_HELP)
        return "cont"
    if cmd in ("/quit", "/exit"):
        return "quit"
    if cmd == "/model":
        try:
            h = client.health()
        except WebUnavailable:
            h = {}
        print(cli._c(f"[蓝] Web 引擎模型：{h.get('model', '?')}（v{h.get('version', '?')}）", cli._C.BLUE))
        return "cont"
    if cmd == "/new":
        new_tid = client.create_session()
        consumers.switch(new_tid)
        tid_box["tid"] = new_tid
        print(cli._c(f"[蓝] 新会话 {new_tid}", cli._C.BLUE))
        return "cont"
    if cmd == "/sessions":
        sessions = client.list_sessions()
        if not sessions:
            print("[蓝] 暂无会话。")
            return "cont"
        print("[蓝] Web 端会话（最近在前）：")
        for i, s in enumerate(sessions[:10], 1):
            marker = " 👈 当前" if s["thread_id"] == tid_box["tid"] else ""
            print(f"  {i}. {s['thread_id']}  轮次={s['rounds']}  最后活动={s['last_active']}{marker}")
        return "cont"
    if cmd.startswith("/use"):
        n = cmd[len("/use"):].strip()
        if not n.isdigit():
            print("用法：/use N（/sessions 查看序号）")
            return "cont"
        sessions = client.list_sessions()
        idx = int(n) - 1
        if not (0 <= idx < len(sessions)):
            print("[蓝] 序号无效。")
            return "cont"
        new_tid = sessions[idx]["thread_id"]
        consumers.switch(new_tid)
        tid_box["tid"] = new_tid
        print(cli._c(f"[蓝] 已切换到 {new_tid}", cli._C.BLUE))
        return "cont"
    if cmd == "/history":
        snap = client.snapshot(tid_box["tid"])
        print(cli._c(f"[蓝] 会话 {tid_box['tid']}：round={snap.get('round')} "
                     f"busy={snap.get('busy')}", cli._C.BLUE))
        if snap.get("plan"):
            print(f"  计划：{json.dumps(snap['plan'], ensure_ascii=False)}")
        if snap.get("verdict"):
            print(f"  状态：{snap.get('verdict')}（评审 {snap.get('review_rounds', 0)} 轮）")
        if snap.get("pending_approval"):
            print(f"  ⏸ 待审批：{snap['pending_approval']['approval_id']}（可在 Web 或此处审批）")
        if snap.get("report"):
            print(cli._c(f"\n[蓝] {snap['report']}", cli._C.BRIGHT_CYAN))
        print(f"  token：{json.dumps(snap.get('token_usage', {}), ensure_ascii=False)}")
        return "cont"
    if cmd == "/context":
        data = client.context(tid_box["tid"])
        print(cli._c(f"[蓝] 上下文构成（{tid_box['tid']}）", cli._C.BLUE))
        print(cli._c("  ── ① 压缩摘要 ──", cli._C.DIM))
        for s in data.get("summaries") or []:
            print(f"    {s.split(chr(10))[0]}（共 {len(s)} 字符）")
        print(cli._c("  ── ② 消息尾部 ──", cli._C.DIM))
        for m in data.get("tail") or []:
            print(f"    {m['type']:<12} {m['len']:>6} 字符  {m['first']}")
        print(cli._c("  ── ③ 归档记录 ──", cli._C.DIM))
        for e in data.get("archive") or []:
            print(f"    #{e['cursor']} [{e['source']}] {e['ts']}  "
                  f"{e['summary'].split(chr(10))[0][:80]}")
        return "cont"
    if cmd == "/undo":
        status, data = client.undo(tid_box["tid"])
        if status == 200:
            print(cli._c(f"[蓝] ↩ {data.get('result', '')}", cli._C.YELLOW))
        else:
            print(cli._c(f"[蓝] ⚠ {data.get('message', data)}", cli._C.RED))
        return "cont"
    if cmd == "/retry":
        status, data = client.retry(tid_box["tid"])
        if status == 202:
            print(cli._c("[蓝] 🔁 已提交断点续跑（Web 引擎执行中）", cli._C.BLUE))
        else:
            print(cli._c(f"[蓝] ⚠ {data.get('message', data)}", cli._C.RED))
        return "cont"
    if cmd == "/status":
        snap = client.snapshot(tid_box["tid"])
        print(cli._c(
            f"[蓝] {tid_box['tid']}：busy={snap.get('busy')} "
            f"待执行={snap.get('graph', {}).get('next', [])} "
            f"verdict={snap.get('verdict', '')}", cli._C.BLUE))
        return "cont"
    print(f"未知命令 {cmd}（/help 查看）")
    return "cont"


class _Consumers:
    """SSE 消费线程管理：/new、/use 切换会话时停旧开新（Last-Event-ID 从头重放该会话缓冲）。"""

    def __init__(self, client: WebClient, evq: "queue.Queue[dict]"):
        self.client = client
        self.evq = evq
        self.current: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, tid: str) -> None:
        self._stop.clear()
        self.current = threading.Thread(target=self._run, args=(tid,),
                                        daemon=True, name="blue-client-sse")
        self.current.start()

    def switch(self, tid: str) -> None:
        self._stop.set()  # 停旧线程（其 events() 连接随线程终止关闭）
        self.start(tid)

    def close(self) -> None:
        self._stop.set()

    def _run(self, tid: str) -> None:
        last_id = 0
        while not self._stop.is_set():
            try:
                for ev in self.client.events(tid, last_id):
                    if self._stop.is_set():
                        return
                    last_id = max(last_id, ev["id"])
                    self.evq.put(ev)
            except WebUnavailable:
                if self._stop.is_set():
                    return
                time.sleep(2.0)  # Web 引擎重启/暂不可达：重连（服务端 Last-Event-ID 续传）
            except Exception:  # noqa: BLE001 — 事件流是旁路，断线重连不炸主循环
                if self._stop.is_set():
                    return
                time.sleep(2.0)


# ─────────────────────────── 主循环 ───────────────────────────


def run_client(base: str = DEFAULT_BASE, token: str | None = None,
               request: str | None = None) -> int:
    """客户端模式主循环。request 给定时单发一轮并等 round_end 后退出，否则交互式。"""
    import cli  # 惰性取用（避免循环导入）
    token = token or os.environ.get("BLUE_WEB_TOKEN")
    client = WebClient(base, token)
    try:
        health = client.health()
    except WebUnavailable as exc:
        print(cli._c(
            f"[蓝] ⚠ 无法连接 Web 控制台 {base}（{exc}）。请先运行 `blue web`，"
            f"或用 --local 直连执行。", cli._C.RED))
        return 1
    print(cli._c(
        f"[蓝] 客户端模式（执行引擎：{base}，v{health.get('version', '')}，"
        f"模型：{health.get('model', '?')}）。输入 /help 查看命令。", cli._C.BLUE))
    tid = client.create_session()
    print(cli._c(f"[蓝] 新会话 {tid}（Web 端会话列表同步可见）", cli._C.BLUE))

    evq: "queue.Queue[dict]" = queue.Queue()
    tid_box = {"tid": tid}
    pending_approval: dict = {"card": None}
    done = {"value": False}
    oneshot = bool(request)

    def _drain_events() -> None:
        """处理事件队列：打印播报；审批卡进 pending（主线程决策）；round_end 置 done。"""
        while True:
            try:
                ev = evq.get_nowait()
            except queue.Empty:
                break
            etype, data = ev["event"], ev["data"]
            if etype == "round_start":
                print(cli._c(f"[蓝] ★ 第 {data.get('round')} 轮收到：{data.get('request', '')}",
                             cli._C.BLUE))
            elif etype == "node":
                _print_node_client(data.get("node", ""), data.get("data", {}))
            elif etype == "approval_required":
                pending_approval["card"] = data
                print(cli._c(f"\n[蓝] ⏸ 收到审批卡（{data.get('approval_id')}），等待你决策…",
                             cli._C.YELLOW))
            elif etype == "round_end":
                usage = data.get("usage") or {}
                print(cli._c(
                    f"[蓝] 📊 本轮（{data.get('verdict', '')}）：输入 {usage.get('prompt', 0)} + "
                    f"输出 {usage.get('completion', 0)} = {usage.get('total', 0)}", cli._C.DIM))
                if oneshot:
                    done["value"] = True
            elif etype == "error":
                print(cli._c(f"[蓝] ⚠ {data.get('message', '')}（{data.get('hint', '')}）",
                             cli._C.RED))
                if oneshot:
                    done["value"] = True
            elif etype == "info":
                print(cli._c(f"[蓝] {data.get('message', '')}", cli._C.BLUE))

    def _submit_approval() -> None:
        card = pending_approval["card"]
        _print_approval_card(card)
        decision = _ask_decision(client, tid_box["tid"], card)
        status, data = client.approve(tid_box["tid"], card["approval_id"], decision)
        if status == 404:
            print(cli._c(f"[蓝] 该审批已在 Web 端处理（{data.get('message', '')}）。", cli._C.YELLOW))
        elif status != 200:
            print(cli._c(f"[蓝] ⚠ 审批投递失败：{data.get('message', data)}", cli._C.RED))
        pending_approval["card"] = None

    consumers = _Consumers(client, evq)
    consumers.start(tid)
    try:
        if request:
            status, data = client.send(tid, request)
            if status != 202:
                print(cli._c(f"[蓝] ⚠ 提交失败：{data.get('message', data)}", cli._C.RED))
        while True:
            _drain_events()
            if pending_approval["card"] is not None:
                _submit_approval()
                continue
            if oneshot:
                if done["value"]:
                    break
                time.sleep(0.2)
                continue
            try:
                line = input(cli._prompt("\n> ", cli._C.BRIGHT_CYAN)).strip()
            except EOFError:
                print(cli._c("\n[蓝] 👋 输入流关闭，退出。", cli._C.BLUE))
                break
            if not line:
                continue
            if line.startswith("/"):
                if _handle_client_slash(line, client, tid_box, consumers) == "quit":
                    break
                continue
            status, data = client.send(tid_box["tid"], line)
            if status != 202:
                print(cli._c(f"[蓝] ⚠ 提交失败：{data.get('message', data)}", cli._C.RED))
    finally:
        consumers.close()
    return 0


if __name__ == "__main__":
    sys.exit(run_client())
