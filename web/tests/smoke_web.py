"""web/tests/smoke_web.py — Web 后端冒烟测试（M0–M2）。

风格照 validate_graph.py：python3 直跑（非 pytest）、fake model 离线、断言驱动。
补丁三件套（agent._make_model 与 _make_plain_model 指向**同一个** fake 实例、
patch should_skip_planner）与隔离三件套（DB_PATH/AUDIT_LOG/BACKUP_ROOT 指临时目录）
全部照抄 validate_graph.py 的坑位清单；starlette TestClient 进程内打 app（不起真实端口）。
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

# python3 web/tests/smoke_web.py 直跑时根目录不在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_TMP = tempfile.mkdtemp(prefix="blue-web-smoke-")

import agent
import session as _session_mod
import tools

# 隔离三件套（session 模块运行时读自己的 DB_PATH，两边都要 patch——照抄 validate_graph.py）
agent.DB_PATH = _session_mod.DB_PATH = os.path.join(_TMP, "checkpoints.sqlite")
agent.AUDIT_LOG = os.path.join(_TMP, "audit.jsonl")
agent.BACKUP_ROOT = os.path.join(_TMP, "backups")
agent.LOG_DIR = os.path.join(_TMP, "logs")
# 防真实 ~/.blue/config.toml（若用户配了 write=allow，guard 会跳过 interrupt）污染场景
tools.GLOBAL_CONFIG_PATH = os.path.join(_TMP, "no-such-config.toml")
# 防读真实 ~/.blue/models.toml（v0.7.2 坑位：doctor/init 测试须隔离用户注册表）
import models as _models_mod
_models_mod.MODELS_PATH = os.path.join(_TMP, "no-models.toml")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
import httpx
from starlette.testclient import TestClient

from web import server as web_server
from web.events import change_detail, structured_diff
from web.executor import ExecutorRegistry


class FakeModel:
    """按调用次序返回脚本化响应（照抄 validate_graph.py 的 FakeModel）。"""
    def __init__(self, sequence):
        self.seq = list(sequence)

    def invoke(self, messages, *a, **k):
        item = self.seq.pop(0) if self.seq else "（耗尽）fallback"
        if isinstance(item, Exception):
            raise item
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=item)


def make_app(saver=None, token=None):
    """新 app + registry；graph_factory 注入共享 MemorySaver（测试契约）。"""
    saver = saver or MemorySaver()
    registry = ExecutorRegistry(graph_factory=lambda: agent.build_graph(saver))
    return web_server.create_app(registry=registry, token=token), registry


def model_patches(fake):
    """fake 补丁三件套：两个 make_model 指向同一实例 + should_skip_planner 关掉。"""
    return (patch("agent._make_model", lambda: fake),
            patch("agent._make_plain_model", lambda: fake),
            patch("agent.should_skip_planner", lambda r: False))


def wait_event(rt, name, timeout=20.0):
    """轮询环形缓冲等事件（工作线程在后台跑，fake model 秒回，20s 足够）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ev in list(rt.bus.events):
            if ev["event"] == name:
                return ev
        time.sleep(0.05)
    raise AssertionError(f"超时未等到事件 {name}（现有：{[e['event'] for e in rt.bus.events]}）")


def read_audit():
    with open(agent.AUDIT_LOG, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── SSE 读取：真实 uvicorn 实例（127.0.0.1 临时端口） ──
# 两条 in-process 路都走不通（实测）：
# ① starlette TestClient 的 stream 不能跨线程用（reader 线程里 iter_lines 永远收不到
#   数据、不报错地挂死）；② httpx ASGITransport 会缓冲完整响应体，SSE 无限流永远等
#   不到「响应完成」→ 连响应头都拿不到。任务书允许降级，遂只对 SSE 段起真实 uvicorn
#   （loopback + 临时端口，测完即关）；REST 断言仍走 TestClient。

import uvicorn


class LiveServer:
    """后台线程跑 uvicorn（127.0.0.1:0 临时端口），上下文退出即关停。"""

    def __init__(self, app):
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.port = 0

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("uvicorn 启动超时")
            time.sleep(0.05)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _sse_reader(client, tid, out, stop_on, last_event_id=None):
    """reader 线程：读 SSE 流到 stop_on 事件（httpx.Client 线程安全，可跨线程）。"""
    headers = {}
    if last_event_id is not None:
        headers["Last-Event-ID"] = str(last_event_id)
    cur: dict = {}
    try:
        with client.stream("GET", f"/api/sessions/{tid}/events", headers=headers) as r:
            out.append({"event": "__status__", "id": r.status_code})
            for line in r.iter_lines():
                if line.startswith("id: "):
                    cur["id"] = int(line[4:])
                elif line.startswith("event: "):
                    cur["event"] = line[7:]
                elif line.startswith("data: "):
                    cur["data"] = json.loads(line[6:])
                    out.append(dict(cur))
                    cur = {}
                    if out[-1]["event"] == stop_on:
                        break
    except Exception as exc:  # noqa: BLE001
        out.append({"event": "__reader_error__", "data": f"{type(exc).__name__}: {exc}"})


def _sse_collect(client, rt, tid, stop_on, last_event_id=None, timeout=20.0):
    """起 reader 线程收集事件直到 stop_on；超时报错。"""
    out: list[dict] = []
    t = threading.Thread(target=_sse_reader,
                         args=(client, rt.sess.thread_id, out, stop_on, last_event_id),
                         daemon=True)
    t.start()
    deadline = time.time() + timeout
    while t.is_alive() and time.time() < deadline:
        t.join(timeout=0.1)
    assert not t.is_alive(), \
        f"SSE 读取超时（已收：{[e['event'] for e in out]}）"
    assert not any(e["event"] == "__reader_error__" for e in out), out[-1]
    return [e for e in out if not e["event"].startswith("__")]


def _wait_subscribed(rt, timeout=10.0):
    """等 SSE 端点把订阅者注册进 bus（确保后续 publish 走实时推送而非只靠重放）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rt.bus._subscribers:
            return
        time.sleep(0.05)
    raise AssertionError("SSE 订阅未建立")


# ─────────────────────────── 场景 ───────────────────────────


def test_readonly_round():
    """只读轮：round_start → node 序列 → round_end（无审批）。"""
    calls = [
        AIMessage(content='["读文件", "报告"]'),
        AIMessage(content="先 grep 一下", tool_calls=[
            {"name": "grep", "args": {"pattern": "小蓝"}, "id": "t1"}]),
        AIMessage(content="统计完成（2 处匹配）。最终答复：完成。"),
    ]
    p1, p2, p3 = model_patches(FakeModel(calls))
    with p1, p2, p3:
        app, registry = make_app()
        with TestClient(app) as client:
            r = client.post("/api/sessions", json={"request": "统计项目里的中文词"})
            assert r.status_code == 200, r.text
            tid = r.json()["thread_id"]
            assert r.json()["started"] is True
            rt = registry.get_or_create(tid)
            end = wait_event(rt, "round_end")
            names = [e["event"] for e in rt.bus.events]
            assert names[0] == "round_start", names
            node_names = [e["data"]["node"] for e in rt.bus.events if e["event"] == "node"]
            assert node_names[:2] == ["planner", "agent"], node_names
            assert "report" in node_names, node_names
            assert not any(e["event"] == "approval_required" for e in rt.bus.events), "只读不应有审批"
            assert end["data"]["verdict"] == "pass", end
            assert end["data"]["usage"]["calls"] >= 1, end["data"]["usage"]
            assert "session_total" in end["data"]
    print("PASS 只读轮出 round_end（事件序列 round_start→node×N→round_end）✔\n")


def test_write_approve_flow():
    """写轮全链路：暂存 → approval_required → 409 忙拒 → 详情端点 → approve → 真写入
    → 审计 source:web → SSE 重连重放。"""
    scratch = "__web_smoke_write__.txt"
    content = "web 冒烟第一行\n第二行\n第三行\n第四行\n第五行\n第六行\n"
    calls = [
        AIMessage(content='["写 scratch"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file",
             "args": {"path": scratch, "content": content}, "id": "w1"}]),
        AIMessage(content="verdict: pass\nfeedback: 简单改动，放行。"),
        AIMessage(content="# 报告\n改动完成。"),
    ]
    p1, p2, p3 = model_patches(FakeModel(calls))
    try:
        with p1, p2, p3:
            app, registry = make_app()
            with TestClient(app) as client:
                r = client.post("/api/sessions", json={"request": "把内容写入 scratch"})
                tid = r.json()["thread_id"]
                rt = registry.get_or_create(tid)
                ap = wait_event(rt, "approval_required")
                aid = ap["data"]["approval_id"]
                assert aid.startswith(f"{tid}:guard:"), aid
                # 审批卡：预览规则与 CLI 对齐（write_file 前 5 行）、category/permission
                ch = ap["data"]["changes"]
                assert len(ch) == 1 and ch[0]["action"] == "plan_write_file", ch
                assert ch[0]["category"] == "write" and ch[0]["permission"] == "ask", ch
                assert ch[0]["path"] == scratch
                assert ch[0]["preview"]["rule"] == "write_5lines", ch[0]["preview"]
                assert ch[0]["preview"]["content_head"] == content.split("\n")[:5]
                assert ch[0]["bytes_hint"] > 0
                # 审批挂起时会话忙：再发消息 → 409 round_running
                r = client.post(f"/api/sessions/{tid}/messages", json={"text": "再来一轮"})
                assert r.status_code == 409 and r.json()["error"] == "round_running", r.text
                # 未知 approval_id → 404（fail-closed，绝不默认放行）
                r = client.post(f"/api/sessions/{tid}/approvals",
                                json={"approval_id": "bogus", "action": "approve"})
                assert r.status_code == 404 and r.json()["error"] == "unknown_approval_id", r.text
                # 详情端点（懒加载全文 + 行号）
                r = client.get(f"/api/sessions/{tid}/changes/0")
                assert r.status_code == 200, r.text
                detail = r.json()
                assert detail["content"] == content and detail["total_lines"] == 7, detail
                assert detail["lines"][0] == {"no": 1, "text": "web 冒烟第一行"}
            # SSE 段：实时推送 + 重连重放（起真实 uvicorn，见 LiveServer 注释；
            # REST 断言仍走 TestClient，两边共享同一 app/registry）
            snap = _sse_live_and_replay(app, rt, aid)
            assert snap["round"] == 1 and snap["verdict"] == "pass", snap
            assert snap["pending_approval"] is None
            assert snap["graph"]["next"] == []
            # 文件真写入
            assert Path(scratch).read_text(encoding="utf-8") == content, "审批通过后应真写入"
            # 审计落盘含 source:web
            entries = read_audit()
            assert any(e.get("action") == "approve" and e.get("source") == "web"
                       and e.get("thread") == tid for e in entries), entries
    finally:
        Path(scratch).unlink(missing_ok=True)
    print("PASS 写轮：暂存→审批卡→409→详情→approve→真写入→审计 source:web→SSE 重放 ✔\n")


def _sse_live_and_replay(app, rt, aid):
    """SSE 实时推送（订阅后 approve，活到 round_end）+ Last-Event-ID 重放 + 快照。"""
    tid = rt.sess.thread_id
    with LiveServer(app) as live, \
            httpx.Client(base_url=f"http://127.0.0.1:{live.port}", timeout=30) as h:
        # 实时推送：先订阅（重放应立刻给出 approval_required），再 approve
        out: list[dict] = []
        t = threading.Thread(target=_sse_reader, args=(h, tid, out, "round_end"),
                             daemon=True)
        t.start()
        _wait_subscribed(rt)
        max_id_before = max(e["id"] for e in rt.bus.events)
        r = h.post(f"/api/sessions/{tid}/approvals",
                   json={"approval_id": aid, "action": "approve"})
        assert r.status_code == 200 and r.json()["mode"] == "delivered", r.text
        t.join(timeout=20)
        assert not t.is_alive(), f"SSE 未实时收到 round_end：{[e['event'] for e in out]}"
        events = [e for e in out if not e["event"].startswith("__")]
        names = [e["event"] for e in events]
        assert "approval_required" in names, f"SSE 未重放到审批卡：{names}"
        assert names[-1] == "round_end", names
        assert events[-1]["data"]["verdict"] == "pass"
        # 实时推送确实走了订阅通道：round_end 的 id 大于 approve 前已发事件的最大 id
        assert events[-1]["id"] > max_id_before
        # 重连重放：Last-Event-ID 过滤后仍能拿到 round_end
        first_id = rt.bus.events[0]["id"]
        replayed = _sse_collect(h, rt, tid, "round_end", last_event_id=first_id)
        assert replayed, "重放为空"
        assert all(e["id"] > first_id for e in replayed), "Last-Event-ID 过滤失效"
        assert replayed[-1]["event"] == "round_end"
        # 会话快照：轮次与报告
        r = h.get(f"/api/sessions/{tid}")
        assert r.status_code == 200, r.text
        return r.json()


def test_reject_and_indices():
    """reject 不写入；indices [0] 只批一条；modify 路径动作校验。"""
    f1, f2 = "__web_smoke_idx1__.txt", "__web_smoke_idx2__.txt"
    frej = "__web_smoke_reject__.txt"
    p1, p2, p3 = model_patches(FakeModel([
        # 场景一：reject
        AIMessage(content='["写文件"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": frej, "content": "x"}, "id": "r1"}]),
        # 场景二：两条改动 indices [0] 选批
        AIMessage(content='["写两个文件"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": f1, "content": "一"}, "id": "i1"},
            {"name": "plan_write_file", "args": {"path": f2, "content": "二"}, "id": "i2"}]),
        AIMessage(content="verdict: pass\nfeedback: 放行。"),
        AIMessage(content="# 报告\n完成。"),
    ]))
    try:
        with p1, p2, p3:
            app, registry = make_app()
            with TestClient(app) as client:
                # 场景一：reject
                tid = client.post("/api/sessions", json={"request": "写文件"}).json()["thread_id"]
                rt = registry.get_or_create(tid)
                ap = wait_event(rt, "approval_required")
                r = client.post(f"/api/sessions/{tid}/approvals",
                                json={"approval_id": ap["data"]["approval_id"],
                                      "action": "reject", "note": "不需要"})
                assert r.status_code == 200, r.text
                end = wait_event(rt, "round_end")
                assert end["data"]["verdict"] == "pass"
                assert not Path(frej).exists(), "reject 不应写入"
                entries = read_audit()
                assert any(e.get("action") == "reject" and e.get("note") == "不需要"
                           and e.get("source") == "web" for e in entries), entries

                # 场景二：indices [0] 只批第一条
                tid = client.post("/api/sessions",
                                  json={"request": "写两个文件"}).json()["thread_id"]
                rt = registry.get_or_create(tid)
                ap = wait_event(rt, "approval_required")
                assert len(ap["data"]["changes"]) == 2
                r = client.post(f"/api/sessions/{tid}/approvals",
                                json={"approval_id": ap["data"]["approval_id"],
                                      "action": "approve", "indices": [0]})
                assert r.status_code == 200, r.text
                wait_event(rt, "round_end")
                assert Path(f1).read_text(encoding="utf-8") == "一", "indices[0] 应写入"
                assert not Path(f2).exists(), "indices[0] 选批：第二条应跳过"
                entries = read_audit()
                assert any(e.get("indices") == [0] and e.get("source") == "web"
                           for e in entries), entries

                # 非法动作 → 422（fail-closed）
                r = client.post(f"/api/sessions/{tid}/approvals",
                                json={"approval_id": "whatever", "action": "yolo"})
                assert r.status_code == 422, r.text
    finally:
        for f in (f1, f2, frej):
            Path(f).unlink(missing_ok=True)
    print("PASS reject 不写入 / indices [0] 只批一条 / 非法动作 422 ✔\n")


def test_restart_rebuild():
    """M2 重启场景：审批挂起中「服务重启」（新 registry+app，同一 MemorySaver）
    → GET 快照重建审批卡 → 新 app 上 approve → 新工作线程续跑跑完。"""
    scratch = "__web_smoke_restart__.txt"
    calls = [
        AIMessage(content='["写文件"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file",
             "args": {"path": scratch, "content": "重启后批准\n"}, "id": "w1"}]),
        AIMessage(content="verdict: pass\nfeedback: 放行。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    p1, p2, p3 = model_patches(FakeModel(calls))
    try:
        with p1, p2, p3:
            saver = MemorySaver()
            app1, registry1 = make_app(saver)
            with TestClient(app1) as client1:
                tid = client1.post("/api/sessions",
                                   json={"request": "写文件"}).json()["thread_id"]
                rt1 = registry1.get_or_create(tid)
                wait_event(rt1, "approval_required")
                # 「重启」：rt1 的工作线程阻塞在等决策（被遗弃）；新 registry + 新 app
                app2, registry2 = make_app(saver)
                with TestClient(app2) as client2:
                    r = client2.get(f"/api/sessions/{tid}")
                    snap = r.json()
                    assert snap["pending_approval"] is not None, \
                        f"重启后应重建审批卡：{snap}"
                    assert snap["graph"]["next"] == ["guard"], snap["graph"]
                    new_aid = snap["pending_approval"]["approval_id"]
                    assert snap["pending_approval"]["changes"][0]["path"] == scratch
                    r = client2.post(f"/api/sessions/{tid}/approvals",
                                     json={"approval_id": new_aid, "action": "approve"})
                    assert r.status_code == 200 and r.json()["mode"] == "resumed", r.text
                    rt2 = registry2.get_or_create(tid)
                    end = wait_event(rt2, "round_end")
                    assert end["data"]["verdict"] == "pass", end
                    assert Path(scratch).read_text(encoding="utf-8") == "重启后批准\n"
                    cur = rt2.graph.get_state(rt2.sess.config)
                    assert not cur.next, "续跑后图应跑完"
    finally:
        Path(scratch).unlink(missing_ok=True)
    print("PASS M2 重启：快照重建审批卡 + 新 app approve 续跑跑完 ✔\n")


def test_retry_nothing_pending():
    """无断点时 POST retry → 409 nothing_pending。"""
    p1, p2, p3 = model_patches(FakeModel([]))
    with p1, p2, p3:
        app, registry = make_app()
        with TestClient(app) as client:
            tid = client.post("/api/sessions", json={}).json()["thread_id"]
            r = client.post(f"/api/sessions/{tid}/retry", json={})
            assert r.status_code == 409 and r.json()["error"] == "nothing_pending", r.text
    print("PASS retry 无断点 → 409 nothing_pending ✔\n")


def test_undo_endpoint():
    """undo 端点直调 agent._undo_latest：无快照时返回人可读提示。"""
    p1, p2, p3 = model_patches(FakeModel([]))
    with p1, p2, p3:
        app, registry = make_app()
        with TestClient(app) as client:
            tid = client.post("/api/sessions", json={}).json()["thread_id"]
            r = client.post(f"/api/sessions/{tid}/undo", json={})
            assert r.status_code == 200 and "没有可回退的快照" in r.json()["result"], r.text
    print("PASS undo 端点（无快照场景）✔\n")


def test_health_masking_and_token():
    """health 掩码（不出完整 base_url/key）+ token 模式 /api/* 全量校验 Bearer。"""
    env = {"MODEL_NAME": "web-smoke-model",
           "OPENAI_BASE_URL": "https://api.secret-host.example.com/v1",
           "OPENAI_API_KEY": "sk-web-smoke-key"}
    with patch.dict(os.environ, env):
        app, _ = make_app()
        with TestClient(app) as client:
            r = client.get("/api/health")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["model"] == "web-smoke-model", body
            assert body["key_configured"] is True
            assert "secret-host" not in r.text, f"base_url 未掩码：{body}"
            assert "sk-web-smoke-key" not in r.text, "key 泄露"
            assert set(body["permissions"]) == {"write", "command", "python"}, body

        # token 模式：无头 401，正确 Bearer 200（health 也在校验范围内）
        app_t, _ = make_app(token="tok-123")
        with TestClient(app_t) as client_t:
            assert client_t.get("/api/health").status_code == 401
            assert client_t.get("/api/sessions").status_code == 401
            r = client_t.get("/api/health", headers={"Authorization": "Bearer tok-123"})
            assert r.status_code == 200, r.text
            r = client_t.get("/api/health", headers={"Authorization": "Bearer wrong"})
            assert r.status_code == 401
    print("PASS health 掩码 + token 模式 Bearer 校验 ✔\n")


def test_415_and_host_guard():
    """POST 非 JSON → 415；非 loopback 强制 token：缺省自动生成随机 token 打印（Jupyter 式），
    显式提供（--token）时不生成（fail-closed 语义不变：token 模式所有 /api 校验 Bearer）。"""
    p1, p2, p3 = model_patches(FakeModel([]))
    with p1, p2, p3:
        app, _ = make_app()
        with TestClient(app) as client:
            tid = client.post("/api/sessions", json={}).json()["thread_id"]
            r = client.post(f"/api/sessions/{tid}/messages",
                            content="hello", headers={"content-type": "text/plain"})
            assert r.status_code == 415, r.text
    saved = os.environ.pop("BLUE_WEB_TOKEN", None)
    try:
        import io
        from contextlib import redirect_stdout
        with patch("uvicorn.run"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = web_server.main(["--host", "0.0.0.0", "--no-browser"])
            assert rc == 0, rc
            out = buf.getvalue()
            assert "token 模式" in out and "随机 token" in out, out
        with patch("uvicorn.run"):
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                web_server.main(["--host", "0.0.0.0", "--token", "fixed-tok", "--no-browser"])
            assert "随机 token" not in buf2.getvalue(), buf2.getvalue()
    finally:
        if saved is not None:
            os.environ["BLUE_WEB_TOKEN"] = saved
    print("PASS POST 非 JSON 415 + 非 loopback token 强制（缺省自动生成）✔\n")


def test_diff_helpers():
    """change_detail 的 patch 结构化 diff（服务端 difflib，与 CLI 同算法）。"""
    detail = change_detail(0, {"action": "plan_patch", "path": "a.py",
                               "old": "x = 1\ny = 2\n", "new": "x = 1\ny = 3\n"})
    assert detail["path"] == "a.py" and detail["category"] == "write"
    hunks = detail["diff"]["hunks"]
    assert hunks, "应有 diff hunks"
    types = [l["type"] for h in hunks for l in h["lines"]]
    assert "add" in types and "del" in types and "ctx" in types, types
    # 行号对齐：del 只有 old 行号，add 只有 new 行号
    for h in hunks:
        for l in h["lines"]:
            if l["type"] == "del":
                assert l["old"] is not None and l["new"] is None
            if l["type"] == "add":
                assert l["new"] is not None and l["old"] is None
    d2 = structured_diff("a\n", "a\nb\n")
    assert d2["new_lines"] == 2 and d2["hunks"]
    print("PASS patch 结构化 diff hunks（行号对齐）✔\n")


def test_no_cross_session_event_leak():
    """并发串扰回归（design §8 / 评审 P2）：会话 A 执行中（挂审批）→ 会话 B 发起新轮
    （busy 是会话级，B 可启动，但其线程阻塞在全局信号量上，step 回调已注册进全局列表）
    → 批准 A 后 A 的 guard/verifier/reviewer/report 节点事件不得混入 B 的事件总线。
    以 A 的指纹（改动文件路径）断言——B 自身的事件永不包含该路径，时序无关不抖。"""
    scratch = "__web_smoke_leak__.txt"
    calls = [
        AIMessage(content='["写文件"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file",
             "args": {"path": scratch, "content": "leak test\n"}, "id": "l1"}]),
        AIMessage(content="verdict: pass\nfeedback: 放行。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    p1, p2, p3 = model_patches(FakeModel(calls))
    try:
        with p1, p2, p3:
            app, registry = make_app()  # concurrency=1（默认）：全局串行
            with TestClient(app) as client:
                # 会话 A：写轮，停在审批点
                tid_a = client.post("/api/sessions",
                                    json={"request": "写文件"}).json()["thread_id"]
                rt_a = registry.get_or_create(tid_a)
                wait_event(rt_a, "approval_required")
                # 会话 B：发起新轮——round_start 在信号量 acquire 前发布，B 的回调此时已入全局列表
                tid_b = client.post("/api/sessions",
                                    json={"request": "B 的需求"}).json()["thread_id"]
                rt_b = registry.get_or_create(tid_b)
                wait_event(rt_b, "round_start")
                # 批准 A：guard 恢复，后续节点事件（含改动路径指纹）相继发出
                ap = wait_event(rt_a, "approval_required")
                r = client.post(f"/api/sessions/{tid_a}/approvals",
                                json={"approval_id": ap["data"]["approval_id"],
                                      "action": "approve"})
                assert r.status_code == 200, r.text
                wait_event(rt_a, "round_end")
                # A 自己的总线收到后续节点（guard 恢复后的 verifier）
                a_nodes = {e["data"]["node"] for e in rt_a.bus.events if e["event"] == "node"}
                assert "verifier" in a_nodes, a_nodes
                # B 的总线绝不能出现 A 的节点事件（含路径指纹）
                leaked = [
                    e for e in rt_b.bus.events if e["event"] == "node"
                    and scratch in json.dumps(e["data"], ensure_ascii=False)
                ]
                assert not leaked, f"跨会话串扰：B 收到了 {[e['data']['node'] for e in leaked]}"
                # B 排到信号量后自己会跑完（fake 序列耗尽走 fallback，不影响上述断言）
                wait_event(rt_b, "round_end", timeout=30.0)
    finally:
        Path(scratch).unlink(missing_ok=True)
    print("PASS 两会话并发：A 的节点事件未混入 B 的总线（串扰回归）✔\n")


if __name__ == "__main__":
    test_readonly_round()
    test_write_approve_flow()
    test_reject_and_indices()
    test_restart_rebuild()
    test_retry_nothing_pending()
    test_undo_endpoint()
    test_health_masking_and_token()
    test_415_and_host_guard()
    test_diff_helpers()
    test_no_cross_session_event_leak()
    print("ALL WEB SMOKE TESTS PASSED ✅")
