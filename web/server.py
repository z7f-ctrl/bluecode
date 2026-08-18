"""web/server.py — FastAPI app + REST 路由 + 启动入口（blue web 委托到此）。

协议合同见 design-web.md §4.1（REST）/ §4.2（SSE）/ §7（安全）：
- 默认只绑 127.0.0.1；非 loopback 强制 token（--token / BLUE_WEB_TOKEN，缺省自动生成随机 token 打印到控制台，Jupyter 式）；
- token 模式所有 /api/*（含 health）校验 Authorization: Bearer；
- 绝不提供 auto-approve 端点；POST 只收 application/json（否则 415）；不设 CORS；
- /api/health 密钥零暴露：base_url 只报掩码域名、key 只报 key_configured 布尔。

静态资源：web/static 存在时挂载 /static 且 GET / 服务 index.html；不存在时 GET /
返回占位页（前端落地前的 M0/M1 后端可独立跑通）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage

import agent
import session as session_mod
import tools
from models import model_kwargs, list_models, set_active_model
from session import Session
from web.events import change_detail, format_sse, validate_decision
from web.executor import ExecutorRegistry

STATIC_DIR = Path(__file__).resolve().parent / "static"

_PLACEHOLDER_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Bluecode Web</title></head>
<body><p>Bluecode Web 后端运行中。前端页面（web/static/index.html）尚未落地。</p></body>
</html>"""


# ─────────────────────────── 安全辅助（design §7） ───────────────────────────


def _mask_base_url(url: str) -> str:
    """base_url 域名掩码：只报 scheme + 域名前 3 字符 + ***，绝不出完整地址（密钥零暴露）。"""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url if "//" in url else f"https://{url}")
    except ValueError:
        return "***"
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        return f"{parsed.scheme}://{parsed.netloc}"  # 本机地址不算敏感
    return f"{parsed.scheme}://{host[:3]}***" if host else "***"


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def _make_auth(token: str | None):
    """token 模式：所有 /api/*（含 health）校验 Bearer；loopback 模式豁免。"""
    async def check(request: Request) -> None:
        if not token:
            return
        auth = request.headers.get("authorization", "")
        if not (auth.startswith("Bearer ")
                and secrets.compare_digest(auth[len("Bearer "):], token)):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    return check


async def _require_json(request: Request) -> None:
    """POST 只收 application/json（design §7.5：跨站表单打不进来）。"""
    ct = request.headers.get("content-type", "")
    if "application/json" not in ct.lower():
        raise HTTPException(status_code=415, detail="POST 只接受 application/json")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})


# ─────────────────────────── app 工厂 ───────────────────────────


def create_app(registry: ExecutorRegistry | None = None, *,
               token: str | None = None, concurrency: int = 1) -> FastAPI:
    registry = registry or ExecutorRegistry(concurrency=concurrency)
    app = FastAPI(title="Bluecode Web", docs_url=None, redoc_url=None, openapi_url=None)
    auth = Depends(_make_auth(token))

    @app.on_event("startup")
    async def _startup() -> None:
        # 捕获 asyncio loop 供工作线程 → SSE 桥（call_soon_threadsafe）。
        # starlette TestClient 的 portal loop 下同样成立。
        registry.set_loop(asyncio.get_running_loop())

    # ── 静态页 ──

    @app.get("/", include_in_schema=False)
    async def index():
        from fastapi.responses import FileResponse
        index_html = STATIC_DIR / "index.html"
        if index_html.is_file():
            return FileResponse(index_html)
        return HTMLResponse(_PLACEHOLDER_HTML)

    if STATIC_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ── health / audit ──

    @app.get("/api/health", dependencies=[auth])
    async def health():
        kwargs = model_kwargs()
        return {
            "model": agent.current_model_name(),
            "base_url": _mask_base_url(str(kwargs.get("base_url", ""))),
            "key_configured": bool(kwargs.get("api_key")),
            "permissions": tools.load_permissions(),
            "auth": "token" if token else "loopback",
            "version": agent.BLUE_VERSION,
            "context_window": agent.active_context_window(),
        }

    # ── 模型（M4 观测增强：列表 / 切换，语义与 CLI /model 一致：清缓存、下一轮生效）──

    @app.get("/api/models", dependencies=[auth])
    async def models_list():
        return {"models": list_models(), "active": agent.current_model_name(),
                "window": agent.active_context_window()}

    @app.post("/api/models", dependencies=[auth, Depends(_require_json)])
    async def models_set(request: Request):
        body = await request.json()
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="name 不能为空")
        ok, msg = set_active_model(name)
        if ok:
            agent._reset_model_cache()  # 与 agent.set_active_model 同语义：下一轮生效
        if not ok:
            return _error(404, "unknown_model", msg)
        return {"ok": True, "message": msg, "active": agent.current_model_name(),
                "window": agent.active_context_window()}

    @app.get("/api/audit", dependencies=[auth])
    async def audit(limit: int = 50):
        limit = max(1, min(int(limit), 500))
        entries: list[dict] = []
        path = agent.AUDIT_LOG
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        entries.append({"raw": line})
        except FileNotFoundError:
            pass
        return {"entries": entries}

    # ── 会话 ──

    @app.get("/api/sessions", dependencies=[auth])
    async def list_all_sessions():
        return {"sessions": session_mod.list_sessions()}

    @app.post("/api/sessions", dependencies=[auth, Depends(_require_json)])
    async def create_session(request: Request):
        body = await request.json()
        text = str(body.get("request") or "").strip()
        sess = Session()
        agent._save_session_meta(sess)
        rt = registry.add_session(sess)
        started = False
        if text:
            started = rt.try_start_round(text)
        return {"thread_id": sess.thread_id, "started": started}

    def _runtime_or_404(tid: str):
        rt = registry.get_or_create(tid, must_exist=True)
        if rt is None:
            raise HTTPException(status_code=404, detail=f"未知会话 {tid!r}")
        return rt

    @app.get("/api/sessions/{tid}", dependencies=[auth])
    async def session_snapshot(tid: str):
        rt = _runtime_or_404(tid)
        graph = rt.ensure_graph()
        cur = graph.get_state(rt.sess.config)
        values = cur.values if cur else {}
        # 挂起中的审批：interrupts 非空 → 重建审批卡（服务重启后 SSE 缓冲丢失的兜底）
        pending = rt.rebuild_pending(cur) if cur and cur.next else None
        return {
            "thread_id": tid,
            "created_at": rt.sess.created_at,
            "round": rt.sess.round,
            "busy": rt.busy,
            "graph": {"next": list(cur.next) if cur else []},
            "plan": values.get("plan", []),
            "verdict": values.get("verdict", ""),
            "review_rounds": values.get("review_rounds", 0),
            "report": values.get("feedback", ""),
            "token_usage": dict(rt.sess.token_usage),
            "pending_approval": pending,
        }

    @app.get("/api/sessions/{tid}/context", dependencies=[auth])
    async def session_context(tid: str):
        """/context 数据源（M2 客户端模式复用）：压缩摘要 + 消息尾部 + 归档记录。"""
        rt = _runtime_or_404(tid)
        graph = rt.ensure_graph()
        cur = graph.get_state(rt.sess.config)
        msgs = cur.values.get("messages", []) if cur else []
        summaries = [
            str(m.content) for m in msgs
            if isinstance(m, HumanMessage) and str(m.content).startswith("【")
        ]
        tail = [{
            "type": type(m).__name__.replace("Message", ""),
            "len": len(str(m.content)),
            "first": str(m.content).split("\n")[0][:80],
        } for m in msgs[-5:]]
        return {
            "thread_id": tid,
            "summaries": summaries,
            "tail": tail,
            "archive": agent.read_archive(tid, 3),
        }

    @app.post("/api/sessions/{tid}/messages", dependencies=[auth, Depends(_require_json)])
    async def post_message(tid: str, request: Request):
        rt = _runtime_or_404(tid)
        body = await request.json()
        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text 不能为空")
        # M1 跨进程执行锁（v0.8.x）：CLI 直连进程正执行该会话时拒绝，报持有方
        held = session_mod.peek_exec_lock(tid)
        if held:
            return _error(409, "session_busy",
                          f"该会话正被 {held['holder']}（pid {held['pid']}）执行中，请稍后再试")
        if not rt.try_start_round(text):
            return _error(409, "round_running", "该会话正有一轮在执行，稍后再发")
        return JSONResponse(status_code=202,
                            content={"thread_id": tid, "started": True})

    # ── 审批 ──

    @app.post("/api/sessions/{tid}/approvals", dependencies=[auth, Depends(_require_json)])
    async def post_approval(tid: str, request: Request):
        rt = _runtime_or_404(tid)
        body = await request.json()
        approval_id = str(body.get("approval_id") or "")
        try:
            decision = validate_decision(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        waiter = rt.find_waiter(approval_id)
        if waiter is None:
            # fail-closed：未知 approval_id 一律 404，绝不默认放行
            return _error(404, "unknown_approval_id", f"未知审批 {approval_id!r}")
        # M1 跨进程执行锁：CLI 直连进程正执行（含等待审批）时拒绝跨端投递决策
        held = session_mod.peek_exec_lock(tid)
        if held:
            return _error(409, "session_busy",
                          f"该会话正被 {held['holder']}（pid {held['pid']}）执行中，请稍后再试")
        mode = rt.submit_decision(waiter, decision)
        if mode == "resume_needed":
            # 服务重启后重建的审批卡：无工作线程在等，新线程注入决策续跑
            if not rt.try_start_decision_resume(waiter, decision):
                return _error(409, "round_running", "该会话正有一轮在执行，稍后再试")
            return {"ok": True, "mode": "resumed"}
        return {"ok": True, "mode": "delivered"}

    @app.get("/api/sessions/{tid}/changes/{index}", dependencies=[auth])
    async def get_change(tid: str, index: int):
        rt = _runtime_or_404(tid)
        waiter = rt.current_waiter()
        if waiter is None:
            return _error(404, "no_pending_changes", "当前没有挂起中的审批改动")
        if not (0 <= index < len(waiter.changes)):
            return _error(404, "change_out_of_range",
                          f"改动序号需在 0~{len(waiter.changes) - 1} 之间")
        return change_detail(index, waiter.changes[index])

    # ── undo / retry ──

    @app.post("/api/sessions/{tid}/undo", dependencies=[auth, Depends(_require_json)])
    async def post_undo(tid: str):
        rt = _runtime_or_404(tid)
        if rt.busy:
            return _error(409, "round_running", "该会话正有一轮在执行，稍后再试")
        held = session_mod.peek_exec_lock(tid)
        if held:
            return _error(409, "session_busy",
                          f"该会话正被 {held['holder']}（pid {held['pid']}）执行中，请稍后再试")
        return {"result": agent._undo_latest(tid)}

    @app.post("/api/sessions/{tid}/retry", dependencies=[auth, Depends(_require_json)])
    async def post_retry(tid: str):
        rt = _runtime_or_404(tid)
        graph = rt.ensure_graph()
        cur = graph.get_state(rt.sess.config)
        if not (cur and cur.next):
            return _error(409, "nothing_pending", "没有可续的断点（上一轮已正常结束）")
        held = session_mod.peek_exec_lock(tid)
        if held:
            return _error(409, "session_busy",
                          f"该会话正被 {held['holder']}（pid {held['pid']}）执行中，请稍后再试")
        if not rt.try_start_retry():
            return _error(409, "round_running", "该会话正有一轮在执行，稍后再试")
        return JSONResponse(status_code=202, content={"thread_id": tid, "started": True})

    # ── SSE 事件流 ──

    @app.get("/api/sessions/{tid}/events", dependencies=[auth])
    async def session_events(tid: str, request: Request):
        rt = _runtime_or_404(tid)
        try:
            last_id = int(request.headers.get("last-event-id") or 0)
        except ValueError:
            last_id = 0

        async def gen():
            queue: asyncio.Queue = asyncio.Queue()
            backlog = rt.bus.subscribe_replay(queue, last_id)
            try:
                for ev in backlog:
                    yield format_sse(ev)
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"  # 注释行保活，防代理掐空闲连接
                        continue
                    yield format_sse(ev)
            finally:
                rt.bus.unsubscribe(queue)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


# ─────────────────────────── 启动入口（blue web 委托到此） ───────────────────────────


def _serve_with_repl(app, *, host: str, port: int, token: str | None,
                     open_browser: bool = True, request: str | None = None) -> int:
    """REPL 模式（v0.8.4）：uvicorn 后台线程（静音）+ 主线程交互式客户端 REPL。

    - uvicorn：log_level=error + access_log=False，服务日志不再污染终端；
    - agent.QUIET_CONSOLE=True：banner/token/auto_allow 等服务端镜像打印静音——
      播报统一由 REPL 从 SSE 打印（否则与 REPL 双份输出）；
    - REPL = webclient.run_client：与浏览器同走 REST/SSE，同一引擎单写者，天然同步；
      输入 /quit 或 Ctrl-D 退出 REPL 并关停服务。request 给定时单发一轮（测试用）。
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port,
                                           log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn 启动超时")
        time.sleep(0.05)
    # 实际绑定端口（port=0 时由系统分配，REPL 与浏览器必须连真实端口）
    bound = server.servers[0].sockets[0].getsockname()[1] if port == 0 else port
    url = f"http://{host}:{bound}"
    if open_browser and _is_loopback(host):
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"[蓝] 🌐 Web 控制台启动：{url}（v{agent.BLUE_VERSION}）")
    if token:
        print(f"[蓝] 🔑 会话 token（请求需带 Authorization: Bearer）：{token}")
    print("[蓝] 终端已切换为交互控制台（与 Web 页面实时同步，同一引擎）——输入 /quit 退出并关停服务。")
    agent.QUIET_CONSOLE = True
    try:
        from webclient import run_client
        return run_client(url, token=token, request=request)
    finally:
        agent.QUIET_CONSOLE = False
        server.should_exit = True
        thread.join(timeout=10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blue web", description="小蓝 Web 控制台（v0.8）")
    parser.add_argument("--version", action="version",
                        version=f"小蓝 Blue {agent.BLUE_VERSION}", help="显示版本号并退出")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（默认仅本机 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="端口（默认 8765）")
    parser.add_argument("--token", default=None,
                        help="Bearer token（非 loopback 绑定必填；或环境变量 BLUE_WEB_TOKEN）")
    parser.add_argument("--concurrency", type=int, default=1, help="并发执行数（默认 1，全局串行）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--no-repl", action="store_true",
                        help="终端不进入交互 REPL（仅跑服务；stdin 非 TTY 时自动等同此模式）")
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("BLUE_WEB_TOKEN") or None
    generated_token = False
    # 安全红线（design §7.1）：非 loopback 绑定强制 token（fail-closed）——所有 /api 校验
    # Bearer。未显式提供（--token / BLUE_WEB_TOKEN）时启动生成随机 token 打印到控制台
    # （Jupyter 式）；检查在 uvicorn.run 之前。
    if not _is_loopback(args.host) and not token:
        token = secrets.token_urlsafe(16)
        generated_token = True

    try:
        import uvicorn
    except ImportError:
        print("[蓝] Web 控制台依赖未安装：pip install bluecode[web]"
              "（或 pip install fastapi uvicorn）", file=sys.stderr)
        raise SystemExit(1) from None

    app = create_app(token=token, concurrency=args.concurrency)
    # REPL 模式（v0.8.4）：终端 = 与 Web 页面平级的交互控制台（单写者引擎，双视图同步）。
    # 服务端 uvicorn 转后台线程并静音，主线程跑客户端 REPL（webclient.run_client）——
    # 播报全走 SSE 由 REPL 打印，不再当日志镜像。stdin 非 TTY（管道/CI/systemd）自动回退
    # 纯服务模式（uvicorn.run 前台阻塞）。
    if sys.stdin.isatty() and not args.no_repl:
        return _serve_with_repl(app, host=args.host, port=args.port, token=token,
                                open_browser=not args.no_browser)

    url = f"http://{args.host}:{args.port}"
    print(f"[蓝] 🌐 Web 控制台启动：{url}"
          + ("（token 模式：请求需带 Authorization: Bearer）" if token else "（仅本机访问）")
          + f"（v{agent.BLUE_VERSION}）")
    if generated_token:
        print(f"[蓝] 🔑 随机 token（请保存，重启后失效）：{token}")
    if not args.no_browser and _is_loopback(args.host):
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
