"""web/executor.py — ExecutorRegistry：每会话一个工作线程 + web_drain 审批桥。

架构合同（design-web.md §3/§5/§8）：
- 同进程 import agent；LangGraph 同步 API 跑在每会话一个的工作线程里（D2）；
- 默认全局串行（concurrency 信号量，D3）；同一会话运行中再发需求 → 409 round_running；
- web_drain（§5.1）：guard interrupt → 发 approval_required → threading.Event 阻塞
  等 REST 决策 → 审计（source=web）→ Command(resume=decision) 续跑，节点输出回流 SSE；
- 服务重启兜底：GET 快照从 checkpoint interrupts 重建审批卡，决策后由新工作线程
  直接注入 resume 续跑（原等待线程已死，interrupt 态天然持久不丢）。

graph_factory 参数（默认 lambda: agent.build_graph()）供测试注入共享 MemorySaver。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime

from langgraph.types import Command

import agent
import cli
from session import Session, list_sessions, exec_lock
from web.events import SessionBus, build_approval_card, redact_node_output


def _audit_log_web(thread_id: str, decision: dict, changes: list[dict]) -> None:
    """Web 审批审计：格式逐字段镜像 agent._audit_log（ts/thread/action/changes/indices/note），
    额外落 "source": "web"（design §7.6：审计决策区分 Web/CLI 来源）。

    不直接调 agent._audit_log 的原因：它对 decision 只抄 action/indices/note 三个键，
    {**decision, "source": "web"} 里的 source 会被静默丢弃（v0.8 读码确认），
    而 source 恰是 Web/CLI 审计区分的承载字段——故 Web 层自行写记录（同一 jsonl 文件、
    同一行格式），而不是改核心 agent.py。写入失败不阻断主流程（同 agent._audit_log）。
    """
    try:
        record: dict = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "thread": thread_id,
            "action": decision.get("action"),
            "changes": [cli.shown_change(c) for c in changes],
            "source": "web",
        }
        if decision.get("indices") is not None:
            record["indices"] = decision["indices"]
        if decision.get("note"):
            record["note"] = decision["note"]
        os.makedirs(agent.BLUE_DIR, exist_ok=True)
        with open(agent.AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — 审计是旁路，绝不阻断主流程
        pass


class ApprovalWaiter:
    """一次挂起审批：REST 决策的投递目标。waiting=False 表示服务重启后重建的卡
    （无工作线程在等，决策走「新线程注入 resume」路径而非 Event 投递）。"""

    __slots__ = ("approval_id", "changes", "card", "event", "decision", "waiting")

    def __init__(self, approval_id: str, changes: list[dict], card: list[dict]):
        self.approval_id = approval_id
        self.changes = changes
        self.card = card
        self.event = threading.Event()
        self.decision: dict | None = None
        self.waiting = True


class SessionRuntime:
    """单会话运行时：Session + graph（懒建）+ 事件总线 + 工作线程 + 未决审批。"""

    def __init__(self, sess: Session, registry: "ExecutorRegistry"):
        self.sess = sess
        self.registry = registry
        self.bus = SessionBus(sess.thread_id)
        self.graph = None
        self.busy = False
        self.lock = threading.Lock()
        self.pending: dict[str, ApprovalWaiter] = {}
        self.approval_seq = 0
        self.worker: threading.Thread | None = None

    # ── 基础设施 ──

    def ensure_graph(self):
        if self.graph is None:
            self.graph = self.registry.graph_factory()
        return self.graph

    def _new_waiter(self, node: str, changes: list[dict]) -> ApprovalWaiter:
        self.approval_seq += 1
        aid = f"{self.sess.thread_id}:{node}:{self.approval_seq}"
        waiter = ApprovalWaiter(
            aid, changes,
            build_approval_card(self.sess.round, aid, changes)["changes"],
        )
        self.pending[aid] = waiter
        return waiter

    def _step_bridge(self, node_name: str, output) -> None:
        """step 回调桥：节点输出 → SSE node 事件（裁剪后）。工作线程 finally 摘除。

        串扰防护：cli._step_callbacks 是进程级全局列表（并发会话共享同一份），且回调
        在拿全局信号量之前就注册（排队中的会话也会在列表里）——只有当前执行线程
        （self.worker）发出的步骤才进本会话事件总线，否则会话 A 执行中、会话 B 排队
        等信号量时，A 的节点事件会混入 B 的 SSE 流（design §8 并发场景）。
        """
        if threading.current_thread() is not self.worker:
            return
        self.bus.publish("node", {
            "round": self.sess.round,
            "node": node_name,
            "data": redact_node_output(node_name, output),
        })

    def _remove_bridge(self, bridge) -> None:
        try:
            cli._step_callbacks.remove(bridge)
        except ValueError:
            pass

    def _finish_thread(self, bridge, phase: str, err: BaseException | None) -> None:
        """工作线程收尾：摘回调、发 error/round_end、清 busy。"""
        self._remove_bridge(bridge)
        sess = self.sess
        if err is not None:
            self.bus.publish("error", {
                "round": sess.round, "phase": phase,
                "message": f"{type(err).__name__}: {err}",
                "recoverable": True, "hint": "可用 /retry（POST retry）断点续跑",
            })
        verdict = ""
        try:
            if self.graph is not None:
                verdict = self.graph.get_state(sess.config).values.get("verdict", "") or ""
        except Exception:  # noqa: BLE001 — 快照失败不影响 round_end
            pass
        self.bus.publish("round_end", {
            "round": sess.round,
            "usage": agent._token_usage_snapshot(),
            "session_total": dict(sess.token_usage),
            "verdict": verdict,
        })
        with self.lock:
            self.busy = False

    # ── web_drain（design §5.1）：guard interrupt ↔ REST 决策的桥 ──

    def web_drain(self, graph, config: dict, sess: Session) -> None:
        """签名与 agent._drain / _auto_drain 一致，经 _run_graph_core/resume_pending 注入。"""
        while True:
            cur = graph.get_state(config)
            if not cur.next:
                break
            tasks = [t for t in cur.tasks if t.interrupts]
            if not tasks:
                break  # 断在非审批点：留给 /retry（语义同 _drain）
            payload = tasks[0].interrupts[0].value
            changes = payload.get("changes", [])
            waiter = self._new_waiter(cur.next[0], changes)
            self.bus.publish("approval_required", {
                "round": sess.round,
                "approval_id": waiter.approval_id,
                "changes": waiter.card,
            })
            # 阻塞等 REST 决策：SSE 断开/前端崩溃/服务停止都绝不自动批（fail-closed），
            # interrupt 态天然挂起，重启后由 GET 快照重建审批卡（rebuild_pending）。
            waiter.event.wait()
            decision = waiter.decision or {"action": "reject", "note": "（审批通道异常关闭）"}
            self.pending.pop(waiter.approval_id, None)
            _audit_log_web(sess.thread_id, decision, changes)
            for chunk in graph.stream(Command(resume=decision), config=config,
                                      stream_mode="updates"):
                for node_name, output in chunk.items():
                    agent._emit_step(node_name, output)

    # ── 入口一：新一轮需求（POST messages） ──

    def try_start_round(self, text: str) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        self.worker = threading.Thread(target=self._round_thread, args=(text,),
                                       daemon=True, name=f"blue-web-{self.sess.thread_id}")
        self.worker.start()
        return True

    def _round_thread(self, text: str) -> None:
        sess = self.sess
        bridge = self._step_bridge
        agent.register_step_callback(bridge)
        self.bus.publish("round_start", {
            "round": sess.round + 1, "request": text, "thread_id": sess.thread_id,
        })
        err: BaseException | None = None
        self.registry.semaphore.acquire()
        try:
            banner = agent._c(f"[蓝] ★ Web 第 {sess.next_round()} 轮收到：{text}", agent._C.BLUE)
            agent._run_graph_core(self.ensure_graph(), sess, text, banner=banner,
                                  drain=self.web_drain, holder="web")
        except BaseException as exc:  # noqa: BLE001 — 兜底成 error 事件，绝不静默丢线程
            err = exc
        finally:
            self.registry.semaphore.release()
        self._finish_thread(bridge, "round", err)

    # ── 入口二：断点续跑（POST retry） ──

    def try_start_retry(self) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        self.worker = threading.Thread(target=self._retry_thread,
                                       daemon=True, name=f"blue-web-retry-{self.sess.thread_id}")
        self.worker.start()
        return True

    def _retry_thread(self) -> None:
        sess = self.sess
        bridge = self._step_bridge
        agent.register_step_callback(bridge)
        self.bus.publish("info", {"message": "🔁 从断点续跑（Web /retry）"})
        err: BaseException | None = None
        self.registry.semaphore.acquire()
        try:
            agent.resume_pending(self.ensure_graph(), sess, drain=self.web_drain,
                                 holder="web")
        except BaseException as exc:  # noqa: BLE001
            err = exc
        finally:
            self.registry.semaphore.release()
        self._finish_thread(bridge, "retry", err)

    # ── 审批决策投递（POST approvals） ──

    def find_waiter(self, approval_id: str) -> ApprovalWaiter | None:
        with self.lock:
            return self.pending.get(approval_id)

    def current_waiter(self) -> ApprovalWaiter | None:
        with self.lock:
            return next(iter(self.pending.values()), None)

    def submit_decision(self, waiter: ApprovalWaiter, decision: dict) -> str:
        """投递决策。返回 delivered（有工作线程在等）或 resume_needed（重启重建卡）。"""
        if waiter.waiting:
            waiter.decision = decision
            waiter.event.set()
            return "delivered"
        return "resume_needed"

    def try_start_decision_resume(self, waiter: ApprovalWaiter, decision: dict) -> bool:
        """重启场景：无等待中的工作线程，新线程直接注入决策续跑挂起的 guard。"""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.pending.pop(waiter.approval_id, None)
        self.worker = threading.Thread(
            target=self._decision_resume_thread, args=(waiter, decision),
            daemon=True, name=f"blue-web-resume-{self.sess.thread_id}")
        self.worker.start()
        return True

    def _decision_resume_thread(self, waiter: ApprovalWaiter, decision: dict) -> None:
        sess = self.sess
        bridge = self._step_bridge
        agent.register_step_callback(bridge)
        err: BaseException | None = None
        self.registry.semaphore.acquire()
        try:
            # M1 跨进程执行锁（v0.8.x）：重启重建审批卡的续跑同样占用会话，
            # 防止与 CLI 直连进程对同一 thread 并发 stream。
            with exec_lock(sess.thread_id, "web"):
                graph = self.ensure_graph()
                config = sess.config
                _audit_log_web(sess.thread_id, decision, waiter.changes)
                for chunk in graph.stream(Command(resume=decision), config=config,
                                          stream_mode="updates"):
                    for node_name, output in chunk.items():
                        agent._emit_step(node_name, output)
                self.web_drain(graph, config, sess)  # 可能再遇 interrupt（多段审批）
                agent._finish_round_usage(sess)
                agent._save_session_meta(sess)
        except BaseException as exc:  # noqa: BLE001
            err = exc
        finally:
            self.registry.semaphore.release()
        self._finish_thread(bridge, "resume", err)

    # ── 服务重启兜底：从 checkpoint 重建审批卡（GET 快照用） ──

    def rebuild_pending(self, cur) -> dict | None:
        """get_state 显示 interrupts 非空 → 重建审批卡并登记（决策走 resume 路径）。

        已有未决审批（等待中或已重建）时直接返回那张卡，不重复登记。
        """
        waiter = self.current_waiter()
        if waiter is not None:
            return {"round": self.sess.round, "approval_id": waiter.approval_id,
                    "changes": waiter.card}
        tasks = [t for t in cur.tasks if t.interrupts]
        if not tasks:
            return None
        payload = tasks[0].interrupts[0].value
        changes = payload.get("changes", [])
        waiter = self._new_waiter(cur.next[0] if cur.next else "guard", changes)
        waiter.waiting = False  # 重建卡：无工作线程在等决策
        return {"round": self.sess.round, "approval_id": waiter.approval_id,
                "changes": waiter.card}


class ExecutorRegistry:
    """会话运行时注册表：tid → SessionRuntime；全局串行信号量保护执行。"""

    def __init__(self, graph_factory=None, concurrency: int = 1):
        # graph_factory 默认生产图（SqliteSaver）；测试注入共享 MemorySaver 的图
        self.graph_factory = graph_factory or (lambda: agent.build_graph())
        self.semaphore = threading.BoundedSemaphore(max(1, concurrency))
        self._runtimes: dict[str, SessionRuntime] = {}
        self._lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """app startup 时捕获 asyncio loop（TestClient 的 portal loop 同样成立）。"""
        self.loop = loop
        with self._lock:
            runtimes = list(self._runtimes.values())
        for rt in runtimes:
            rt.bus.set_loop(loop)

    def add_session(self, sess: Session) -> SessionRuntime:
        with self._lock:
            rt = self._runtimes.get(sess.thread_id)
            if rt is None:
                rt = SessionRuntime(sess, self)
                if self.loop is not None:
                    rt.bus.set_loop(self.loop)
                self._runtimes[sess.thread_id] = rt
            return rt

    def get_or_create(self, tid: str, must_exist: bool = False) -> SessionRuntime | None:
        """按 tid 取运行时；不存在则从会话元信息恢复（轮次沿用，语义同 cli /resume）。"""
        with self._lock:
            rt = self._runtimes.get(tid)
        if rt is not None:
            return rt
        metas = {m["thread_id"]: m for m in list_sessions()}
        if must_exist and tid not in metas:
            return None
        sess = Session(thread_id=tid)
        meta = metas.get(tid)
        if meta:
            sess.round = meta.get("rounds", 0)
            sess.created_at = meta.get("created_at", sess.created_at)
        return self.add_session(sess)
