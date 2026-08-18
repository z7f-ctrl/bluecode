"""小蓝 Blue —— 会话管理 + 持久化（Session 类 + sessions 表 + 目录常量）。

从 agent.py 拆出（#7 模块拆分）：会话元信息（thread_id / 轮次 / token 累计）
落 sqlite 辅助表，供 --resume 列表查询；目录常量（BLUE_DIR/DB_PATH/AUDIT_LOG/
BACKUP_ROOT/ENV_GLOBAL_PATH）集中在此，agent 重导出以保证测试对 agent.X 的 patch 生效。
本模块不依赖 agent，避免循环导入。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime

BLUE_DIR = os.path.expanduser("~/.blue")
DB_PATH = os.path.join(BLUE_DIR, "checkpoints.sqlite")
AUDIT_LOG = os.path.join(BLUE_DIR, "audit.jsonl")
BACKUP_ROOT = os.path.join(BLUE_DIR, "backups")
ARCHIVE_DIR = os.path.join(BLUE_DIR, "archives")  # 压缩摘要归档（跨轮/revise 落盘，可追溯）
ENV_GLOBAL_PATH = os.path.join(BLUE_DIR, ".env")

LOCK_STALE_SECONDS = 90           # 执行锁心跳过期阈值：超过视为持有者已崩溃，可接管
LOCK_HEARTBEAT_INTERVAL = 30      # 持有者心跳刷新间隔（必须远小于 STALE）


def _ensure_blue_dir() -> None:
    os.makedirs(BLUE_DIR, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    _ensure_blue_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Session:
    """一次交互式会话：维护 thread_id 与轮次，支持多轮需求。"""

    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id or f"blue-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.round = 0
        self.created_at = datetime.now().isoformat(timespec="seconds")
        # 会话级 token 累计（内存，不落库；重启清零）
        # context = 会话内峰值单次调用 prompt tokens（"当前上下文占用"的近似，
        # 跨轮取 max：历史累积只增，压缩后可能回落，max 反映最接近窗口的时刻）
        self.token_usage = {"prompt": 0, "completion": 0, "calls": 0, "context": 0}

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def next_round(self) -> int:
        self.round += 1
        return self.round


def _save_session_meta(sess: Session) -> None:
    """把会话元信息写入 sqlite 辅助表，供 --resume 列表查询。"""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active TEXT,
                rounds INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (thread_id, created_at, last_active, rounds)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                last_active = excluded.last_active,
                rounds = excluded.rounds
            """,
            (sess.thread_id, sess.created_at, datetime.now().isoformat(timespec="seconds"), sess.round),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions() -> list[dict]:
    """从辅助表读历史会话列表。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT thread_id, created_at, last_active, rounds FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        return [
            {"thread_id": r[0], "created_at": r[1], "last_active": r[2], "rounds": r[3]}
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ─────────────────────────── 跨进程执行锁（M1，v0.8.x） ───────────────────────────
# 背景：CLI 与 Web 是两个独立进程，各自直连同一 checkpoints.sqlite；对同一 thread
# 并发 graph.stream 是数据竞争（checkpoint 互相覆盖丢失）。Web 的 busy/信号量只在
# 进程内，CLI 不知道——故加一把 sqlite 里的执行锁：执行一轮（含审批等待期）前
# acquire、结束后 release；持有者崩溃残留靠心跳过期自动接管。语义：单写者串行化，
# 绝不抢活锁（fail-closed）。锁表独立于 sessions 表（新表无迁移问题）。


class SessionBusyError(RuntimeError):
    """跨进程执行锁冲突：同一会话正被另一持有者执行中。"""

    def __init__(self, thread_id: str, holder: str, pid: int):
        self.thread_id = thread_id
        self.holder = holder
        self.pid = pid
        super().__init__(
            f"会话 {thread_id} 正被 {holder}（pid {pid}）执行中，请稍后再试"
        )


def _lock_conn() -> sqlite3.Connection:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exec_locks (
            thread_id TEXT PRIMARY KEY,
            holder TEXT NOT NULL,
            pid INTEGER NOT NULL,
            heartbeat_at TEXT NOT NULL
        )
        """
    )
    return conn


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _lock_is_stale(heartbeat_at: str) -> bool:
    try:
        ts = datetime.fromisoformat(heartbeat_at)
    except ValueError:
        return True
    return (datetime.now() - ts).total_seconds() > LOCK_STALE_SECONDS


def peek_exec_lock(thread_id: str) -> dict | None:
    """只读探测：锁被「其他进程的持有者」且心跳新鲜时返回 {"holder","pid"}，否则 None。

    供 API 层在发起执行前给出友好 409（不抢锁、不抛异常）。同进程内持有不报告
    ——进程内由 busy 标志管辖（消息更准：round_running 而非 session_busy）。
    """
    if not thread_id:
        return None
    try:
        conn = _lock_conn()
        try:
            row = conn.execute(
                "SELECT holder, pid, heartbeat_at FROM exec_locks WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return None
    if row is None or _lock_is_stale(row[2]) or row[1] == os.getpid():
        return None
    return {"holder": row[0], "pid": row[1]}


def acquire_exec_lock(thread_id: str, holder: str) -> None:
    """获取跨进程执行锁。可重入：同持有者重复获取 = 刷新心跳（Web 线程内二次获取安全）。

    被其他持有者占用（心跳新鲜）→ 抛 SessionBusyError，绝不抢活锁；
    持有者崩溃残留（心跳过期）→ 自动接管。失败静默（锁是串行化辅助，不阻断主流程
    的更严格场景由调用方决定如何处理 SessionBusyError）。
    """
    if not thread_id:
        return
    conn = _lock_conn()
    try:
        row = conn.execute(
            "SELECT holder, pid, heartbeat_at FROM exec_locks WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if row is not None and not _lock_is_stale(row[2]) and row[0] != holder:
            raise SessionBusyError(thread_id, row[0], row[1])
        conn.execute(
            """
            INSERT INTO exec_locks (thread_id, holder, pid, heartbeat_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                holder = excluded.holder, pid = excluded.pid,
                heartbeat_at = excluded.heartbeat_at
            """,
            (thread_id, holder, os.getpid(), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def heartbeat_exec_lock(thread_id: str, holder: str) -> None:
    """刷新心跳（长执行期间周期调用）。只刷自己的行（holder+pid 匹配），
    已被他人接管时绝不误刷他人锁的心跳。"""
    if not thread_id:
        return
    try:
        conn = _lock_conn()
        try:
            conn.execute(
                "UPDATE exec_locks SET heartbeat_at=? "
                "WHERE thread_id=? AND holder=? AND pid=?",
                (_now_iso(), thread_id, holder, os.getpid()),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        pass


def release_exec_lock(thread_id: str) -> None:
    if not thread_id:
        return
    try:
        conn = _lock_conn()
        try:
            conn.execute("DELETE FROM exec_locks WHERE thread_id=?", (thread_id,))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        pass


class _HeartbeatThread(threading.Thread):
    def __init__(self, thread_id: str, holder: str):
        super().__init__(daemon=True, name=f"blue-lock-hb-{thread_id}")
        self._thread_id = thread_id
        self._holder = holder
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(LOCK_HEARTBEAT_INTERVAL):
            heartbeat_exec_lock(self._thread_id, self._holder)

    def stop(self) -> None:
        self._stop.set()


@contextmanager
def exec_lock(thread_id: str, holder: str):
    """跨进程执行锁上下文：acquire → 心跳线程保活 → finally release。

    用法：`with exec_lock(sess.thread_id, "cli"): <执行一轮>`。审批等待期也在锁内
    （锁的语义是「同一会话同一时刻只有一个执行者」，跨端审批是 M2 客户端模式的事，
    直连双写者场景宁可不跨端审批也不并发执行）。
    """
    if not thread_id:
        yield
        return
    acquire_exec_lock(thread_id, holder)
    hb = _HeartbeatThread(thread_id, holder)
    hb.start()
    try:
        yield
    finally:
        hb.stop()
        release_exec_lock(thread_id)
