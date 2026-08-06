"""小蓝 Blue —— 会话管理 + 持久化（Session 类 + sessions 表 + 目录常量）。

从 agent.py 拆出（#7 模块拆分）：会话元信息（thread_id / 轮次 / token 累计）
落 sqlite 辅助表，供 --resume 列表查询；目录常量（BLUE_DIR/DB_PATH/AUDIT_LOG/
BACKUP_ROOT/ENV_GLOBAL_PATH）集中在此，agent 重导出以保证测试对 agent.X 的 patch 生效。
本模块不依赖 agent，避免循环导入。
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime

BLUE_DIR = os.path.expanduser("~/.blue")
DB_PATH = os.path.join(BLUE_DIR, "checkpoints.sqlite")
AUDIT_LOG = os.path.join(BLUE_DIR, "audit.jsonl")
BACKUP_ROOT = os.path.join(BLUE_DIR, "backups")
ENV_GLOBAL_PATH = os.path.join(BLUE_DIR, ".env")


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
        self.token_usage = {"prompt": 0, "completion": 0, "calls": 0}

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
