"""SQLite 并发写测试。

WAL 允许「读不阻塞写」，但写与写仍互斥。没有 busy_timeout 时，
并发写的第二个连接会立刻抛 "database is locked" 而不是排队等待 ——
实测中前端连续发两条消息就会命中。
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from sqlalchemy import text

from app.config import get_settings
from app.storage.db import get_engine, session_scope
from app.storage.models import Conversation


def test_pragmas_are_applied() -> None:
    with get_engine().connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == (
            get_settings().sqlite_busy_timeout_ms
        )
        # synchronous: 0=OFF 1=NORMAL 2=FULL
        assert conn.execute(text("PRAGMA synchronous")).scalar() == 1
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def _insert_conversation(index: int) -> str:
    with session_scope() as session:
        convo = Conversation(
            id=f"concurrent-{index}",
            user_id="u-1001",
            title=f"并发写 {index}",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(convo)
    return f"concurrent-{index}"


def test_concurrent_writes_do_not_raise_database_locked() -> None:
    """8 个线程同时写入应全部成功，不出现 database is locked。"""
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(_insert_conversation, range(8)))

    assert len(set(ids)) == 8
    with session_scope() as session:
        stored = {
            row[0]
            for row in session.execute(
                text("SELECT id FROM conversations WHERE id LIKE 'concurrent-%'")
            )
        }
    assert stored == set(ids)


def test_read_during_write_is_not_blocked() -> None:
    """WAL 的核心收益：写事务未提交时读仍可进行。"""
    with session_scope() as writer:
        writer.add(
            Conversation(
                id="wal-writer",
                user_id="u-1001",
                title="未提交的写",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        writer.flush()  # 已持有写锁，尚未提交

        with session_scope() as reader:
            count = reader.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
            assert count is not None, "读操作不应被未提交的写阻塞"
