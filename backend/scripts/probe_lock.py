"""诊断 SQLite 写锁行为：busy_timeout 是否真的让第二个写入排队等待。

运行: python scripts/probe_lock.py
"""

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JWT_SECRET_KEY", "probe-only-jwt-secret-not-for-production")
os.environ.setdefault("STARTUP_PROBE_EXTERNAL", "false")
os.environ.setdefault("WARMUP_RERANKER", "false")
os.environ["DATABASE_URL"] = f"sqlite:///{(ROOT / 'data' / 'probe.db').as_posix()}"

from sqlalchemy import text  # noqa: E402

from app.storage.db import get_engine, init_db, session_scope  # noqa: E402

_INSERT = (
    "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
    "VALUES (:id, 'u-1001', 'probe', datetime('now'), datetime('now'))"
)


def main() -> int:
    for stale in ("probe.db", "probe.db-wal", "probe.db-shm"):
        (ROOT / "data" / stale).unlink(missing_ok=True)

    init_db()
    engine = get_engine()

    with engine.connect() as conn:
        print(f"  busy_timeout = {conn.execute(text('PRAGMA busy_timeout')).scalar()}ms")
        print(f"  journal_mode = {conn.execute(text('PRAGMA journal_mode')).scalar()}")
        print(f"  synchronous  = {conn.execute(text('PRAGMA synchronous')).scalar()}")

    hold_seconds = 3.0
    ready = threading.Event()

    def holder() -> None:
        with engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            conn.execute(text(_INSERT), {"id": "holder"})
            ready.set()
            time.sleep(hold_seconds)
            conn.exec_driver_sql("ROLLBACK")

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    ready.wait(timeout=5)

    print(f"\n  一个写事务已持锁 {hold_seconds}s，第二个写入尝试中...")
    started = time.perf_counter()
    try:
        with session_scope() as session:
            session.execute(text(_INSERT), {"id": "waiter"})
        waited = time.perf_counter() - started
        print(f"  成功：等待 {waited:.1f}s 后拿到写锁")
        verdict = 0 if waited > hold_seconds * 0.8 else 1
        if verdict:
            print("  异常：几乎没等待就成功，说明前一个事务没真正持锁")
    except Exception as exc:
        waited = time.perf_counter() - started
        print(f"  失败（等待 {waited:.1f}s）：{type(exc).__name__}")
        print(f"    {str(exc)[:160]}")
        print("\n  busy_timeout 未生效——这正是 database is locked 的直接原因")
        verdict = 1

    thread.join(timeout=5)
    for stale in ("probe.db", "probe.db-wal", "probe.db-shm"):
        (ROOT / "data" / stale).unlink(missing_ok=True)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
