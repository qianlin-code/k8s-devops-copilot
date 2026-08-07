from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.storage.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _sqlite_file_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None
    return Path(url[len(prefix) :]).resolve()


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    settings = get_settings()
    db_file = _sqlite_file_path(settings.database_url)
    if db_file is not None:
        db_file.parent.mkdir(parents=True, exist_ok=True)

    is_sqlite = settings.database_url.startswith("sqlite")
    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {}
    if is_sqlite:
        connect_args = {
            "check_same_thread": False,
            # 关键：WAL 允许「读不阻塞写」，但写与写仍互斥。
            # 没有 timeout 时，并发写的第二个连接会立刻抛
            # "database is locked" 而不是排队等待。
            "timeout": settings.sqlite_busy_timeout_ms / 1000,
        }
        # FastAPI 用线程池跑同步端点，多个请求会并发拿连接。
        # 池子放开、复用连接，避免频繁建连触发锁竞争。
        engine_kwargs = {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}

    _engine = create_engine(
        settings.database_url,
        echo=False,
        future=True,
        connect_args=connect_args,
        **engine_kwargs,
    )

    if is_sqlite:
        busy_ms = settings.sqlite_busy_timeout_ms
        autocheckpoint = settings.sqlite_wal_autocheckpoint_pages

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record) -> None:  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            # 连接级重试窗口：拿不到写锁时轮询等待而非立即失败。
            # connect_args 的 timeout 只覆盖建连，这条覆盖后续每次语句。
            cursor.execute(f"PRAGMA busy_timeout={busy_ms}")
            # WAL 下 NORMAL 不牺牲一致性（崩溃最多丢最后一个事务），
            # 但写入快一个数量级，长事务里差别明显。
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 显式设定 checkpoint 阈值：WAL 文件靠 checkpoint 回收，
            # 长时间只写不回收会让它无限增长。
            cursor.execute(f"PRAGMA wal_autocheckpoint={autocheckpoint}")
            cursor.close()

    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI 依赖注入用。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """测试用：切换数据库后需要重建引擎。"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
