import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import inspect, text

from app.config import Environment, get_settings
from app.dependencies import get_ingestor
from app.errors import StartupCheckError
from app.llm.factory import get_embedding_client, get_llm_client
from app.rag.bm25_index import get_bm25_index
from app.rag.reranker import get_reranker
from app.rag.vector_store import get_vector_store
from app.storage.db import get_engine, init_db, session_scope
from app.storage.models import Base
from app.storage.seed import seed_mock_data

_REQUIRED_TABLES = {
    "conversations",
    "messages",
    "kb_documents",
    "tickets",
    "mock_accounts",
    "mock_orders",
    "tool_call_audits",
    "pending_sedimentations",
}


@dataclass(slots=True)
class CheckOutcome:
    name: str
    ok: bool
    detail: str | None
    elapsed_ms: int


def run_startup_checks(*, strict: bool) -> list[CheckOutcome]:
    """启动期依次校验依赖。strict=True 时任一硬性检查失败即终止启动。"""
    results: list[CheckOutcome] = [
        _check("config", _check_config),
        _check("database", _check_database),
        _check("vector_store", _check_vector_store),
        _check("bm25_index", _check_bm25),
    ]
    # 外部模型探测会触发本地模型冷启动（可能数十秒），开发阶段默认跳过
    if get_settings().startup_probe_external:
        results.extend(
            [
                _check("llm", _check_llm),
                _check("embedding", _check_embedding),
                _check("reranker", _check_reranker),
            ]
        )
    else:
        results.extend(
            CheckOutcome(name=n, ok=True, detail="probe skipped (dev)", elapsed_ms=0)
            for n in ("llm", "embedding", "reranker")
        )
    # LLM/Embedding/Rerank 属于外部依赖，缺失时降级运行而非阻止启动
    hard_failures = [
        r for r in results if not r.ok and r.name in {"config", "database", "vector_store"}
    ]
    if strict and hard_failures:
        detail = "; ".join(f"{r.name}: {r.detail}" for r in hard_failures)
        raise StartupCheckError(f"Startup checks failed -> {detail}")
    return results


def _check(name: str, fn) -> CheckOutcome:
    started = time.perf_counter()
    try:
        detail = fn()
        ok = True
    except Exception as exc:
        detail, ok = f"{type(exc).__name__}: {exc}"[:300], False
    return CheckOutcome(
        name=name,
        ok=ok,
        detail=detail,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _check_config() -> str:
    settings = get_settings()
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return (
        f"llm={settings.llm_provider.value} embedding={settings.embedding_provider.value} "
        f"collection={settings.collection_name}"
    )


def _check_database() -> str:
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    missing = _REQUIRED_TABLES - set(inspect(engine).get_table_names())
    if missing:
        raise RuntimeError(f"missing tables: {sorted(missing)}")
    settings = get_settings()
    # 演示用的假账号/订单只在非生产环境灌入。生产库若恰好是全新的，
    # 无条件跑这一步会把 u-1004(admin 权限)这类演示数据插进真实数据库。
    if settings.environment is Environment.PROD:
        return f"tables={len(Base.metadata.tables)} seeded=skipped(prod)"
    with session_scope() as session:
        seeded = seed_mock_data(session)
    return f"tables={len(Base.metadata.tables)} seeded={seeded}"


def _check_vector_store() -> str:
    store = get_vector_store()
    return f"collection={store.collection} dim={store.dim} points={store.count()}"


def _check_bm25() -> str:
    # 顺带对账，避免单边重置留下的孤儿向量污染检索
    with session_scope() as session:
        stats = get_ingestor().reconcile(session)
    return (
        f"indexed_chunks={stats['indexed_chunks']} "
        f"orphans_removed={stats['orphan_documents']}"
    )


def _check_llm() -> str:
    client = get_llm_client()
    reply = client.chat(
        [{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=5
    )
    return f"model={client.model} replied={len(reply)}chars"


def _check_embedding() -> str:
    client = get_embedding_client()
    vector = client.embed_one("connectivity probe")
    return f"model={client.model} dim={len(vector)}"


def _check_reranker() -> str:
    reranker = get_reranker()
    if not hasattr(reranker, "_ensure_model"):
        return f"reranker={getattr(reranker, 'name', 'unknown')} (no preload)"
    reranker._ensure_model()  # type: ignore[attr-defined]
    device = getattr(reranker, "resolved_device", None) or "unknown"
    return f"model={getattr(reranker, 'model_name', 'unknown')} device={device}"


def get_bm25_index_size() -> int:
    return get_bm25_index().size
