"""向量库与元数据对账测试。

SQLite 存元数据、Qdrant 存向量，单边重置会留下孤儿向量并永久污染检索。
"""

from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.dependencies import get_ingestor
from app.rag.bm25_index import get_bm25_index
from app.rag.vector_store import get_vector_store
from app.storage.db import session_scope
from app.storage.models import KBDocument
from tests.conftest import API_HEADERS


def test_reconcile_removes_orphan_vectors(client: TestClient, seeded_kb: str) -> None:
    store = get_vector_store()
    before = store.count()
    assert before > 0

    # 模拟"只重置了 SQLite"：元数据消失但向量还在
    with session_scope() as session:
        session.execute(delete(KBDocument))

    assert store.count() == before, "向量此时仍残留"

    with session_scope() as session:
        stats = get_ingestor().reconcile(session)

    assert stats["orphan_documents"] == 1
    assert store.count() == 0, "孤儿向量应被清除"
    assert get_bm25_index().size == 0, "BM25 索引应同步收敛"


def test_reconcile_keeps_valid_documents(client: TestClient, seeded_kb: str) -> None:
    store = get_vector_store()
    before = store.count()

    with session_scope() as session:
        stats = get_ingestor().reconcile(session)

    assert stats["orphan_documents"] == 0
    assert store.count() == before, "正常文档不应被误删"


def test_reconcile_is_idempotent(client: TestClient, seeded_kb: str) -> None:
    with session_scope() as session:
        first = get_ingestor().reconcile(session)
        second = get_ingestor().reconcile(session)
    assert first == second


def test_document_list_matches_vector_count(client: TestClient, seeded_kb: str) -> None:
    """列表接口报的向量数应与实际一致，否则前端展示会误导。"""
    body = client.get("/api/v1/knowledge/documents", headers=API_HEADERS).json()
    total_chunks = sum(d["chunk_count"] for d in body["documents"])
    assert body["vector_count"] == total_chunks
    assert body["bm25_index_size"] == total_chunks
