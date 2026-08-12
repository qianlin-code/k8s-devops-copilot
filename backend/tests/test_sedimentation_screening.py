"""沉淀自动初筛：去重与质量打分的三条路径。

用替身覆盖 embedding/vector_store/quality_client，不依赖真实云端调用。
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.knowledge.ingest import KnowledgeIngestor
from app.knowledge.sedimentation import SedimentationService
from app.rag.vector_store import ScoredChunk
from app.storage.models import AUTO_QUALITY_REVIEWER
from tests.conftest import ADMIN_HEADERS
from tests.fakes import ScriptedLLMClient


class _FakeEmbedding:
    def embed_one(self, text: str) -> list[float]:
        return [0.1] * 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class _FakeVectorStore:
    """search 返回值由测试用例通过 hits 属性控制。"""

    def __init__(self, hits: list[ScoredChunk] | None = None) -> None:
        self.hits = hits or []

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        return self.hits


class _FakeQualityClient:
    """结构化调用的替身：固定返回构造时传入的字段。"""

    def __init__(self, *, quality_score: float, sensitive: bool = False, reasoning: str = "ok") -> None:
        self._payload = {
            "quality_score": quality_score,
            "reasoning": reasoning,
            "contains_sensitive_info": sensitive,
        }

    def structured(self, messages: list[dict[str, str]], schema: type, **_: Any) -> Any:
        return schema.model_validate(self._payload)


def _make_conversation(client: TestClient, llm: ScriptedLLMClient) -> str:
    llm.queue(
        {"action": "answer", "reasoning": "可直接回答", "confidence": 0.9},
        {"sufficient": True, "reasoning": "片段足够"},
        "permission_level 为 restricted 会导致 403，请提权到 standard。",
    )
    resp = client.post(
        "/api/v1/chat",
        headers=ADMIN_HEADERS,  # 审核操作需要 admin 权限
        json={"question": "u-1001 登录 403 怎么办"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["conversation_id"]


def _make_service(
    *,
    hits: list[ScoredChunk] | None = None,
    quality_score: float = 0.9,
    sensitive: bool = False,
    quality_client: object | None = "default",
) -> SedimentationService:
    """approve() 真正入库要靠 ingestor 的向量库/embedding 真实工作，
    用测试环境已接好替身的 conftest 单例；查重则单独注入可控的 fake，
    两者职责不同，不能共用同一个替身。
    """
    from app.llm.factory import get_embedding_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.vector_store import get_vector_store

    ingestor = KnowledgeIngestor(
        vector_store=get_vector_store(),
        embedding_client=get_embedding_client(),
        bm25_index=get_bm25_index(),
    )
    resolved_quality_client = (
        _FakeQualityClient(quality_score=quality_score, sensitive=sensitive)
        if quality_client == "default"
        else quality_client
    )
    return SedimentationService(
        ingestor,
        embedding_client=_FakeEmbedding(),
        vector_store=_FakeVectorStore(hits),
        quality_client=resolved_quality_client,
    )


@pytest.fixture
def sedimentation_deps(client: TestClient, llm: ScriptedLLMClient):
    conversation_id = _make_conversation(client, llm)
    from app.storage.db import get_session_factory

    return conversation_id, get_session_factory()


def test_high_quality_auto_approves(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    service = _make_service(hits=[], quality_score=0.95)

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.status == "approved"
        assert entry.auto_approved is True
        assert entry.reviewed_by == AUTO_QUALITY_REVIEWER
        assert entry.quality_score == pytest.approx(0.95)
        assert entry.kb_document_id is not None
    finally:
        session.close()


def test_low_quality_stays_pending_for_human_review(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    service = _make_service(hits=[], quality_score=0.4)

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.status == "pending"
        assert entry.auto_approved is False
        assert entry.quality_score == pytest.approx(0.4)
        assert entry.kb_document_id is None
    finally:
        session.close()


def test_sensitive_content_forces_pending_even_with_high_raw_score(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    service = _make_service(hits=[], quality_score=0.9, sensitive=True)

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.status == "pending"
        assert entry.quality_score == 0.0  # 含敏感信息时分数被清零
    finally:
        session.close()


def test_duplicate_hit_skips_quality_scoring(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    dup_hit = ScoredChunk(
        chunk_id="c1",
        document_id="doc-existing",
        document_title="已有文档",
        text="重复内容",
        heading_path=[],
        chunk_index=0,
        score=0.97,
    )
    service = _make_service(hits=[dup_hit], quality_score=0.95)

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.status == "pending"
        assert entry.duplicate_of_document_id == "doc-existing"
        assert entry.duplicate_score == pytest.approx(0.97)
        # 重复命中后不应该再打质量分——没有意义
        assert entry.quality_score is None
    finally:
        session.close()


def test_similar_but_below_threshold_is_not_flagged_as_duplicate(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    near_hit = ScoredChunk(
        chunk_id="c1",
        document_id="doc-existing",
        document_title="已有文档",
        text="相似但不算重复",
        heading_path=[],
        chunk_index=0,
        score=0.5,
    )
    service = _make_service(hits=[near_hit], quality_score=0.9)

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.duplicate_of_document_id is None
        assert entry.duplicate_score == pytest.approx(0.5)
        # 未命中重复，继续走质量打分
        assert entry.status == "approved"
    finally:
        session.close()


def test_quality_client_unavailable_degrades_to_manual_review(sedimentation_deps) -> None:
    conversation_id, factory = sedimentation_deps
    service = _make_service(hits=[], quality_client=None)  # 未配置 QWEN_API_KEY 时的情形

    session = factory()
    try:
        entry = service.mark(session, conversation_id=conversation_id, marked_by="admin-1")
        session.commit()

        assert entry.status == "pending"
        assert entry.quality_score is None
        assert "自动初筛不可用" in (entry.quality_reasoning or "")
    finally:
        session.close()
