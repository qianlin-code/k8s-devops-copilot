from app.rag.reranker import RerankedChunk
from app.rag.retriever import (
    Retriever,
    RetrievalStage,
    _restore_explicit_heading_context,
    _restore_topic_context,
)
from app.rag.vector_store import ScoredChunk


def _item(
    chunk_id: str, heading: str, score: float, document_id: str = "pod-doc"
) -> RerankedChunk:
    return RerankedChunk(
        chunk=ScoredChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title="pod lifecycle",
            text="evidence",
            heading_path=["Pod 生命周期", *heading.split(" > ")],
            chunk_index=0,
            score=score,
        ),
        rerank_score=score,
        rank_before=1,
        rank_after=1,
    )


def test_restores_same_document_top_n_chunks_with_explicit_heading_token() -> None:
    kept = [_item("crash-observed", "CrashLoopBackOff > 现象", 0.0528)]
    candidates = kept + [
        _item("crash-root", "CrashLoopBackOff > 根因", 0.0172),
        _item("crash-steps", "CrashLoopBackOff > 处理步骤", 0.0255),
        _item("image-root", "ImagePullBackOff > 根因", 0.0338),
    ]
    stages: list[RetrievalStage] = []

    result = _restore_explicit_heading_context(
        query="CrashLoopBackOff 要等多久恢复",
        candidates=candidates,
        kept=kept,
        stages=stages,
    )

    assert [item.chunk.chunk_id for item in result] == [
        "crash-observed",
        "crash-root",
        "crash-steps",
    ]
    assert stages[-1].hit_count == 2


def test_does_not_restore_matching_heading_from_untrusted_document() -> None:
    kept = [_item("pending", "Pending > 根因", 0.20)]
    other = _item("waiting", "Waiting > 现象", 0.01, document_id="other-doc")
    stages: list[RetrievalStage] = []

    result = _restore_explicit_heading_context(
        query="Pending 和 Waiting 有什么区别",
        candidates=kept + [other],
        kept=kept,
        stages=stages,
    )

    assert result == kept


def test_restores_rerank_top_n_siblings_from_the_same_fault_topic() -> None:
    kept = [_item("rbac-observed", "Forbidden > 现象", 0.14, "rbac-doc")]
    candidates = kept + [
        _item("rbac-root", "Forbidden > 根因", 0.01, "rbac-doc"),
        _item("quota-root", "ResourceQuota > 根因", 0.02, "rbac-doc"),
        _item("other-root", "Forbidden > 根因", 0.02, "other-doc"),
    ]
    stages: list[RetrievalStage] = []

    result = _restore_topic_context(
        candidates=candidates,
        kept=kept,
        stages=stages,
    )

    assert [item.chunk.chunk_id for item in result] == ["rbac-observed", "rbac-root"]
    assert stages[-1].name == "topic_context"
    assert stages[-1].hit_count == 1


def test_topic_context_never_creates_evidence_without_a_threshold_anchor() -> None:
    candidate = _item("root", "Forbidden > 根因", 0.01, "rbac-doc")
    stages: list[RetrievalStage] = []

    assert _restore_topic_context(candidates=[candidate], kept=[], stages=stages) == []
    assert stages[-1].note == "no_kept_context"


def test_topic_context_accepts_same_topic_sibling_loaded_outside_rerank_top_n() -> None:
    kept = [_item("ingress-root", "Ingress 无法访问 > 根因", 0.31, "ingress-doc")]
    sibling = _item(
        "ingress-steps",
        "Ingress 无法访问 > 处理步骤",
        0.0,
        "ingress-doc",
    )
    sibling.rank_before = 999
    sibling.rank_after = 999
    unrelated = _item("network-steps", "NetworkPolicy > 处理步骤", 0.0, "ingress-doc")
    stages: list[RetrievalStage] = []

    result = _restore_topic_context(
        candidates=[*kept, sibling, unrelated],
        kept=kept,
        stages=stages,
    )

    assert [item.chunk.chunk_id for item in result] == ["ingress-root", "ingress-steps"]


def test_associated_recall_loads_same_document_siblings_outside_rerank_top_n() -> None:
    anchor = _item(
        "ingress-root",
        "Ingress unavailable > root cause",
        0.31,
        "ingress-doc",
    )
    sibling = _item(
        "ingress-steps",
        "Ingress unavailable > procedure",
        0.0,
        "ingress-doc",
    ).chunk

    class _Store:
        def get_chunks_by_document(self, document_id: str) -> list[ScoredChunk]:
            assert document_id == "ingress-doc"
            return [anchor.chunk, sibling]

    retriever = object.__new__(Retriever)
    retriever._store = _Store()  # type: ignore[attr-defined]
    stages: list[RetrievalStage] = []

    result = retriever._associated_recall([anchor], stages)

    assert [item.chunk.chunk_id for item in result] == ["ingress-root", "ingress-steps"]
    assert result[1].rank_before == 999
    assert result[1].rank_after == 999
    assert stages[-1].name == "associated_recall"
    assert stages[-1].top_chunk_ids == ["ingress-steps"]
