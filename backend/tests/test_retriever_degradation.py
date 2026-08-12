"""检索依赖故障必须收敛为可观测降级，而不是中断整条对话链路。"""

from app.errors import RerankerUnavailableError
from app.rag.fusion import FusedChunk
from app.rag.retriever import RetrievalStage, Retriever
from app.rag.vector_store import ScoredChunk


class _UnavailableReranker:
    name = "unavailable-test-double"

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int):  # noqa: ANN001
        raise RerankerUnavailableError("test-only reranker outage")


def test_reranker_outage_falls_back_to_rrf_with_explicit_stage_note() -> None:
    """Rerank 故障保留 RRF 候选，但 trace 绝不能伪称已执行 rerank。"""
    retriever = Retriever(
        vector_store=object(),  # type: ignore[arg-type]
        embedding_client=object(),  # type: ignore[arg-type]
        bm25_index=object(),  # type: ignore[arg-type]
        reranker=_UnavailableReranker(),
    )
    candidate = ScoredChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        document_title="test",
        text="Pod Pending troubleshooting",
        heading_path=[],
        chunk_index=0,
        score=0.7,
    )
    stages: list[RetrievalStage] = []

    reranked, applied = retriever._rerank(  # noqa: SLF001 - test degradation boundary
        "Pod Pending",
        [FusedChunk(candidate, rrf_score=0.42, vector_rank=1, bm25_rank=1)],
        top_n=1,
        stages=stages,
        enable_rerank=True,
    )

    assert applied is False
    assert [item.chunk.chunk_id for item in reranked] == ["chunk-1"]
    assert reranked[0].rerank_score == 0.42
    assert stages[-1].name == "rerank"
    assert stages[-1].note == "degraded:RERANKER_UNAVAILABLE"
