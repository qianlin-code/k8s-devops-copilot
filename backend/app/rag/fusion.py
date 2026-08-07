from dataclasses import dataclass

from app.rag.vector_store import ScoredChunk


@dataclass(slots=True)
class FusedChunk:
    chunk: ScoredChunk
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None

    @property
    def sources(self) -> list[str]:
        found = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.bm25_rank is not None:
            found.append("bm25")
        return found


def reciprocal_rank_fusion(
    vector_hits: list[ScoredChunk],
    bm25_hits: list[ScoredChunk],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedChunk]:
    """RRF 融合两路召回。用排名而非原始分数，避免余弦相似度与 BM25 分数量纲不可比。"""
    ranks: dict[str, dict[str, int]] = {}
    chunks: dict[str, ScoredChunk] = {}

    for position, hit in enumerate(vector_hits, start=1):
        ranks.setdefault(hit.chunk_id, {})["vector"] = position
        chunks[hit.chunk_id] = hit
    for position, hit in enumerate(bm25_hits, start=1):
        ranks.setdefault(hit.chunk_id, {})["bm25"] = position
        chunks.setdefault(hit.chunk_id, hit)

    fused: list[FusedChunk] = []
    for chunk_id, positions in ranks.items():
        score = sum(1.0 / (k + rank) for rank in positions.values())
        fused.append(
            FusedChunk(
                chunk=chunks[chunk_id],
                rrf_score=score,
                vector_rank=positions.get("vector"),
                bm25_rank=positions.get("bm25"),
            )
        )

    fused.sort(key=lambda f: (-f.rrf_score, f.chunk.chunk_id))
    return fused[:top_k] if top_k else fused
