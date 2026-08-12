import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.errors import AppError
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient
from app.rag.bm25_index import BM25Index
from app.rag.fusion import FusedChunk, reciprocal_rank_fusion
from app.rag.query_rewrite import QueryRewriteResult, rewrite_query
from app.rag.reranker import RerankedChunk, Reranker
from app.rag.vector_store import ScoredChunk, VectorStore


@dataclass(slots=True)
class RetrievalStage:
    name: str
    hit_count: int
    elapsed_ms: int
    top_chunk_ids: list[str] = field(default_factory=list)
    note: str | None = None


@dataclass(slots=True)
class RetrievalResult:
    query_rewrite: QueryRewriteResult
    chunks: list[RerankedChunk]
    stages: list[RetrievalStage]
    hybrid_enabled: bool
    rerank_applied: bool

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
    """检索链路：(可选)查询改写 → 向量召回 ∥ BM25召回 → RRF融合 → Rerank。"""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        bm25_index: BM25Index,
        reranker: Reranker,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._store = vector_store
        self._embed = embedding_client
        self._bm25 = bm25_index
        self._reranker = reranker
        self._llm = llm_client

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        enable_hybrid: bool | None = None,
        enable_rewrite: bool | None = None,
        enable_rerank: bool = True,
        history_snippet: str | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        settings = get_settings()
        top_k = top_k or settings.retrieve_top_k
        top_n = top_n or settings.rerank_top_n
        use_hybrid = (
            settings.enable_hybrid_retrieve if enable_hybrid is None else enable_hybrid
        )
        use_rewrite = (
            settings.enable_query_rewrite if enable_rewrite is None else enable_rewrite
        )
        stages: list[RetrievalStage] = []

        started = time.perf_counter()
        if use_rewrite and self._llm is not None:
            rewrite = rewrite_query(
                query, self._llm, enabled=True, history_snippet=history_snippet
            )
        else:
            rewrite = QueryRewriteResult(
                original=query,
                rewritten=query,
                applied=False,
                skip_reason="disabled" if not use_rewrite else "no_llm_client",
            )
        stages.append(
            RetrievalStage(
                name="query_rewrite",
                hit_count=1 if rewrite.applied else 0,
                elapsed_ms=_ms_since(started),
                note=rewrite.skip_reason,
            )
        )
        search_query = rewrite.rewritten

        vector_hits = self._vector_search(search_query, top_k, stages)
        bm25_hits = (
            self._bm25_search(search_query, rewrite.keywords, top_k, stages)
            if use_hybrid
            else []
        )
        if not use_hybrid:
            stages.append(
                RetrievalStage(
                    name="bm25_search", hit_count=0, elapsed_ms=0, note="disabled"
                )
            )

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            vector_hits, bm25_hits, k=settings.rrf_k, top_k=top_k
        )
        stages.append(
            RetrievalStage(
                name="rrf_fusion",
                hit_count=len(fused),
                elapsed_ms=_ms_since(started),
                top_chunk_ids=[f.chunk.chunk_id for f in fused[:5]],
                note=None if use_hybrid else "vector_only",
            )
        )

        reranked, rerank_applied = self._rerank(
            search_query, fused, top_n, stages, enable_rerank
        )

        # 关联召回：命中"处理步骤" chunk 时，连带召回同文档的"现象/根因"段
        if reranked:
            reranked = self._associated_recall(reranked, stages)

        # 只有真正经过 rerank 打分才做阈值过滤：降级路径的分数是 RRF 值，量纲不同
        if rerank_applied and reranked:
            threshold = (
                settings.min_rerank_score
                if min_score is None
                else min_score
            )
            kept = [r for r in reranked if r.rerank_score >= threshold]
            stages.append(
                RetrievalStage(
                    name="relevance_filter",
                    hit_count=len(kept),
                    elapsed_ms=0,
                    note=f"threshold={threshold} dropped={len(reranked) - len(kept)}",
                )
            )
            reranked = kept

        return RetrievalResult(
            query_rewrite=rewrite,
            chunks=reranked,
            stages=stages,
            hybrid_enabled=use_hybrid,
            rerank_applied=rerank_applied,
        )

    def _vector_search(
        self, query: str, top_k: int, stages: list[RetrievalStage]
    ) -> list[ScoredChunk]:
        started = time.perf_counter()
        try:
            hits = self._store.search(self._embed.embed_one(query), top_k)
            note = None
        except AppError as exc:
            hits, note = [], f"failed:{exc.code.value}"
        stages.append(
            RetrievalStage(
                name="vector_search",
                hit_count=len(hits),
                elapsed_ms=_ms_since(started),
                top_chunk_ids=[h.chunk_id for h in hits[:5]],
                note=note,
            )
        )
        return hits

    def _bm25_search(
        self,
        query: str,
        keywords: list[str],
        top_k: int,
        stages: list[RetrievalStage],
    ) -> list[ScoredChunk]:
        started = time.perf_counter()
        keyword_query = f"{query} {' '.join(keywords)}" if keywords else query
        hits = self._bm25.search(keyword_query, top_k)
        stages.append(
            RetrievalStage(
                name="bm25_search",
                hit_count=len(hits),
                elapsed_ms=_ms_since(started),
                top_chunk_ids=[h.chunk_id for h in hits[:5]],
                note=None if self._bm25.size else "index_empty",
            )
        )
        return hits

    def _rerank(
        self,
        query: str,
        fused: list[FusedChunk],
        top_n: int,
        stages: list[RetrievalStage],
        enable_rerank: bool,
    ) -> tuple[list[RerankedChunk], bool]:
        candidates = [f.chunk for f in fused]
        if not enable_rerank or not candidates:
            stages.append(
                RetrievalStage(
                    name="rerank",
                    hit_count=0,
                    elapsed_ms=0,
                    note="disabled" if not enable_rerank else "no_candidates",
                )
            )
            truncated = [
                RerankedChunk(
                    chunk=f.chunk,
                    rerank_score=f.rrf_score,
                    rank_before=i + 1,
                    rank_after=i + 1,
                )
                for i, f in enumerate(fused[:top_n])
            ]
            return truncated, False

        started = time.perf_counter()
        try:
            reranked = self._reranker.rerank(query, candidates, top_n)
            applied, note = True, None
        except AppError as exc:
            # Rerank 挂了不能让整条链路失败，退回融合顺序
            reranked = [
                RerankedChunk(
                    chunk=f.chunk,
                    rerank_score=f.rrf_score,
                    rank_before=i + 1,
                    rank_after=i + 1,
                )
                for i, f in enumerate(fused[:top_n])
            ]
            applied, note = False, f"degraded:{exc.code.value}"
        stages.append(
            RetrievalStage(
                name="rerank",
                hit_count=len(reranked),
                elapsed_ms=_ms_since(started),
                top_chunk_ids=[r.chunk.chunk_id for r in reranked[:5]],
                note=note,
            )
        )
        return reranked, applied


    def _associated_recall(
        self, reranked: list[RerankedChunk], stages: list[RetrievalStage]
    ) -> list[RerankedChunk]:
        """关联召回：命中"处理步骤" chunk 时，连带召回同文档的其他段落。

        K8s 文档结构"现象/根因/处理步骤"三段式，路由看到孤立的操作指令易误判。
        补充同文档的"现象/根因"段，让路由能判断"操作步骤"是否匹配当前问题。
        """
        started = time.perf_counter()

        # 收集所有标记为 is_procedural 的 chunk 的 document_id
        procedural_docs = {
            r.chunk.document_id
            for r in reranked
            if r.chunk.is_procedural and r.chunk.document_id
        }

        if not procedural_docs:
            stages.append(
                RetrievalStage(
                    name="associated_recall",
                    hit_count=0,
                    elapsed_ms=_ms_since(started),
                    note="no_procedural_chunks",
                )
            )
            return reranked

        # 召回这些文档的所有 chunk
        associated: list[ScoredChunk] = []
        for doc_id in procedural_docs:
            try:
                chunks = self._store.get_chunks_by_document(doc_id)
                associated.extend(chunks)
            except AppError:
                continue  # 单个文档失败不阻塞整体

        # 去重：已经在 reranked 里的不重复加
        existing_ids = {r.chunk.chunk_id for r in reranked}
        new_chunks = [c for c in associated if c.chunk_id not in existing_ids]

        if not new_chunks:
            stages.append(
                RetrievalStage(
                    name="associated_recall",
                    hit_count=0,
                    elapsed_ms=_ms_since(started),
                    note=f"checked_docs={len(procedural_docs)} all_duplicates",
                )
            )
            return reranked

        # 包装成 RerankedChunk，score 使用 0.0（表示关联召回，非相似度排序）
        # rank_before/rank_after 都填 999 表示"补充材料"
        associated_reranked = [
            RerankedChunk(
                chunk=c,
                rerank_score=0.0,
                rank_before=999,
                rank_after=999,
            )
            for c in new_chunks
        ]

        stages.append(
            RetrievalStage(
                name="associated_recall",
                hit_count=len(associated_reranked),
                elapsed_ms=_ms_since(started),
                note=f"docs={len(procedural_docs)} added={len(associated_reranked)}",
                top_chunk_ids=[c.chunk_id for c in new_chunks[:5]],
            )
        )

        # 关联召回的 chunk 追加到结果末尾（保持原有排序）
        return reranked + associated_reranked


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
