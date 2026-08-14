import re
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
    # 仅用于 trace/离线诊断。保留各层的原始候选，使阈值过滤问题不会被
    # 误判为向量、BM25、RRF 或 rerank 本身没有召回。
    vector_hits: list[ScoredChunk] = field(default_factory=list)
    bm25_hits: list[ScoredChunk] = field(default_factory=list)
    fused_chunks: list[FusedChunk] = field(default_factory=list)
    reranked_chunks: list[RerankedChunk] = field(default_factory=list)
    # 仅用于 trace/离线评测诊断，保留阈值过滤前的 Rerank 输出；回答和引用仍只
    # 使用 chunks，不能让被阈值拒绝的片段重新进入生成上下文。
    pre_filter_chunks: list[RerankedChunk] = field(default_factory=list)

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
        reranked_chunks = list(reranked)

        # 关联召回：为 Top-N 中的主题准备同文档候选，阈值后只允许锚定主题补证。
        if reranked:
            reranked = self._associated_recall(reranked, stages)

        # 只有真正经过 rerank 打分才做阈值过滤：降级路径的分数是 RRF 值，量纲不同
        pre_filter_chunks = list(reranked)
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

        # 最终证据必须能解释命中的故障，而不只是复述一个现象。仅从已经进入
        # Rerank Top-N 的候选中，补回与阈值命中项同文档、同故障主题的兄弟小节。
        # 阈值命中项仍是锚点；没有锚点、跨文档或跨主题的低分内容都不能进入回答。
        reranked = _restore_topic_context(
            candidates=pre_filter_chunks,
            kept=reranked,
            stages=stages,
        )

        # 阈值只决定主要相关证据。若用户在问题里明确写出了英文资源/状态标识，
        # 则允许从已经进入 Rerank Top-N 的候选中补回标题精确命中的同文档片段。
        # 这不是绕过召回或扩展任意低分内容：候选必须已被 Rerank 选入 Top-N，
        # 标题必须包含用户原文中的标识，且文档必须已有一条通过阈值的证据。
        reranked = _restore_explicit_heading_context(
            query=query,
            candidates=reranked_chunks,
            kept=reranked,
            stages=stages,
        )

        return RetrievalResult(
            query_rewrite=rewrite,
            chunks=reranked,
            stages=stages,
            hybrid_enabled=use_hybrid,
            rerank_applied=rerank_applied,
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            fused_chunks=fused,
            reranked_chunks=reranked_chunks,
            pre_filter_chunks=pre_filter_chunks,
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
        """加载 Top-N 所在文档的候选，最终只恢复阈值锚定的同主题兄弟。"""
        started = time.perf_counter()

        candidate_docs = {
            r.chunk.document_id
            for r in reranked
            if r.chunk.document_id
        }

        if not candidate_docs:
            stages.append(
                RetrievalStage(
                    name="associated_recall",
                    hit_count=0,
                    elapsed_ms=_ms_since(started),
                    note="no_candidate_documents",
                )
            )
            return reranked

        # 召回这些文档的所有 chunk
        associated: list[ScoredChunk] = []
        for doc_id in candidate_docs:
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
                    note=f"checked_docs={len(candidate_docs)} all_duplicates",
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
                note=f"docs={len(candidate_docs)} added={len(associated_reranked)}",
                top_chunk_ids=[c.chunk_id for c in new_chunks[:5]],
            )
        )

        # 关联召回的 chunk 追加到结果末尾（保持原有排序）
        return reranked + associated_reranked


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)




def _restore_explicit_heading_context(
    *,
    query: str,
    candidates: list[RerankedChunk],
    kept: list[RerankedChunk],
    stages: list[RetrievalStage],
) -> list[RerankedChunk]:
    """Restore Rerank Top-N context anchored by identifiers in the user query."""
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)
    }
    if not tokens or not kept:
        stages.append(
            RetrievalStage(
                name="explicit_heading_context",
                hit_count=0,
                elapsed_ms=0,
                note="no_explicit_token_or_kept_context",
            )
        )
        return kept

    kept_ids = {item.chunk.chunk_id for item in kept}
    kept_documents = {item.chunk.document_id for item in kept}
    additions: list[RerankedChunk] = []
    for item in candidates:
        if item.chunk.chunk_id in kept_ids:
            continue
        if item.chunk.document_id not in kept_documents:
            continue
        heading = " > ".join(item.chunk.heading_path).casefold()
        if any(token in heading for token in tokens):
            additions.append(item)
            kept_ids.add(item.chunk.chunk_id)

    stages.append(
        RetrievalStage(
            name="explicit_heading_context",
            hit_count=len(additions),
            elapsed_ms=0,
            note=(
                "tokens=" + ",".join(sorted(tokens))
                if additions
                else "no_matching_rerank_top_n_heading"
            ),
            top_chunk_ids=[item.chunk.chunk_id for item in additions[:5]],
        )
    )
    return kept + additions


def _restore_topic_context(
    *,
    candidates: list[RerankedChunk],
    kept: list[RerankedChunk],
    stages: list[RetrievalStage],
) -> list[RerankedChunk]:
    """Restore Top-N siblings only when a threshold-passing topic anchor exists."""
    if not kept:
        stages.append(
            RetrievalStage(
                name="topic_context",
                hit_count=0,
                elapsed_ms=0,
                note="no_kept_context",
            )
        )
        return kept

    anchored_topics = {
        (item.chunk.document_id, topic)
        for item in kept
        if (topic := _fault_topic(item.chunk.heading_path)) is not None
    }
    kept_ids = {item.chunk.chunk_id for item in kept}
    additions: list[RerankedChunk] = []
    for item in candidates:
        if item.chunk.chunk_id in kept_ids:
            continue
        topic = _fault_topic(item.chunk.heading_path)
        if (
            topic is None
            or (item.chunk.document_id, topic) not in anchored_topics
        ):
            continue
        additions.append(item)
        kept_ids.add(item.chunk.chunk_id)

    stages.append(
        RetrievalStage(
            name="topic_context",
            hit_count=len(additions),
            elapsed_ms=0,
            note=(
                f"anchored_topics={len(anchored_topics)}"
                if additions
                else "no_matching_rerank_top_n_sibling"
            ),
            top_chunk_ids=[item.chunk.chunk_id for item in additions[:5]],
        )
    )
    return kept + additions


def _fault_topic(heading_path: list[str]) -> tuple[str, ...] | None:
    # Markdown 语料的最后一级是“现象/根因/处理步骤”等证据类别，之前的完整
    # 标题链才标识故障主题。至少保留文档和主题两级，避免把整篇文档混为一组。
    if len(heading_path) < 3:
        return None
    return tuple(part.strip().casefold() for part in heading_path[:-1])
