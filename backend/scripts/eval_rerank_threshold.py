"""min_rerank_score 阈值敏感性分析。

背景: eval_bad_cases.md 记录了 q14/q22/q25 三条 hard 案例的 Rerank 后最佳命中分数
(0.0504/0.1012/0.1343) 全部低于生产阈值 min_rerank_score=0.15，导致 relevance_filter
把命中片段清空，context_recall/precision 双零。本脚本跑全部 38 条案例在多个候选阈值下
的实际表现，量化"调低阈值能救回多少 hard 案例"与"会不会连带放进更多噪声片段"。

运行: python scripts/eval_rerank_threshold.py
"""

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT / "data" / "eval_set.json"
DOCS_DIR = ROOT / "data" / "docs"

CANDIDATE_K = 20
TOP_N = 5
# 待评估的候选阈值，覆盖当前生产值 0.15 两侧
THRESHOLDS = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


def _bootstrap_env() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="rerank-thresh-"))
    os.environ.update(
        {
            "API_KEY": "eval",
            "STARTUP_PROBE_EXTERNAL": "false",
            "DATABASE_URL": f"sqlite:///{(workdir / 'eval.db').as_posix()}",
            "QDRANT_PATH": str(workdir / "qdrant"),
            "ENABLE_QUERY_REWRITE": "false",
            "CHUNK_STRATEGY": "markdown",
            "RETRIEVE_TOP_K": str(CANDIDATE_K),
            "RERANK_TOP_N": str(TOP_N),
            "MIN_RERANK_SCORE": "0.0",
        }
    )
    return workdir


def main() -> int:
    workdir = _bootstrap_env()
    try:
        return _run()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@dataclass(slots=True)
class CaseScore:
    case_id: str
    difficulty: str
    query: str
    # (rank, rerank_score, is_hit) 覆盖 Top-N 全部候选，不只是命中的那条 ——
    # 计算"阈值放宽会带进多少噪声"需要非命中候选的分数。
    chunks: list[tuple[int, float, bool]]

    @property
    def best_score(self) -> float | None:
        hits = [s for _, s, is_hit in self.chunks if is_hit]
        return max(hits) if hits else None

    @property
    def best_rank(self) -> int | None:
        ranks = [r for r, _, is_hit in self.chunks if is_hit]
        return min(ranks) if ranks else None


def _is_hit(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _run() -> int:
    from app.knowledge.ingest import KnowledgeIngestor
    from app.llm.factory import get_embedding_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.reranker import get_reranker
    from app.rag.retriever import Retriever
    from app.rag.vector_store import get_vector_store
    from app.storage.db import init_db, session_scope

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        print(f"no documents in {DOCS_DIR}", file=sys.stderr)
        return 1

    init_db()
    store = get_vector_store()
    bm25 = get_bm25_index()
    embedding = get_embedding_client()
    ingestor = KnowledgeIngestor(
        vector_store=store, embedding_client=embedding, bm25_index=bm25
    )

    print(f"ingesting {len(docs)} documents ...")
    with session_scope() as session:
        for doc in docs:
            ingestor.ingest_text(
                session,
                title=doc.stem.replace("_", " "),
                content=doc.read_text(encoding="utf-8"),
                source="file",
                source_ref=doc.name,
            )
    print(f"  vectors={store.count()} bm25={bm25.size}\n")

    retriever = Retriever(
        vector_store=store,
        embedding_client=embedding,
        bm25_index=bm25,
        reranker=get_reranker(),
        llm_client=None,
    )

    # 只跑一次真实检索(min_score=0.0，拿到未经阈值过滤的完整候选与真实 rerank 分数)，
    # 后面对每个候选阈值做纯内存模拟过滤，不重复调用 embedding/rerank 模型。
    scored: list[CaseScore] = []
    print(f"running retrieval for {len(cases)} cases (min_score=0.0, top_n={TOP_N}) ...")
    for case in cases:
        outcome = retriever.retrieve(
            case["query"],
            top_k=CANDIDATE_K,
            top_n=TOP_N,
            enable_rewrite=False,
            min_score=0.0,
        )
        keywords = case["expected_keywords"]
        chunk_rows = [
            (i, r.rerank_score, _is_hit(r.chunk.contextual_text, keywords))
            for i, r in enumerate(outcome.chunks, start=1)
        ]
        scored.append(
            CaseScore(
                case_id=case["id"],
                difficulty=case.get("difficulty", "unknown"),
                query=case["query"],
                chunks=chunk_rows,
            )
        )

    _print_report(scored)
    return 0


@dataclass(slots=True)
class ThresholdRow:
    threshold: float
    hit_rate: float
    hard_hit_rate: float
    empty_context_rate: float
    avg_kept: float
    avg_noise_kept: float  # 阈值放宽后混进来的、不含期望关键词的候选均数


def _simulate(scored: list[CaseScore], threshold: float) -> ThresholdRow:
    total = len(scored)
    hard = [c for c in scored if c.difficulty == "hard"]
    hits = 0
    hard_hits = 0
    empties = 0
    kept_counts: list[int] = []
    noise_counts: list[int] = []

    for case in scored:
        kept = [(r, s, h) for r, s, h in case.chunks if s >= threshold]
        kept_counts.append(len(kept))
        noise_counts.append(sum(1 for _, _, h in kept if not h))
        if not kept:
            empties += 1
        if any(h for _, _, h in kept):
            hits += 1
            if case.difficulty == "hard":
                hard_hits += 1

    return ThresholdRow(
        threshold=threshold,
        hit_rate=hits / total,
        hard_hit_rate=(hard_hits / len(hard)) if hard else float("nan"),
        empty_context_rate=empties / total,
        avg_kept=sum(kept_counts) / total,
        avg_noise_kept=sum(noise_counts) / total,
    )


def _print_report(scored: list[CaseScore]) -> None:
    print("=" * 92)
    print(f"min_rerank_score 阈值敏感性分析 ({len(scored)} 条案例，Top-{TOP_N} 候选)")
    print("hit_rate: 阈值过滤后 Top-N 内仍保留至少一个命中片段的案例占比")
    print("empty_context_rate: 阈值过滤后 Top-N 全部被清空(context 为空)的案例占比")
    print("avg_noise_kept: 每条案例平均放进多少个不含期望关键词的候选(阈值越低越多)")
    print("=" * 92)
    header = (
        f"{'阈值':>6} {'hit_rate':>9} {'hard命中':>9} {'空context':>10} "
        f"{'均保留数':>9} {'均噪声数':>9}"
    )
    print(header)
    print("-" * 92)
    for t in THRESHOLDS:
        row = _simulate(scored, t)
        marker = " <= 当前生产值" if abs(t - 0.15) < 1e-9 else ""
        print(
            f"{row.threshold:>6.2f} {row.hit_rate:>8.1%} {row.hard_hit_rate:>8.1%} "
            f"{row.empty_context_rate:>9.1%} {row.avg_kept:>9.2f} "
            f"{row.avg_noise_kept:>9.2f}{marker}"
        )
    print("-" * 92)

    # 逐条列出 hard 案例在最佳命中片段上的真实分数，方便对照 eval_bad_cases.md 的记录
    print("\nhard 案例最佳命中分数明细 (score=None 表示 Top-N 内完全没命中期望关键词):")
    print(f"{'case':<6} {'best_rank':>10} {'best_score':>11}  query")
    print("-" * 92)
    for case in scored:
        if case.difficulty != "hard":
            continue
        score_txt = f"{case.best_score:.4f}" if case.best_score is not None else "None"
        rank_txt = str(case.best_rank) if case.best_rank is not None else "-"
        print(f"{case.case_id:<6} {rank_txt:>10} {score_txt:>11}  {case.query}")

    # 找出"调低阈值到某个点才能救回"的案例，即 best_score 落在 (0, 0.15) 区间
    borderline = sorted(
        (c for c in scored if c.best_score is not None and c.best_score < 0.15),
        key=lambda c: c.best_score,
    )
    if borderline:
        print(
            f"\n当前阈值 0.15 下会被误清空、但其实命中了期望关键词的案例 ({len(borderline)} 条):"
        )
        for c in borderline:
            print(f"  {c.case_id} ({c.difficulty}): best_score={c.best_score:.4f}")


if __name__ == "__main__":
    sys.exit(main())
