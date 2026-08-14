"""检索质量评估：纯向量 / 混合检索 / 混合+Rerank 三组对比。

用真实 Embedding 与 Rerank 模型跑，需要先确保对应服务可用。
运行: python scripts/eval_retrieval.py [--fake]
  --fake  用替身模型跑通流程（不产生可用于简历的指标）
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT / "data" / "eval_set.json"
DOCS_DIR = ROOT / "data" / "docs_k8s"


@dataclass
class ConfigResult:
    label: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    avg_latency_ms: float
    misses: list[str]
    # 命中片段的平均排名。对重排敏感——Hit@K 只看是否在集合内，
    # 这个指标能反映 Rerank 把正确答案往前提了多少。越小越好。
    avg_best_rank: float = float("nan")
    # hard 子集（口语化/易混淆查询）的指标。easy 子集基线已接近满分，
    # 整体均值会把 Rerank 的贡献稀释掉，分层看才有区分度。
    hard_mrr: float = float("nan")
    hard_hit_at_3: float = float("nan")
    hard_count: int = 0
    # The runtime can intentionally fall back to RRF when BGE is unavailable.
    # An evaluation must not publish that fallback as a Rerank result.
    rerank_verified: bool = True
    rerank_failure_reasons: list[str] = field(default_factory=list)
    evaluated_case_count: int = 0


def _default_workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "acceptance-evidence" / f"retrieval-{stamp}"


def _bootstrap_env(fake: bool, workdir: Path) -> Path:
    """配置隔离评测工作目录，并保留现场供验收证据复查。"""
    if workdir.exists() and any(workdir.iterdir()):
        raise RuntimeError(f"评测工作目录必须为空，拒绝复用已有证据：{workdir}")
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "JWT_SECRET_KEY": "evaluation-jwt-secret-not-for-production",
            "STARTUP_PROBE_EXTERNAL": "false",
            "DATABASE_URL": f"sqlite:///{(workdir / 'eval.db').as_posix()}",
            "QDRANT_PATH": str(workdir / "qdrant"),
            "ENABLE_QUERY_REWRITE": "false",
            "CHUNK_STRATEGY": "markdown",
            # 与下方 CANDIDATE_K / TOP_N 保持一致；retrieve() 里也显式传参
            "RETRIEVE_TOP_K": "20",
            "RERANK_TOP_N": "5",
            "MIN_RERANK_SCORE": "0.0",
        }
    )
    if fake:
        os.environ.update(
            {
                "LLM_PROVIDER": "ollama",
                "EMBEDDING_PROVIDER": "ollama",
                "OLLAMA_EMBEDDING_MODEL": "fake-embedding",
                "OLLAMA_EMBEDDING_DIM": "64",
            }
        )
    return workdir


def _document_manifest(docs: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "name": doc.name,
            "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
        }
        for doc in docs
    ]


# 召回宽度与最终输出宽度。前者必须显著大于后者，Rerank 才有筛选空间。
CANDIDATE_K = 20
TOP_N = 5

# v1.0 发布门槛。hard 子集显式冻结，避免在看到结果后通过修改 difficulty
# 标签缩小或替换困难样本；全量评测仍保留全部案例并报告相对 MRR 变化。
FIXED_HARD_CASE_IDS = (
    "q11", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20",
    "q21", "q22", "q23", "q24", "q25", "q27", "q29", "q30",
)
MIN_FULL_HIT_AT_5 = 0.85
MIN_FULL_MRR = 0.80
MIN_HARD_MRR_IMPROVEMENT = 0.15


def _is_hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def _validate_hard_subset(cases: list[dict[str, object]]) -> list[str]:
    hard_case_ids = [
        str(case["id"])
        for case in cases
        if case.get("difficulty") == "hard"
    ]
    if hard_case_ids != list(FIXED_HARD_CASE_IDS):
        raise RuntimeError(
            "hard evaluation subset drifted: "
            f"expected={list(FIXED_HARD_CASE_IDS)}, actual={hard_case_ids}"
        )
    return hard_case_ids


def _release_gates(results: list[ConfigResult]) -> dict[str, object]:
    base, _, reranked = results
    checks = {
        "full_hit_at_5": reranked.hit_at_5 >= MIN_FULL_HIT_AT_5,
        "full_mrr": reranked.mrr >= MIN_FULL_MRR,
        "full_hit_at_5_non_regression": reranked.hit_at_5 >= base.hit_at_5,
        "hard_mrr_improvement": (
            reranked.hard_mrr - base.hard_mrr
        ) >= MIN_HARD_MRR_IMPROVEMENT,
    }
    return {
        "passed": all(checks.values()) and reranked.rerank_verified,
        "checks": checks,
        "thresholds": {
            "min_full_hit_at_5": MIN_FULL_HIT_AT_5,
            "min_full_mrr": MIN_FULL_MRR,
            "full_hit_at_5_must_not_regress": True,
            "min_hard_mrr_improvement": MIN_HARD_MRR_IMPROVEMENT,
        },
        "observed": {
            "full_hit_at_5": reranked.hit_at_5,
            "full_mrr": reranked.mrr,
            "full_mrr_improvement": reranked.mrr - base.mrr,
            "hard_mrr": reranked.hard_mrr,
            "hard_mrr_improvement": reranked.hard_mrr - base.hard_mrr,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="用替身模型跑通流程")
    parser.add_argument(
        "--docs-dir", type=Path, default=DOCS_DIR,
        help="知识库文档目录，默认 data/docs。切换行业时指向另一套文档即可，"
        "不需要改代码——例如 data/docs_education 是教育行业的最小示例",
    )
    parser.add_argument(
        "--eval-set", type=Path, default=EVAL_SET,
        help="标注评估集路径，默认 data/eval_set.json，需与 --docs-dir 配套",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="隔离评测数据目录；默认保存到被忽略的 acceptance-evidence，且不自动清理",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="机器可读 JSON 报告路径；默认写入 workdir/retrieval-report.json",
    )
    args = parser.parse_args()

    workdir = _bootstrap_env(args.fake, args.workdir or _default_workdir())
    return _run(
        args.fake,
        args.docs_dir,
        args.eval_set,
        workdir,
        args.report_file or workdir / "retrieval-report.json",
    )


def _run(fake: bool, docs_dir: Path, eval_set: Path, workdir: Path, report_file: Path) -> int:
    import time

    from app.knowledge.ingest import KnowledgeIngestor
    from app.llm.factory import get_embedding_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.reranker import IdentityReranker, get_reranker
    from app.rag.retriever import Retriever
    from app.rag.vector_store import get_vector_store
    from app.storage.db import init_db, session_scope

    if fake:
        sys.path.insert(0, str(ROOT))
        from tests.fakes import FakeEmbeddingClient, KeywordReranker

        embedding = FakeEmbeddingClient()
        real_reranker = KeywordReranker()
    else:
        embedding = get_embedding_client()
        real_reranker = get_reranker()

    eval_bytes = eval_set.read_bytes()
    all_cases = json.loads(eval_bytes.decode("utf-8"))["cases"]
    hard_case_ids = _validate_hard_subset(all_cases)
    # 检索指标只统计知识查询。工具案例用于端到端路由门禁，不能因为工具字段
    # 恰好与某段文档重合而抬高或压低 Hit@K/MRR。
    cases = [case for case in all_cases if case.get("difficulty") != "tool"]
    excluded_tool_case_ids = [
        case["id"] for case in all_cases if case.get("difficulty") == "tool"
    ]
    docs = sorted(docs_dir.glob("*.md"))
    if not docs:
        print(f"no documents in {docs_dir}", file=sys.stderr)
        return 1

    init_db()
    store = get_vector_store()
    bm25 = get_bm25_index()
    ingestor = KnowledgeIngestor(
        vector_store=store, embedding_client=embedding, bm25_index=bm25
    )

    print(f"ingesting {len(docs)} documents ...")
    with session_scope() as session:
        for doc in docs:
            result = ingestor.ingest_text(
                session,
                title=doc.stem.replace("_", " "),
                content=doc.read_text(encoding="utf-8"),
                source="file",
                source_ref=doc.name,
            )
            print(f"  {doc.name}: {result.chunk_count} chunks")
    print(f"  vectors={store.count()} bm25={bm25.size}\n")

    # 召回宽度统一为 CANDIDATE_K，最终输出统一为 TOP_N。
    # 三组配置必须用同一组参数，否则对比不公平。
    #
    # 关键: CANDIDATE_K 必须显著大于 TOP_N，Rerank 才有筛选空间。
    # 若召回 5 条、输出 5 条，重排只是调整这 5 条的顺序，不改变集合成员，
    # Hit@5 必然不动 —— 这是评估设计问题，不是 Rerank 无效。
    configs = [
        ("A. 纯向量检索", dict(enable_hybrid=False, enable_rerank=False), IdentityReranker()),
        ("B. 混合检索(向量+BM25)", dict(enable_hybrid=True, enable_rerank=False), IdentityReranker()),
        ("C. 混合检索 + Rerank", dict(enable_hybrid=True, enable_rerank=True), real_reranker),
    ]

    results: list[ConfigResult] = []
    for label, flags, reranker_impl in configs:
        retriever = Retriever(
            vector_store=store,
            embedding_client=embedding,
            bm25_index=bm25,
            reranker=reranker_impl,
            llm_client=None,
        )
        hits = {1: 0, 3: 0, 5: 0}
        reciprocal = 0.0
        latencies: list[float] = []
        misses: list[str] = []
        best_ranks: list[int] = []
        # 按难度分层：easy 子集基线已接近满分，Rerank 的价值只在 hard 子集显现
        hard_reciprocal = 0.0
        hard_total = 0
        hard_ranks: list[int] = []
        rerank_failures: list[str] = []

        for case in cases:
            started = time.perf_counter()
            outcome = retriever.retrieve(
                case["query"],
                top_k=CANDIDATE_K,
                top_n=TOP_N,
                enable_rewrite=False,
                min_score=0.0,
                **flags,
            )
            latencies.append((time.perf_counter() - started) * 1000)

            if flags["enable_rerank"] and not outcome.rerank_applied:
                rerank_stage = next(
                    (stage for stage in outcome.stages if stage.name == "rerank"),
                    None,
                )
                note = rerank_stage.note if rerank_stage else "missing_rerank_stage"
                rerank_failures.append(f"{case['id']}: {note or 'not_applied'}")
                # Do not mix the RRF fallback rank into Rerank metrics.
                continue

            is_hard = case.get("difficulty") == "hard"
            if is_hard:
                hard_total += 1

            # 用 contextual_text 判定：关键词可能只出现在标题里，
            # 与检索侧使用的文本表示保持一致才不会低估命中。
            ranks = [
                i
                for i, chunk in enumerate(outcome.chunks, start=1)
                if _is_hit(chunk.chunk.contextual_text, case["expected_keywords"])
            ]
            if ranks:
                best = min(ranks)
                reciprocal += 1.0 / best
                best_ranks.append(best)
                for k in hits:
                    if best <= k:
                        hits[k] += 1
                if is_hard:
                    hard_reciprocal += 1.0 / best
                    hard_ranks.append(best)
            else:
                misses.append(case["id"])

        total = len(cases)
        results.append(
            ConfigResult(
                label=label,
                hit_at_1=hits[1] / total,
                hit_at_3=hits[3] / total,
                hit_at_5=hits[5] / total,
                mrr=reciprocal / total,
                avg_latency_ms=sum(latencies) / len(latencies),
                misses=misses,
                avg_best_rank=(
                    sum(best_ranks) / len(best_ranks) if best_ranks else float("nan")
                ),
                hard_mrr=(hard_reciprocal / hard_total if hard_total else float("nan")),
                hard_hit_at_3=(
                    sum(1 for r in hard_ranks if r <= 3) / hard_total
                    if hard_total
                    else float("nan")
                ),
                hard_count=hard_total,
                rerank_verified=not rerank_failures,
                rerank_failure_reasons=rerank_failures,
                evaluated_case_count=total - len(rerank_failures),
            )
        )

    _print_report(results, len(cases), fake)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    release_gates = _release_gates(results)
    report_file.write_text(
        json.dumps(
            {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "fake" if fake else "real",
                "workdir": str(workdir),
                "documents": _document_manifest(docs),
                "document_count": len(docs),
                "chunk_count": store.count(),
                "bm25_chunk_count": bm25.size,
                "models": {
                    "embedding": type(embedding).__name__,
                    "reranker": type(real_reranker).__name__,
                },
                "eval_set": {
                    "path": str(eval_set),
                    "sha256": hashlib.sha256(eval_bytes).hexdigest(),
                    "case_ids": [case["id"] for case in all_cases],
                    "hard_case_ids": hard_case_ids,
                    "excluded_tool_case_ids": excluded_tool_case_ids,
                },
                "source_case_count": len(all_cases),
                "case_count": len(cases),
                "results": [asdict(result) for result in results],
                "release_gates": release_gates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"machine_report={report_file}")
    failed = [result for result in results if not result.rerank_verified]
    if failed:
        for result in failed:
            print(
                f"ERROR: {result.label} was degraded; refusing to publish Rerank metrics: "
                f"{'; '.join(result.rerank_failure_reasons[:3])}",
                file=sys.stderr,
            )
        return 1
    if not fake and not release_gates["passed"]:
        failed_checks = [
            name
            for name, passed in release_gates["checks"].items()
            if not passed
        ]
        print(
            "ERROR: retrieval release gates failed: " + ", ".join(failed_checks),
            file=sys.stderr,
        )
        return 2
    return 0


def _print_report(results: list[ConfigResult], total: int, fake: bool) -> None:
    print("=" * 78)
    print(f"检索质量对比 ({total} 条标注查询)")
    print(f"召回 {CANDIDATE_K} 条候选 -> 最终取 Top-{TOP_N}；命中判定: 输出中出现期望关键词")
    if fake:
        print("!! 使用替身模型，指标仅用于验证流程，不可作为真实效果 !!")
    print("=" * 78)
    header = (
        f"{'配置':<26} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} "
        f"{'MRR':>7} {'均排名':>7} {'延迟ms':>8}"
    )
    print(header)
    print("-" * 78)
    for r in results:
        print(
            f"{r.label:<26} {r.hit_at_1:>6.1%} {r.hit_at_3:>6.1%} "
            f"{r.hit_at_5:>6.1%} {r.mrr:>7.3f} {r.avg_best_rank:>7.2f} "
            f"{r.avg_latency_ms:>8.1f}"
        )
    print("-" * 78)

    base, hybrid, reranked = results
    if not reranked.rerank_verified:
        print("\n!! Rerank was degraded; its evaluation metrics are invalid. !!")
        return

    if base.hard_count:
        print(f"\nhard 子集（{base.hard_count} 条口语化/易混淆查询）:")
        print(f"{'配置':<26} {'Hit@3':>7} {'MRR':>7}")
        print("-" * 44)
        for r in results:
            print(f"{r.label:<26} {r.hard_hit_at_3:>6.1%} {r.hard_mrr:>7.3f}")
        print("-" * 44)

    print("\n提升幅度:")
    print(
        f"  混合检索 vs 纯向量:      Hit@3 {_delta(base.hit_at_3, hybrid.hit_at_3)}  "
        f"MRR {_delta(base.mrr, hybrid.mrr, pct=False)}"
    )
    print(
        f"  混合+Rerank vs 纯向量:   Hit@3 {_delta(base.hit_at_3, reranked.hit_at_3)}  "
        f"MRR {_delta(base.mrr, reranked.mrr, pct=False)}"
    )
    print(
        f"  Rerank 增量贡献:         Hit@1 {_delta(hybrid.hit_at_1, reranked.hit_at_1)}  "
        f"MRR {_delta(hybrid.mrr, reranked.mrr, pct=False)}  "
        f"均排名 {hybrid.avg_best_rank:.2f} -> {reranked.avg_best_rank:.2f}"
    )
    if base.hard_count:
        print(
            f"  Rerank 在 hard 子集:     Hit@3 "
            f"{_delta(hybrid.hard_hit_at_3, reranked.hard_hit_at_3)}  "
            f"MRR {_delta(hybrid.hard_mrr, reranked.hard_mrr, pct=False)}"
        )
    for r in results:
        if r.misses:
            print(f"\n{r.label} 未命中: {', '.join(r.misses)}")


def _delta(before: float, after: float, *, pct: bool = True) -> str:
    diff = after - before
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1%}" if pct else f"{sign}{diff:.3f}"


if __name__ == "__main__":
    sys.exit(main())
