"""诊断单条查询在 Rerank 前后的排名变化。

用于定位「Rerank 反而把正确片段挤出 Top-N」这类问题。

运行: python scripts/diagnose_case.py q01
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CANDIDATE_K = 20
TOP_N = 5


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "q01"

    workdir = Path(tempfile.mkdtemp(prefix="diag-"))
    os.environ.update(
        {
            "JWT_SECRET_KEY": "diagnostic-jwt-secret-not-for-production",
            "STARTUP_PROBE_EXTERNAL": "false",
            "DATABASE_URL": f"sqlite:///{(workdir / 'diag.db').as_posix()}",
            "QDRANT_PATH": str(workdir / "qdrant"),
            "ENABLE_QUERY_REWRITE": "false",
            "CHUNK_STRATEGY": "markdown",
            "RETRIEVE_TOP_K": str(CANDIDATE_K),
            "RERANK_TOP_N": str(TOP_N),
            "MIN_RERANK_SCORE": "0.0",
        }
    )

    try:
        return _run(case_id)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run(case_id: str) -> int:
    import json

    from app.knowledge.ingest import KnowledgeIngestor
    from app.llm.factory import get_embedding_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.reranker import IdentityReranker, get_reranker
    from app.rag.retriever import Retriever
    from app.rag.vector_store import get_vector_store
    from app.storage.db import init_db, session_scope

    cases = json.loads(
        (ROOT / "data" / "eval_set.json").read_text(encoding="utf-8")
    )["cases"]
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        print(f"case {case_id} not found")
        return 1

    init_db()
    store = get_vector_store()
    bm25 = get_bm25_index()
    embedding = get_embedding_client()
    ingestor = KnowledgeIngestor(
        vector_store=store, embedding_client=embedding, bm25_index=bm25
    )
    with session_scope() as session:
        for doc in sorted((ROOT / "data" / "docs_k8s").glob("*.md")):
            ingestor.ingest_text(
                session,
                title=doc.stem,
                content=doc.read_text(encoding="utf-8"),
                source="file",
                source_ref=doc.name,
            )

    keywords = case["expected_keywords"]
    print(f"查询: {case['query']}")
    print(f"期望关键词: {' | '.join(keywords)}")

    def is_hit(text: str) -> bool:
        low = text.lower()
        return any(k.lower() in low for k in keywords)

    def show(label: str, reranker, enable_rerank: bool) -> None:
        r = Retriever(
            vector_store=store,
            embedding_client=embedding,
            bm25_index=bm25,
            reranker=reranker,
            llm_client=None,
        )
        # 取全部候选看完整排名，而非只看最终 Top-N
        out = r.retrieve(
            case["query"],
            top_k=CANDIDATE_K,
            top_n=CANDIDATE_K,
            enable_hybrid=True,
            enable_rerank=enable_rerank,
            enable_rewrite=False,
            min_score=0.0,
        )
        print(f"\n--- {label} (共 {len(out.chunks)} 条候选) ---")
        for i, sc in enumerate(out.chunks, start=1):
            mark = " <== 期望" if is_hit(sc.chunk.text) else ""
            inside = "  " if i <= TOP_N else " x"  # x 表示会被 Top-5 截断掉
            score = getattr(sc, "rerank_score", None)
            score_txt = f"{score:.4f}" if isinstance(score, float) else "  -   "
            print(f"{inside}{i:>3} {score_txt}  {sc.chunk.text[:56]}{mark}")
        ranks = [i for i, sc in enumerate(out.chunks, start=1) if is_hit(sc.chunk.text)]
        best = min(ranks) if ranks else None
        verdict = (
            f"最佳命中排名 {best}，{'在' if best and best <= TOP_N else '不在'} Top-{TOP_N} 内"
            if best
            else "全部候选中无命中"
        )
        print(f"  => {verdict}")

    show("B. 混合检索（无 Rerank）", IdentityReranker(), False)
    show("C. 混合检索 + Rerank", get_reranker(), True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
