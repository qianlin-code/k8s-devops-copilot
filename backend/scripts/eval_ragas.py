"""端到端生成质量评估：RAGAS 风格指标的自研实现。

不装 ragas 库：ragas 默认假设 OpenAI 接口且会拉入 langchain 依赖，
与本项目 llm/client.py 的双 provider 封装（Ollama/百炼）整合成本高，
自己按论文方法论实现这几个指标反而更贴合现有架构。

指标:
  context_precision / context_recall  纯代码计算，基于 eval_set.json 的 expected_keywords
  tool_correctness                    纯代码计算，对比实际调用的工具与 expected_tool
  faithfulness / answer_relevancy     需要 LLM 裁判，合并为一次结构化调用以减半费用

裁判固定用 QWEN_JUDGE_MODEL（默认 qwen-max），不受 --mode 影响：
本地 7B 模型评自己生成的答案噪声太大，裁判必须独立于被测链路。

运行前需要:
  - Ollama 本地服务在跑（--mode local / both 需要）
  - QWEN_API_KEY 已配置（裁判调用是硬性依赖；--mode cloud / both 还需要它做生成）

费用提示: 裁判调用 = 案例数 x 模式数，每次调用是一次 chat completion。
默认 --mode local 只产生「裁判」调用（不产生「生成」调用的云端费用）。

运行:
  python scripts/eval_ragas.py                    # --mode local，跑全部 30 条
  python scripts/eval_ragas.py --mode cloud        # 云端生成 + 云端裁判，会产生计费调用
  python scripts/eval_ragas.py --mode both         # 本地/云端各跑一遍，对比 cost/quality
  python scripts/eval_ragas.py --limit 5           # 只跑前 N 条，调试用
  python scripts/eval_ragas.py --save-json out.json  # 保存逐条明细，供 bad case 分析
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT / "data" / "eval_set.json"
DOCS_DIR = ROOT / "data" / "docs"
EVAL_USER_ID = "eval-user-no-account"

CANDIDATE_K = 20
TOP_N = 5


@dataclass(slots=True)
class CaseResult:
    case_id: str
    query: str
    difficulty: str
    gold_answer: str
    expected_outcome: str
    expected_tool: str | None
    actual_outcome: str
    actual_tool: str | None
    answer: str
    retrieved_texts: list[str] = field(default_factory=list)
    context_precision: float = 0.0
    context_recall: float = 0.0
    tool_correct: bool = True
    faithfulness: float = 0.0
    relevancy: float = 0.0
    judge_reasoning: str = ""


@dataclass(slots=True)
class AggregateResult:
    mode: str
    count: int
    context_precision: float
    context_recall: float
    tool_correctness: float
    faithfulness: float
    answer_relevancy: float


class JudgeVerdict(BaseModel):
    """裁判对单条问答的联合评分,合并 faithfulness 与 answer_relevancy 以减半调用量。"""

    faithfulness: float = Field(
        ge=0.0, le=1.0,
        description="答案中的事实性陈述有多大比例能在检索片段中找到依据,1.0=完全忠实无编造",
    )
    answer_relevancy: float = Field(
        ge=0.0, le=1.0,
        description="答案是否切题地回应了用户问题(参考标准答案判断信息量是否足够),1.0=完全切题",
    )
    reasoning: str = Field(description="给出以上两个分数的简短理由,指出具体哪句话缺乏依据或跑题")


_JUDGE_SYSTEM = """你是 RAG 系统的质量裁判,只做客观评分,不生成新答案。

faithfulness(忠实度): 检查候选答案里的每个事实性陈述,是否能在给定的检索片段中找到依据。
凡是检索片段没有提到、但答案里出现的具体结论,都算编造(hallucination),拉低分数。

answer_relevancy(相关性): 参考标准答案,判断候选答案是否切题地回应了用户问题,
是否覆盖了标准答案里的关键信息点,而不是泛泛而谈或答非所问。

两个分数独立打,不要因为答案「读起来通顺」就给高分。"""


def _bootstrap_env(mode: str) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="ragas-"))
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
            "AGENT_MAX_STEPS": "4",
            "TOOL_CACHE_TTL_SECONDS": "0",
        }
    )
    if mode == "cloud":
        os.environ.update({"LLM_PROVIDER": "qwen", "EMBEDDING_PROVIDER": "ollama"})
    else:
        os.environ.update({"LLM_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "ollama"})
    return workdir


def _context_metrics(
    retrieved_texts: list[str], expected_keywords: list[str]
) -> tuple[float, float]:
    """context_precision/recall 用关键词覆盖率近似计算,不需要 LLM。

    precision: 召回片段中「命中期望关键词」的比例(候选里有多少是真相关)。
    recall: 期望关键词中「至少被某个片段命中」的比例(该找的信息有没有被召回)。
    这是简化版——严格 RAGAS 用 LLM 判每个片段是否相关，这里用关键词命中近似，
    与 scripts/eval_retrieval.py 的判定口径保持一致，可直接对比。
    """
    if not retrieved_texts:
        return 0.0, 0.0
    lowered = [t.lower() for t in retrieved_texts]
    hits_per_chunk = [
        any(k.lower() in t for k in expected_keywords) for t in lowered
    ]
    precision = sum(hits_per_chunk) / len(hits_per_chunk)
    covered = sum(
        1 for k in expected_keywords if any(k.lower() in t for t in lowered)
    )
    recall = covered / len(expected_keywords) if expected_keywords else 0.0
    return precision, recall


def _run_case(case: dict, chat_service, session, judge_client) -> CaseResult:
    from app.agent.tools.base import ToolContext

    trace_id = f"eval-{case['id']}"
    retrieval = chat_service._retriever.retrieve(case["query"])  # noqa: SLF001
    result = chat_service._agent.run(  # noqa: SLF001
        case["query"],
        retrieval.chunks,
        ToolContext(
            session=session,
            trace_id=trace_id,
            user_id=EVAL_USER_ID,
            conversation_id=None,
        ),
    )
    retrieved_texts = [c.chunk.contextual_text for c in retrieval.chunks]
    precision, recall = _context_metrics(retrieved_texts, case["expected_keywords"])

    actual_tool = next(
        (inv.tool_name for inv in result.invocations if inv.success), None
    )
    tool_correct = actual_tool == case.get("expected_tool")

    verdict = _judge(
        judge_client,
        query=case["query"],
        gold_answer=case["gold_answer"],
        candidate_answer=result.answer,
        retrieved_texts=retrieved_texts,
    )

    return CaseResult(
        case_id=case["id"],
        query=case["query"],
        difficulty=case.get("difficulty", "unknown"),
        gold_answer=case["gold_answer"],
        expected_outcome=case["expected_outcome"],
        expected_tool=case.get("expected_tool"),
        actual_outcome=result.outcome.value,
        actual_tool=actual_tool,
        answer=result.answer,
        retrieved_texts=retrieved_texts,
        context_precision=precision,
        context_recall=recall,
        tool_correct=tool_correct,
        faithfulness=verdict.faithfulness,
        relevancy=verdict.answer_relevancy,
        judge_reasoning=verdict.reasoning,
    )


def _judge(
    judge_client,
    *,
    query: str,
    gold_answer: str,
    candidate_answer: str,
    retrieved_texts: list[str],
) -> JudgeVerdict:
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved_texts)) or "(无检索片段)"
    payload = (
        f"[用户问题]\n{query}\n\n"
        f"[检索片段]\n{context}\n\n"
        f"[标准答案(参考,不要求逐字匹配)]\n{gold_answer}\n\n"
        f"[候选答案(待评分)]\n{candidate_answer}"
    )
    return judge_client.structured(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": payload},
        ],
        JudgeVerdict,
    )


def _run_mode(mode: str, cases: list[dict]) -> tuple[AggregateResult, list[CaseResult]]:
    from app.agent.answerer import Answerer
    from app.agent.context_manager import ConversationContextManager
    from app.agent.router import Router
    from app.agent.state_machine import AgentStateMachine
    from app.agent.sufficiency import SufficiencyChecker
    from app.agent.tools.cache import ToolResultCache
    from app.agent.tools.executor import ToolExecutor
    from app.agent.tools.registry import get_tool_registry
    from app.config import get_settings
    from app.knowledge.ingest import KnowledgeIngestor
    from app.llm.factory import get_embedding_client, get_judge_client, get_llm_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.reranker import get_reranker
    from app.rag.retriever import Retriever
    from app.rag.vector_store import get_vector_store
    from app.services.chat_service import ChatService
    from app.storage.db import init_db, session_scope

    get_settings.cache_clear()
    init_db()
    store = get_vector_store()
    bm25 = get_bm25_index()
    embedding = get_embedding_client()
    llm = get_llm_client()
    judge = get_judge_client()

    ingestor = KnowledgeIngestor(vector_store=store, embedding_client=embedding, bm25_index=bm25)
    docs = sorted(DOCS_DIR.glob("*.md"))
    with session_scope() as session:
        for doc in docs:
            ingestor.ingest_text(
                session, title=doc.stem, content=doc.read_text(encoding="utf-8"),
                source="file", source_ref=doc.name,
            )

    retriever = Retriever(
        vector_store=store, embedding_client=embedding, bm25_index=bm25,
        reranker=get_reranker(), llm_client=llm,
    )
    registry = get_tool_registry()
    agent = AgentStateMachine(
        router=Router(llm),
        checker=SufficiencyChecker(llm),
        answerer=Answerer(llm),
        executor=ToolExecutor(registry, ToolResultCache(0)),
        registry=registry,
    )
    chat_service = ChatService(
        retriever=retriever, agent=agent, context_manager=ConversationContextManager(llm)
    )

    results: list[CaseResult] = []
    with session_scope() as session:
        for i, case in enumerate(cases, start=1):
            print(f"  [{mode}] {i}/{len(cases)} {case['id']}: {case['query'][:30]}...", file=sys.stderr)
            results.append(_run_case(case, chat_service, session, judge))

    n = len(results)
    agg = AggregateResult(
        mode=mode,
        count=n,
        context_precision=sum(r.context_precision for r in results) / n,
        context_recall=sum(r.context_recall for r in results) / n,
        tool_correctness=sum(1 for r in results if r.tool_correct) / n,
        faithfulness=sum(r.faithfulness for r in results) / n,
        answer_relevancy=sum(r.relevancy for r in results) / n,
    )
    return agg, results


def _print_report(aggregates: list[AggregateResult]) -> None:
    print("=" * 90)
    print("端到端生成质量评估 (RAGAS 风格自研指标, 裁判模型固定为云端强模型)")
    print("=" * 90)
    header = (
        f"{'模式':<8} {'案例数':>6} {'ctx_precision':>14} {'ctx_recall':>11} "
        f"{'tool_correct':>13} {'faithfulness':>13} {'relevancy':>10}"
    )
    print(header)
    print("-" * 90)
    for a in aggregates:
        print(
            f"{a.mode:<8} {a.count:>6} {a.context_precision:>13.1%} {a.context_recall:>10.1%} "
            f"{a.tool_correctness:>12.1%} {a.faithfulness:>12.3f} {a.answer_relevancy:>10.3f}"
        )
    print("-" * 90)
    print(
        "\n注: tool_correctness 在本数据集上恒为高分——30 条案例均为不含账号 ID 的知识性问题，"
        "按路由规则会走 direct_answer 而非 call_tool，该指标未覆盖工具路由能力，仅反映知识问答路由正确率。"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["local", "cloud", "both"], default="local",
        help="local=Ollama生成+云端裁判(默认,免费); cloud=百炼生成+云端裁判(产生云端调用); both=两者都跑",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条,调试用")
    parser.add_argument("--save-json", type=str, default=None, help="保存逐条明细到该路径")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    modes = ["local", "cloud"] if args.mode == "both" else [args.mode]
    if "cloud" in modes:
        print(
            f"!! --mode 含 cloud: 将对 {len(cases)} 条案例发起云端生成调用 "
            "(裁判调用两种模式都会发生,与 --mode 无关) !!",
            file=sys.stderr,
        )

    workdir = _bootstrap_env(modes[0])
    try:
        all_aggregates: list[AggregateResult] = []
        all_details: dict[str, list[CaseResult]] = {}
        for mode in modes:
            if mode != modes[0]:
                # 切换 provider 需要重新 bootstrap 环境变量并清空客户端缓存
                from app.llm.factory import reset_clients

                workdir2 = _bootstrap_env(mode)
                reset_clients()
                shutil.rmtree(workdir, ignore_errors=True)
                workdir = workdir2
            agg, details = _run_mode(mode, cases)
            all_aggregates.append(agg)
            all_details[mode] = details

        _print_report(all_aggregates)

        if args.save_json:
            payload = {
                mode: [
                    {
                        "case_id": r.case_id,
                        "query": r.query,
                        "difficulty": r.difficulty,
                        "expected_outcome": r.expected_outcome,
                        "actual_outcome": r.actual_outcome,
                        "expected_tool": r.expected_tool,
                        "actual_tool": r.actual_tool,
                        "gold_answer": r.gold_answer,
                        "answer": r.answer,
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                        "tool_correct": r.tool_correct,
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.relevancy,
                        "judge_reasoning": r.judge_reasoning,
                    }
                    for r in details
                ]
                for mode, details in all_details.items()
            }
            Path(args.save_json).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n逐条明细已保存到 {args.save_json}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
