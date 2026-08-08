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
  python scripts/eval_ragas.py --case-ids q04,q14,q27 --mode both  # 只对照指定案例
  python scripts/eval_ragas.py --save-json out.json  # 保存逐条明细，供 bad case 分析

云端调用前会打印本次运行累计的 prompt/completion token 数与一个粗略费用估算
（单价是脚本里硬编码的估计值，不代表 DashScope 当前计费，只用于避免误触大额调用；
真实计费以百炼控制台为准）。
"""

import argparse
import json
import math
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


# 粗略估算，不是 DashScope 真实计费单价——网络环境下无法直接核实官方页面报价，
# 这里用搜索到的公开报价区间取一个保守偏高的估计，只用于避免误触大额调用的量级提示。
# 真实计费以百炼控制台账单为准。
_ROUGH_PRICE_PER_1K_TOKENS_CNY = {
    "qwen-plus": 0.004,
    "qwen-max": 0.02,
}


@dataclass(slots=True)
class AggregateResult:
    mode: str
    count: int
    context_precision: float
    context_recall: float
    tool_correctness: float
    # knowledge_routing_accuracy: 知识性问题（expected_tool=None）里，没有误触发工具的比例。
    # tool_routing_accuracy: 需要工具的问题（expected_tool 非空）里，触发了正确工具的比例。
    # 拆开看才能区分「模型该答不答乱调工具」和「该调工具时调对了没」两种不同的失败模式。
    knowledge_routing_accuracy: float
    tool_routing_accuracy: float | None
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
    # 工具路由案例需要代入真实账号视角才能合理触发工具；纯知识性问题保持
    # 无账号视角(EVAL_USER_ID)，避免路由因为看到账号 ID 就联想到查账号状态。
    user_id = case.get("user_id") or EVAL_USER_ID
    retrieval = chat_service._retriever.retrieve(case["query"])  # noqa: SLF001
    result = chat_service._agent.run(  # noqa: SLF001
        case["query"],
        retrieval.chunks,
        ToolContext(
            session=session,
            trace_id=trace_id,
            user_id=user_id,
            conversation_id=None,
        ),
    )
    retrieved_texts = [c.chunk.contextual_text for c in retrieval.chunks]
    precision, recall = _context_metrics(retrieved_texts, case["expected_keywords"])

    # 写工具在真正执行前会先中断等待确认，此时不出现在 invocations 里，
    # 而是在 pending_write——只看 invocations 会把「正确路由到写工具但等确认」
    # 误判成「没调用任何工具」。
    if result.pending_write is not None:
        actual_tool = result.pending_write.tool_name
    else:
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


@dataclass(slots=True)
class UsageSummary:
    mode: str
    generation_model: str
    generation_prompt_tokens: int
    generation_completion_tokens: int
    judge_model: str
    judge_prompt_tokens: int
    judge_completion_tokens: int

    def estimated_cost_cny(self) -> float | None:
        """粗略估算，价格是脚本里硬编码的保守估计，不是 DashScope 真实计费单价。"""
        total = 0.0
        found_any = False
        for model, prompt_tok, completion_tok in (
            (self.generation_model, self.generation_prompt_tokens, self.generation_completion_tokens),
            (self.judge_model, self.judge_prompt_tokens, self.judge_completion_tokens),
        ):
            price = _ROUGH_PRICE_PER_1K_TOKENS_CNY.get(model)
            if price is None:
                continue
            found_any = True
            total += (prompt_tok + completion_tok) / 1000 * price
        return total if found_any else None


def _run_mode(
    mode: str, cases: list[dict]
) -> tuple[AggregateResult, list[CaseResult], UsageSummary]:
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
    from app.storage.seed import seed_mock_data

    get_settings.cache_clear()
    init_db()
    store = get_vector_store()
    bm25 = get_bm25_index()
    embedding = get_embedding_client()
    llm = get_llm_client()
    judge = get_judge_client()

    with session_scope() as session:
        seed_mock_data(session)  # 工具路由案例需要真实存在的 mock 账号/订单/工单

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
    knowledge_cases = [r for r in results if r.expected_tool is None]
    tool_cases = [r for r in results if r.expected_tool is not None]
    agg = AggregateResult(
        mode=mode,
        count=n,
        context_precision=sum(r.context_precision for r in results) / n,
        context_recall=sum(r.context_recall for r in results) / n,
        tool_correctness=sum(1 for r in results if r.tool_correct) / n,
        knowledge_routing_accuracy=(
            sum(1 for r in knowledge_cases if r.tool_correct) / len(knowledge_cases)
            if knowledge_cases
            else float("nan")
        ),
        tool_routing_accuracy=(
            sum(1 for r in tool_cases if r.tool_correct) / len(tool_cases)
            if tool_cases
            else None
        ),
        faithfulness=sum(r.faithfulness for r in results) / n,
        answer_relevancy=sum(r.relevancy for r in results) / n,
    )
    usage = UsageSummary(
        mode=mode,
        generation_model=llm.model,
        generation_prompt_tokens=llm.total_prompt_tokens,
        generation_completion_tokens=llm.total_completion_tokens,
        judge_model=judge.model,
        judge_prompt_tokens=judge.total_prompt_tokens,
        judge_completion_tokens=judge.total_completion_tokens,
    )
    return agg, results, usage


def _print_report(aggregates: list[AggregateResult]) -> None:
    print("=" * 100)
    print("端到端生成质量评估 (RAGAS 风格自研指标, 裁判模型固定为云端强模型)")
    print("=" * 100)
    header = (
        f"{'模式':<8} {'案例数':>6} {'ctx_prec':>9} {'ctx_recall':>11} "
        f"{'knowledge_routing':>18} {'tool_routing':>13} {'faithfulness':>13} {'relevancy':>10}"
    )
    print(header)
    print("-" * 100)
    for a in aggregates:
        tool_str = f"{a.tool_routing_accuracy:.1%}" if a.tool_routing_accuracy is not None else "n/a"
        # knowledge_cases 为空时是 float("nan")，{:.1%} 会打印成 "nan%"——
        # 用同样的 "n/a" 展示，而不是让读者误以为是个真实算出来的百分比
        knowledge_str = (
            "n/a"
            if math.isnan(a.knowledge_routing_accuracy)
            else f"{a.knowledge_routing_accuracy:.1%}"
        )
        print(
            f"{a.mode:<8} {a.count:>6} {a.context_precision:>8.1%} {a.context_recall:>10.1%} "
            f"{knowledge_str:>17} {tool_str:>13} "
            f"{a.faithfulness:>12.3f} {a.answer_relevancy:>10.3f}"
        )
    print("-" * 100)
    print(
        "\nknowledge_routing_accuracy: 知识性问题(expected_tool=None)里没有误触发工具的比例，"
        "反映模型是否 over-action。"
        "\ntool_routing_accuracy: 需要工具的问题(expected_tool 非空)里触发了正确工具的比例，"
        "反映模型是否 conservative 该调不调。n/a 表示该模式下数据集没有此类案例。"
    )


def _print_usage(all_usage: list[UsageSummary]) -> None:
    print("\ntoken 用量与粗略费用估算（单价非官方实时报价，仅供量级参考，请以百炼控制台账单为准）:")
    for u in all_usage:
        cost = u.estimated_cost_cny()
        cost_str = f"约 ¥{cost:.3f}" if cost is not None else "未知模型单价,无法估算"
        print(
            f"  [{u.mode}] 生成({u.generation_model}): "
            f"prompt={u.generation_prompt_tokens} completion={u.generation_completion_tokens} | "
            f"裁判({u.judge_model}): prompt={u.judge_prompt_tokens} completion={u.judge_completion_tokens} "
            f"| 估算费用 {cost_str}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["local", "cloud", "both"], default="local",
        help="local=Ollama生成+云端裁判(默认,免费); cloud=百炼生成+云端裁判(产生云端调用); both=两者都跑",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条,调试用")
    parser.add_argument(
        "--case-ids", type=str, default=None,
        help="逗号分隔的案例 id 列表,只跑这些案例(如 q04,q14,q27),用于定向复现/对照实验",
    )
    parser.add_argument("--save-json", type=str, default=None, help="保存逐条明细到该路径")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    if args.case_ids:
        wanted = {c.strip() for c in args.case_ids.split(",") if c.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"警告: 案例 id 不存在，已忽略: {sorted(missing)}", file=sys.stderr)
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
        all_usage: list[UsageSummary] = []
        for mode in modes:
            if mode != modes[0]:
                # 切换 provider 需要重新 bootstrap 环境变量并清空客户端缓存
                from app.llm.factory import reset_clients

                workdir2 = _bootstrap_env(mode)
                reset_clients()
                shutil.rmtree(workdir, ignore_errors=True)
                workdir = workdir2
            agg, details, usage = _run_mode(mode, cases)
            all_aggregates.append(agg)
            all_details[mode] = details
            all_usage.append(usage)

        _print_report(all_aggregates)
        _print_usage(all_usage)

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
