"""端到端 RAG / Agent 评测，支持无付费机器门禁与可选云端裁判。

`--no-judge` 是发布候选的第一道质量门：只使用本地 Ollama、Embedding 与
Reranker，严格核对 outcome、工具和引用结构，不初始化裁判客户端、不读取
QWEN_API_KEY，也不输出费用。默认模式保留原有的本地生成 + 独立 Qwen 裁判。
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT / "data" / "eval_set.json"
DOCS_DIR = ROOT / "data" / "docs_k8s"
EVAL_USER_ID = "eval-ops-no-resource"
MANUAL_REVIEW_CASE_IDS = tuple(f"q{i:02d}" for i in range(1, 21))
CANDIDATE_K = 20
TOP_N = 5
_ROUGH_PRICE_PER_1K_TOKENS_CNY = {"qwen-plus": 0.004, "qwen-max": 0.02}
_GENERATION_PRICES_PER_1K_CNY = {
    "qwen-plus": {
        "prompt_cache_miss": 0.004,
        "prompt_cache_hit": 0.004,
        "completion": 0.004,
    },
    "deepseek-v4-pro": {
        "prompt_cache_miss": 0.003,
        "prompt_cache_hit": 0.000025,
        "completion": 0.006,
    },
}
_GENERATION_STRUCTURED_MAX_TOKENS = 512
_JUDGE_PROMPT_VERSION = "ragas-judge-v2-verified-evidence"


@dataclass(slots=True)
class CitationAudit:
    required: bool
    count: int
    parseable: bool
    keyword_supported: bool
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrictCaseVerdict:
    passed: bool
    outcome_matches: bool
    tool_matches: bool
    write_not_executed: bool
    answer_generation_verified: bool
    answer_evidence_valid: bool
    citation: CitationAudit
    failures: list[str] = field(default_factory=list)


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
    retrieval_citation_records: list[dict[str, Any]] = field(default_factory=list)
    citation_records: list[dict[str, Any]] = field(default_factory=list)
    verified_evidence_records: list[dict[str, Any]] = field(default_factory=list)
    judge_context_records: list[dict[str, Any]] = field(default_factory=list)
    answer_generation: dict[str, Any] | None = None
    retrieval_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    retrieval_layers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    context_precision: float = 0.0
    context_recall: float = 0.0
    strict: StrictCaseVerdict | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    judge_reasoning: str | None = None


@dataclass(slots=True)
class AggregateResult:
    mode: str
    count: int
    context_precision: float
    context_recall: float
    knowledge_routing_accuracy: float | None
    readonly_tool_routing_accuracy: float | None
    write_confirmation_routing_accuracy: float | None
    citation_coverage: float | None
    citation_keyword_support: float | None
    strict_passed: int
    strict_failed_case_ids: list[str]
    faithfulness: float | None
    answer_relevancy: float | None


class JudgeVerdict(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _bootstrap_env(mode: str) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="ragas-"))
    os.environ.update(
        {
            "JWT_SECRET_KEY": "evaluation-jwt-secret-not-for-production",
            "STARTUP_PROBE_EXTERNAL": "false",
            "DATABASE_URL": f"sqlite:///{(workdir / 'eval.db').as_posix()}",
            "QDRANT_PATH": str(workdir / "qdrant"),
            "LLM_PROVIDER": (
                "qwen" if mode == "cloud" else "deepseek" if mode == "deepseek" else "ollama"
            ),
            "EMBEDDING_PROVIDER": "ollama",
            "ENABLE_QUERY_REWRITE": "false",
            "CHUNK_STRATEGY": "markdown",
            "RETRIEVE_TOP_K": str(CANDIDATE_K),
            "RERANK_TOP_N": str(TOP_N),
            "AGENT_MAX_STEPS": "4",
            "TOOL_CACHE_TTL_SECONDS": "0",
        }
    )
    return workdir


def _context_metrics(texts: list[str], keywords: list[str]) -> tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    lowered = [text.lower() for text in texts]
    hits = [any(keyword.lower() in text for keyword in keywords) for text in lowered]
    precision = sum(hits) / len(hits)
    covered = sum(1 for keyword in keywords if any(keyword.lower() in text for text in lowered))
    return precision, covered / len(keywords) if keywords else 0.0


def _citation_record(reranked: Any) -> dict[str, Any]:
    chunk = reranked.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "heading_path": list(chunk.heading_path),
        "citation_label": chunk.citation_label(),
        "text": chunk.text,
        "contextual_text": chunk.contextual_text,
        "rerank_score": round(reranked.rerank_score, 4),
        "rank_before": reranked.rank_before,
        "rank_after": reranked.rank_after,
    }


def audit_citations(
    *, case: dict[str, Any], citations: list[dict[str, Any]], allowed_titles: set[str]
) -> CitationAudit:
    required = case.get("expected_tool") is None
    failures: list[str] = []
    if required and not citations:
        failures.append("missing_citations")
    parseable = True
    for citation in citations:
        if not all(citation.get(key) for key in ("chunk_id", "document_id", "document_title")):
            parseable = False
            failures.append("citation_missing_identity")
        if citation.get("document_title") not in allowed_titles:
            parseable = False
            failures.append("citation_unknown_document")
        if not citation.get("heading_path"):
            parseable = False
            failures.append("citation_missing_heading_path")
        if not citation.get("rendered_text") or not citation.get("exact_quote"):
            parseable = False
            failures.append("citation_missing_text")
    keywords = [str(keyword).lower() for keyword in case.get("expected_keywords", [])]
    keyword_supported = bool(citations) and any(
        any(keyword in str(citation.get("rendered_text", "")).lower() for keyword in keywords)
        for citation in citations
    )
    if required and not keyword_supported:
        failures.append("citation_missing_expected_keyword_support")
    return CitationAudit(
        required=required,
        count=len(citations),
        parseable=parseable,
        keyword_supported=keyword_supported,
        failures=sorted(set(failures)),
    )


def evaluate_strict_case(
    *, case: dict[str, Any], result: CaseResult, allowed_titles: set[str], write_executed: bool
) -> StrictCaseVerdict:
    citation = audit_citations(case=case, citations=result.citation_records, allowed_titles=allowed_titles)
    expected_outcome = case["expected_outcome"]
    outcome_matches = result.actual_outcome == expected_outcome
    tool_matches = result.actual_tool == case.get("expected_tool")
    write_not_executed = not write_executed
    generated_answer_expected = expected_outcome in {"direct_answer", "tool_assisted_answer"}
    answer_generation_verified = (
        not generated_answer_expected
        or bool(result.answer_generation)
        and result.answer_generation.get("status") == "verified"
    )
    tool_evidence = [
        item
        for item in result.verified_evidence_records
        if item.get("evidence_kind") == "tool"
    ]
    answer_evidence_valid = (
        bool(result.citation_records)
        if expected_outcome == "direct_answer"
        else any(item.get("tool_name") == case.get("expected_tool") for item in tool_evidence)
        if expected_outcome == "tool_assisted_answer"
        else True
    )
    failures: list[str] = []
    if not outcome_matches:
        failures.append(f"outcome_expected_{expected_outcome}_got_{result.actual_outcome}")
    if not tool_matches:
        failures.append(f"tool_expected_{case.get('expected_tool')}_got_{result.actual_tool}")
    if case.get("expected_write_executed", False) is False and not write_not_executed:
        failures.append("write_tool_executed_during_evaluation")
    if not answer_generation_verified:
        failures.append("answer_generation_not_verified")
    if not answer_evidence_valid:
        failures.append(
            "missing_verified_tool_evidence"
            if expected_outcome == "tool_assisted_answer"
            else "missing_verified_knowledge_evidence"
        )
    failures.extend(citation.failures)
    return StrictCaseVerdict(
        passed=not failures,
        outcome_matches=outcome_matches,
        tool_matches=tool_matches,
        write_not_executed=write_not_executed,
        answer_generation_verified=answer_generation_verified,
        answer_evidence_valid=answer_evidence_valid,
        citation=citation,
        failures=sorted(set(failures)),
    )


def _verified_evidence_records(agent_result: Any) -> list[dict[str, Any]]:
    verified = agent_result.verified_answer
    if verified is None:
        return []
    chunks = {
        item.chunk.chunk_id: item.chunk
        for item in agent_result.citations
    }
    records: list[dict[str, Any]] = []
    for evidence in verified.evidence:
        chunk = chunks.get(evidence.chunk_id) if evidence.chunk_id else None
        records.append(
            {
                "item_index": evidence.item_index,
                "section": evidence.section,
                "evidence_kind": evidence.evidence_kind,
                "source_id": evidence.source_id,
                "rendered_text": evidence.rendered_text,
                "chunk_id": evidence.chunk_id,
                "document_id": chunk.document_id if chunk is not None else None,
                "document_title": chunk.document_title if chunk is not None else None,
                "heading_path": list(chunk.heading_path) if chunk is not None else [],
                "citation_label": evidence.citation_label,
                "exact_quote": evidence.exact_quote,
                "tool_name": evidence.tool_name,
                "invocation_index": evidence.invocation_index,
                "json_pointer": evidence.json_pointer,
                "serialized_value": evidence.serialized_value,
            }
        )
    return records


def _judge_context_records(
    result: CaseResult,
    *,
    pending_write: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for evidence in result.verified_evidence_records:
        if evidence.get("evidence_kind") == "knowledge":
            records.append(
                {
                    "kind": "knowledge",
                    "source_id": evidence.get("source_id"),
                    "citation_label": evidence.get("citation_label"),
                    "text": evidence.get("exact_quote")
                    or evidence.get("rendered_text"),
                }
            )
        elif evidence.get("evidence_kind") == "tool":
            records.append(
                {
                    "kind": "tool",
                    "source_id": evidence.get("source_id"),
                    "tool_name": evidence.get("tool_name"),
                    "json_pointer": evidence.get("json_pointer"),
                    "text": evidence.get("serialized_value")
                    or evidence.get("rendered_text"),
                }
            )
    if pending_write is not None:
        arguments = {
            str(key): value
            for key, value in sorted(
                dict(pending_write.get("arguments") or {}).items()
            )
            if key not in {"request_id", "confirmation_token"}
        }
        records.append(
            {
                "kind": "pending_write",
                "tool_name": pending_write.get("tool_name"),
                "description": pending_write.get("description"),
                "arguments": arguments,
            }
        )
    return records


def _render_judge_context(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(无最终验证证据)"
    rendered: list[str] = []
    for index, record in enumerate(records, start=1):
        kind = record.get("kind")
        if kind == "knowledge":
            rendered.append(
                f"[{index}] knowledge {record.get('source_id')} "
                f"{record.get('citation_label')}\n{record.get('text')}"
            )
        elif kind == "tool":
            rendered.append(
                f"[{index}] tool {record.get('source_id')} "
                f"{record.get('tool_name')} result{record.get('json_pointer') or '/'}\n"
                f"{record.get('text')}"
            )
        elif kind == "pending_write":
            rendered.append(
                f"[{index}] pending_write {record.get('tool_name')} "
                f"{record.get('description')}\n"
                f"arguments={json.dumps(record.get('arguments') or {}, ensure_ascii=False, sort_keys=True)}"
            )
    return "\n\n".join(rendered) or "(无最终验证证据)"


def _actual_tool(result: Any) -> str | None:
    if result.pending_write is not None:
        return result.pending_write.tool_name
    invoked = next((inv.tool_name for inv in result.invocations if inv.success), None)
    if invoked is not None:
        return invoked
    # 缺字段写操作会在执行前追问，不能从 pending_write 或审计中看到工具名；
    # route trace 仍然是该受控请求的可审计选择依据。
    return next((decision.tool_name for decision in reversed(result.decisions) if decision.tool_name), None)


def _retrieval_diagnostics(retrieval: Any) -> list[dict[str, Any]]:
    kept_ids = {chunk.chunk.chunk_id for chunk in retrieval.chunks}
    threshold_stage = next(
        (stage for stage in retrieval.stages if stage.name == "relevance_filter"), None
    )
    threshold = None
    if threshold_stage and threshold_stage.note:
        marker = "threshold="
        if marker in threshold_stage.note:
            threshold = float(threshold_stage.note.split(marker, 1)[1].split()[0])
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "document_id": item.chunk.document_id,
            "document_title": item.chunk.document_title,
            "heading_path": list(item.chunk.heading_path),
            "rerank_score": round(item.rerank_score, 4),
            "rank_before": item.rank_before,
            "rank_after": item.rank_after,
            "kept": item.chunk.chunk_id in kept_ids,
            "filter_reason": (
                "kept"
                if item.chunk.chunk_id in kept_ids
                else "associated_context_not_scored"
                if item.rank_after == 999
                else "below_rerank_threshold"
                if threshold is not None and item.rerank_score < threshold
                else "not_in_final_context"
            ),
        }
        for item in retrieval.pre_filter_chunks
    ]


def _layer_records(items: list[Any], keywords: list[str], *, layer: str) -> list[dict[str, Any]]:
    """将每一层的候选统一序列化，保留排名和关键词命中用于故障归因。"""
    normalized_keywords = [str(keyword).lower() for keyword in keywords]
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        if layer == "rrf":
            chunk = item.chunk
            score = item.rrf_score
            sources = item.sources
            rank_before = rank
            rank_after = rank
        elif layer == "rerank":
            chunk = item.chunk
            score = item.rerank_score
            sources = []
            rank_before = item.rank_before
            rank_after = item.rank_after
        else:
            chunk = item
            score = item.score
            sources = []
            rank_before = rank
            rank_after = rank
        text = chunk.contextual_text.lower()
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_title": chunk.document_title,
                "heading_path": list(chunk.heading_path),
                "rank": rank,
                "rank_before": rank_before,
                "rank_after": rank_after,
                "score": round(float(score), 4),
                "keyword_hit": any(keyword in text for keyword in normalized_keywords),
                "sources": sources,
            }
        )
    return records


def _retrieval_layers(retrieval: Any, keywords: list[str]) -> dict[str, list[dict[str, Any]]]:
    """输出向量、BM25、RRF、rerank 和阈值后的候选，不改变回答上下文。"""
    return {
        "vector": _layer_records(retrieval.vector_hits, keywords, layer="vector"),
        "bm25": _layer_records(retrieval.bm25_hits, keywords, layer="bm25"),
        "rrf": _layer_records(retrieval.fused_chunks, keywords, layer="rrf"),
        "rerank": _layer_records(retrieval.reranked_chunks, keywords, layer="rerank"),
        "post_threshold": _layer_records(retrieval.chunks, keywords, layer="rerank"),
    }


def _run_case(case: dict[str, Any], chat_service: Any, session: Any, judge_client: Any | None, allowed_titles: set[str]) -> CaseResult:
    from app.agent.tools.base import ToolContext
    from app.config import get_settings
    from app.rag.query_policy import min_rerank_score_for_query

    trace_id = f"eval-{case['id']}"
    settings = get_settings()
    retrieval = chat_service._retriever.retrieve(  # noqa: SLF001
        case["query"],
        min_score=min_rerank_score_for_query(
            case["query"],
            production_score=settings.min_rerank_score,
            knowledge_score=settings.knowledge_min_rerank_score,
        ),
    )
    agent_result = chat_service._agent.run(  # noqa: SLF001
        case["query"], retrieval.chunks,
        ToolContext(session=session, trace_id=trace_id, user_id=case.get("user_id") or EVAL_USER_ID, conversation_id=None),
    )
    retrieval_citations = [_citation_record(chunk) for chunk in agent_result.citations]
    verified_evidence = _verified_evidence_records(agent_result)
    citations = [
        item for item in verified_evidence if item["evidence_kind"] == "knowledge"
    ]
    answer_generation = (
        {
            "status": agent_result.verified_answer.status,
            "attempts": agent_result.verified_answer.attempts,
            "fallback_reason": agent_result.verified_answer.fallback_reason,
        }
        if agent_result.verified_answer is not None
        else None
    )
    precision, recall = _context_metrics([chunk.chunk.contextual_text for chunk in retrieval.chunks], case["expected_keywords"])
    result = CaseResult(
        case_id=case["id"], query=case["query"], difficulty=case.get("difficulty", "unknown"),
        gold_answer=case["gold_answer"], expected_outcome=case["expected_outcome"],
        expected_tool=case.get("expected_tool"), actual_outcome=agent_result.outcome.value,
        actual_tool=_actual_tool(agent_result), answer=agent_result.answer,
        retrieved_texts=[chunk.chunk.contextual_text for chunk in retrieval.chunks],
        retrieval_citation_records=retrieval_citations,
        citation_records=citations,
        verified_evidence_records=verified_evidence,
        answer_generation=answer_generation,
        retrieval_diagnostics=_retrieval_diagnostics(retrieval),
        retrieval_layers=_retrieval_layers(retrieval, case["expected_keywords"]),
        context_precision=precision, context_recall=recall,
    )
    pending_write = (
        {
            "tool_name": agent_result.pending_write.tool_name,
            "description": agent_result.pending_write.description,
            "arguments": agent_result.pending_write.arguments,
        }
        if agent_result.pending_write is not None
        else None
    )
    result.judge_context_records = _judge_context_records(
        result, pending_write=pending_write
    )
    result.strict = evaluate_strict_case(
        case=case, result=result, allowed_titles=allowed_titles,
        write_executed=any(inv.is_write and inv.success for inv in agent_result.invocations),
    )
    if judge_client is not None:
        verdict = _judge(
            judge_client,
            query=case["query"],
            gold_answer=case["gold_answer"],
            candidate_answer=agent_result.answer,
            context_records=result.judge_context_records,
        )
        result.faithfulness = verdict.faithfulness
        result.answer_relevancy = verdict.answer_relevancy
        result.judge_reasoning = verdict.reasoning
    return result


def _judge(
    judge_client: Any,
    *,
    query: str,
    gold_answer: str,
    candidate_answer: str,
    context_records: list[dict[str, Any]],
) -> JudgeVerdict:
    context = _render_judge_context(context_records)
    prompt = (
        f"[用户问题]\n{query}\n\n[最终验证证据与受控流程]\n{context}\n\n"
        f"[标准答案(参考,不要求逐字匹配)]\n{gold_answer}\n\n[候选答案(待评分)]\n{candidate_answer}"
    )
    return judge_client.structured(
        [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        JudgeVerdict,
        max_repairs=0,
        max_tokens=_GENERATION_STRUCTURED_MAX_TOKENS,
    )


_JUDGE_SYSTEM = """你是 RAG 系统的质量裁判，只做客观评分，不生成新答案。

faithfulness(忠实度): 检查候选答案里的每个事实性陈述是否能在最终验证证据或受控流程中找到依据。
answer_relevancy(相关性): 参考标准答案，判断候选答案是否切题并覆盖关键信息。
两个分数独立打分，不要因为答案通顺就给高分。"""


def _aggregate(mode: str, results: list[CaseResult], judge_enabled: bool) -> AggregateResult:
    knowledge = [result for result in results if result.expected_tool is None]
    readonly = [result for result in results if result.expected_outcome == "tool_assisted_answer"]
    writes = [result for result in results if result.expected_outcome == "write_confirmation_required"]
    strict_failed = [result.case_id for result in results if result.strict is not None and not result.strict.passed]
    def rate(items: list[CaseResult], predicate: Any) -> float | None:
        return sum(bool(predicate(item)) for item in items) / len(items) if items else None

    return AggregateResult(
        mode=mode, count=len(results),
        context_precision=sum(result.context_precision for result in results) / len(results),
        context_recall=sum(result.context_recall for result in results) / len(results),
        knowledge_routing_accuracy=rate(knowledge, lambda result: result.strict and result.strict.outcome_matches and result.strict.tool_matches),
        readonly_tool_routing_accuracy=rate(readonly, lambda result: result.strict and result.strict.outcome_matches and result.strict.tool_matches),
        write_confirmation_routing_accuracy=rate(writes, lambda result: result.strict and result.strict.outcome_matches and result.strict.tool_matches and result.strict.write_not_executed),
        citation_coverage=rate(knowledge, lambda result: result.strict and result.strict.citation.parseable and result.strict.citation.count),
        citation_keyword_support=rate(knowledge, lambda result: result.strict and result.strict.citation.keyword_supported),
        strict_passed=len(results) - len(strict_failed), strict_failed_case_ids=strict_failed,
        faithfulness=(sum(result.faithfulness or 0 for result in results) / len(results)) if judge_enabled else None,
        answer_relevancy=(sum(result.answer_relevancy or 0 for result in results) / len(results)) if judge_enabled else None,
    )


def _run_mode(
    mode: str,
    cases: list[dict[str, Any]],
    judge_enabled: bool = False,
    generation_cost_limit_cny: float | None = None,
) -> tuple[AggregateResult, list[CaseResult], dict[str, Any]]:
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
    from app.llm.client import LLMCostPolicy
    from app.llm.factory import get_embedding_client, get_llm_client
    from app.rag.bm25_index import get_bm25_index
    from app.rag.reranker import get_reranker
    from app.rag.retriever import Retriever
    from app.rag.vector_store import get_vector_store
    from app.services.chat_service import ChatService
    from app.storage.db import init_db, session_scope
    from app.storage.seed import seed_mock_data

    get_settings.cache_clear()
    init_db()
    store, bm25, embedding, llm = get_vector_store(), get_bm25_index(), get_embedding_client(), get_llm_client()
    generation_prices = generation_prices_per_1k_tokens_cny(llm.model) if mode != "local" else None
    if mode != "local":
        if generation_cost_limit_cny is None:
            raise ValueError(f"{mode} generation requires an explicit cost limit")
        assert generation_prices is not None
        llm.cost_policy = LLMCostPolicy(
            max_cost_cny=generation_cost_limit_cny,
            prompt_cache_miss_price_per_1k_cny=generation_prices["prompt_cache_miss"],
            prompt_cache_hit_price_per_1k_cny=generation_prices["prompt_cache_hit"],
            completion_price_per_1k_cny=generation_prices["completion"],
            structured_max_tokens=_GENERATION_STRUCTURED_MAX_TOKENS,
        )
    judge = None
    if judge_enabled:
        from app.llm.factory import get_judge_client
        judge = get_judge_client()
    with session_scope() as session:
        seed_mock_data(session)
    ingestor = KnowledgeIngestor(vector_store=store, embedding_client=embedding, bm25_index=bm25)
    docs = sorted(DOCS_DIR.glob("*.md"))
    with session_scope() as session:
        for doc in docs:
            ingestor.ingest_text(session, title=doc.stem, content=doc.read_text(encoding="utf-8"), source="file", source_ref=doc.name)
    retriever = Retriever(vector_store=store, embedding_client=embedding, bm25_index=bm25, reranker=get_reranker(), llm_client=llm)
    registry = get_tool_registry()
    agent = AgentStateMachine(router=Router(llm), checker=SufficiencyChecker(llm), answerer=Answerer(llm), executor=ToolExecutor(registry, ToolResultCache(0)), registry=registry)
    chat_service = ChatService(retriever=retriever, agent=agent, context_manager=ConversationContextManager(llm))
    allowed_titles = {doc.stem for doc in docs}
    results: list[CaseResult] = []
    with session_scope() as session:
        for index, case in enumerate(cases, start=1):
            print(f"  [{mode}] {index}/{len(cases)} {case['id']}: {case['query'][:30]}...", file=sys.stderr)
            results.append(_run_case(case, chat_service, session, judge, allowed_titles))
    settings = get_settings()
    metadata = {
        "generation_provider": settings.llm_provider.value,
        "generation_model": llm.model, "embedding_model": embedding.model,
        "reranker": getattr(retriever._reranker, "name", "unknown"),  # noqa: SLF001
        "rerank_top_n": settings.rerank_top_n, "retrieve_top_k": settings.retrieve_top_k,
        "min_rerank_score": settings.min_rerank_score,
        "knowledge_min_rerank_score": settings.knowledge_min_rerank_score,
        "generation_usage": {
            "calls": llm.total_request_attempts,
            "successful_calls": llm.total_calls,
            "prompt_tokens": llm.total_prompt_tokens,
            "cached_prompt_tokens": llm.total_cached_prompt_tokens,
            "uncached_prompt_tokens": llm.total_uncached_prompt_tokens,
            "completion_tokens": llm.total_completion_tokens,
            "total_tokens": llm.total_prompt_tokens + llm.total_completion_tokens,
            "accounted_tokens": llm.total_accounted_tokens,
            "usage_estimated": llm.usage_estimated,
            "prices_per_1k_tokens_cny": generation_prices,
            "cost_cny": (
                round(llm.estimated_cost_cny, 6)
                if llm.estimated_cost_cny is not None
                else 0.0 if mode == "local" else None
            ),
            "budget_cny": generation_cost_limit_cny,
        },
        "judge_enabled": judge_enabled,
        "judge_model": judge.model if judge is not None else None,
        "judge_usage": ({"prompt_tokens": judge.total_prompt_tokens, "completion_tokens": judge.total_completion_tokens, "calls": judge.total_calls} if judge is not None else None),
    }
    return _aggregate(mode, results, judge_enabled), results, metadata


def _source_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(DOCS_DIR.glob("*.md"))}


def estimate_judge_cost_cny(results: list[CaseResult], judge_model: str) -> tuple[int, float | None]:
    price = _ROUGH_PRICE_PER_1K_TOKENS_CNY.get(judge_model)
    if price is None:
        return 0, None
    # 保守上界：中文字符按一个 token 计，再为结构化输出和 JSON schema 预留 1,500 tokens/条。
    prompt_chars = sum(
        len(_JUDGE_SYSTEM)
        + len(result.query)
        + len(result.gold_answer)
        + len(result.answer)
        + len(_render_judge_context(result.judge_context_records))
        for result in results
    )
    estimated_tokens = prompt_chars + 1_500 * len(results)
    return estimated_tokens, estimated_tokens / 1000 * price


def generation_prices_per_1k_tokens_cny(model: str) -> dict[str, float]:
    prices = _GENERATION_PRICES_PER_1K_CNY.get(model)
    if prices is None:
        raise ValueError(f"no generation price configured for model {model}")
    return dict(prices)


def generation_price_per_1k_tokens_cny(model: str) -> float:
    """Backward-compatible conservative shorthand used by older callers/tests."""
    prices = generation_prices_per_1k_tokens_cny(model)
    return max(prices["prompt_cache_miss"], prices["completion"])


def _case_results_from_report(payload: dict[str, Any]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for item in payload.get("cases", []):
        raw = copy.deepcopy(item)
        raw_strict = raw.get("strict")
        if isinstance(raw_strict, dict):
            raw_citation = raw_strict.get("citation", {})
            raw_strict["citation"] = CitationAudit(**raw_citation)
            raw["strict"] = StrictCaseVerdict(**raw_strict)
        for key in ("faithfulness", "answer_relevancy"):
            if raw.get(key) == "not_run":
                raw[key] = None
        results.append(CaseResult(**raw))
    return results


def _validate_generation_report(
    payload: dict[str, Any], *, expected_provider: str | None = None,
    expected_model: str | None = None,
) -> list[CaseResult]:
    failures: list[str] = []
    if payload.get("schema_version") != 3:
        failures.append("generation_schema_version_mismatch")
    if payload.get("script_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        failures.append("script_sha256_mismatch")
    if payload.get("eval_set_sha256") != hashlib.sha256(EVAL_SET.read_bytes()).hexdigest():
        failures.append("eval_set_sha256_mismatch")
    if payload.get("document_sha256") != _source_hashes():
        failures.append("document_sha256_mismatch")
    if payload.get("exit_code") != 0:
        failures.append("generation_exit_code_not_zero")
    if payload.get("judge_mode") != "not_run":
        failures.append("generation_report_already_judged")
    cases = payload.get("cases")
    if not isinstance(cases, list) or payload.get("case_count") != len(cases):
        failures.append("case_count_mismatch")
    aggregate = payload.get("aggregate", {})
    if not isinstance(cases, list) or aggregate.get("strict_passed") != len(cases):
        failures.append("generation_not_strictly_complete")
    metadata = payload.get("metadata", {})
    provider = metadata.get("generation_provider")
    model = metadata.get("generation_model")
    if provider not in {"qwen", "deepseek"}:
        failures.append("generation_provider_mismatch")
    if expected_provider is not None and provider != expected_provider:
        failures.append("generation_provider_mismatch")
    if expected_model is not None and model != expected_model:
        failures.append("generation_model_mismatch")
    try:
        generation_prices_per_1k_tokens_cny(str(model))
    except ValueError:
        failures.append("generation_model_price_missing")
    usage = metadata.get("generation_usage")
    if not isinstance(usage, dict) or usage.get("cost_cny") is None:
        failures.append("generation_usage_missing")
    current_cases = {
        item["id"]: item
        for item in json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    }
    seen_ids: set[str] = set()
    for item in cases if isinstance(cases, list) else []:
        case_id = item.get("case_id")
        current = current_cases.get(case_id)
        if case_id in seen_ids:
            failures.append("duplicate_case_id")
        seen_ids.add(case_id)
        if current is None:
            failures.append("unknown_case_id")
            continue
        expected_fields = {
            "query": current["query"],
            "gold_answer": current["gold_answer"],
            "expected_outcome": current["expected_outcome"],
            "expected_tool": current.get("expected_tool"),
        }
        if any(item.get(key) != value for key, value in expected_fields.items()):
            failures.append(f"case_contract_mismatch:{case_id}")
        if not isinstance(item.get("judge_context_records"), list):
            failures.append(f"judge_context_missing:{case_id}")
    if failures:
        raise ValueError("invalid generation report: " + ", ".join(failures))
    return _case_results_from_report(payload)


def _judge_existing_report(
    generation_report: Path,
    output_path: Path,
    *,
    max_total_cost_cny: float,
) -> int:
    raw_bytes = generation_report.read_bytes()
    payload = json.loads(raw_bytes)
    results = _validate_generation_report(payload)
    judge_model = os.environ.get("QWEN_JUDGE_MODEL", "qwen-max")
    estimated_tokens, estimated_judge_cost = estimate_judge_cost_cny(results, judge_model)
    generation_cost = float(payload["metadata"]["generation_usage"]["cost_cny"])
    if estimated_judge_cost is None:
        raise ValueError(f"no judge price configured for model {judge_model}")
    estimated_total = generation_cost + estimated_judge_cost
    preflight = {
        "generation_report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "judge_model": judge_model,
        "prompt_version": _JUDGE_PROMPT_VERSION,
        "estimated_tokens_upper_bound": estimated_tokens,
        "estimated_judge_cost_cny": round(estimated_judge_cost, 6),
        "generation_cost_cny": generation_cost,
        "estimated_total_cost_cny": round(estimated_total, 6),
        "limit_cny": max_total_cost_cny,
    }
    if max_total_cost_cny == 0:
        output = {
            **payload,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_mode": "preflight",
            "metadata": {
                **payload["metadata"],
                "judge_enabled": False,
                "judge_mode": "preflight",
                "judge_cost_preflight": preflight,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return 0
    if estimated_total > max_total_cost_cny:
        raise ValueError("estimated generation and judge cost exceeds total budget")

    from app.llm.client import LLMCostPolicy
    from app.llm.factory import get_judge_client

    judge = get_judge_client()
    judge.cost_policy = LLMCostPolicy(
        max_cost_cny=max_total_cost_cny - generation_cost,
        price_per_1k_tokens_cny=_ROUGH_PRICE_PER_1K_TOKENS_CNY[judge_model],
        structured_max_tokens=_GENERATION_STRUCTURED_MAX_TOKENS,
    )
    for result in results:
        verdict = _judge(
            judge,
            query=result.query,
            gold_answer=result.gold_answer,
            candidate_answer=result.answer,
            context_records=result.judge_context_records,
        )
        result.faithfulness = verdict.faithfulness
        result.answer_relevancy = verdict.answer_relevancy
        result.judge_reasoning = verdict.reasoning
    aggregate = _aggregate(str(payload["aggregate"]["mode"]), results, True)
    judge_cost = judge.estimated_cost_cny
    output = {
        **payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_mode": "qwen",
        "metadata": {
            **payload["metadata"],
            "judge_enabled": True,
            "judge_mode": "qwen",
            "judge_cost_preflight": preflight,
            "judge_model": judge_model,
            "judge_prompt_version": _JUDGE_PROMPT_VERSION,
            "judge_usage": {
                "calls": getattr(judge, "total_request_attempts", judge.total_calls),
                "successful_calls": judge.total_calls,
                "prompt_tokens": judge.total_prompt_tokens,
                "completion_tokens": judge.total_completion_tokens,
                "total_tokens": judge.total_prompt_tokens + judge.total_completion_tokens,
                "accounted_tokens": getattr(
                    judge,
                    "total_accounted_tokens",
                    judge.total_prompt_tokens + judge.total_completion_tokens,
                ),
                "usage_estimated": getattr(judge, "usage_estimated", False),
                "cost_cny": round(judge_cost or 0.0, 6),
            },
            "total_cost_cny": round(generation_cost + (judge_cost or 0.0), 6),
        },
        "aggregate": asdict(aggregate),
        "cases": [_detail_payload(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return 0


def _render_manual_review(results: list[CaseResult]) -> str:
    selected = [result for result in results if result.case_id in MANUAL_REVIEW_CASE_IDS]
    lines = ["# 引用人工审阅清单", "", "固定抽样：q01–q20。不要筛选成功案例；逐项填写“是/否”和简短依据。", ""]
    for result in selected:
        evidence_lines = [
            (
                f"- #{item['item_index']} {item['section']} {item['source_id']} "
                f"知识：{item['citation_label']}（{item['chunk_id']}）\n"
                f"  - 原文：{item['exact_quote']}"
                if item["evidence_kind"] == "knowledge"
                else f"- #{item['item_index']} {item['section']} {item['source_id']} "
                f"工具：{item['tool_name']} invocation={item['invocation_index']} "
                f"pointer={item['json_pointer']}\n  - 值：{item['serialized_value']}"
            )
            for item in result.verified_evidence_records
        ] or ["- （无最终证据映射）"]
        generation = result.answer_generation or {
            "status": "not_recorded", "attempts": 0, "fallback_reason": None
        }
        lines.extend([
            f"## {result.case_id}", "", f"问题：{result.query}", "", f"回答：{result.answer}", "",
            (
                "生成："
                f"status={generation['status']}, attempts={generation['attempts']}, "
                f"fallback_reason={generation['fallback_reason']}"
            ), "", "最终回答证据映射：", *evidence_lines, "",
            "- 核心结论被引用支持：待人工填写", "- 存在误引：待人工填写", "- 引用可定位：待人工填写", "- 审阅依据：待人工填写", "",
        ])
    return "\n".join(lines)


def _detail_payload(result: CaseResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["faithfulness"] = result.faithfulness if result.faithfulness is not None else "not_run"
    payload["answer_relevancy"] = result.answer_relevancy if result.answer_relevancy is not None else "not_run"
    return payload


def _write_outputs(path: Path, *, args: argparse.Namespace, aggregate: AggregateResult, results: list[CaseResult], metadata: dict[str, Any], exit_code: int) -> None:
    payload = {
        "schema_version": 3, "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": os.environ.get("COPILOT_EVAL_COMMIT", "unknown"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "eval_set_sha256": hashlib.sha256(EVAL_SET.read_bytes()).hexdigest(),
        "eval_set_version": json.loads(EVAL_SET.read_text(encoding="utf-8")).get("version", "unversioned"),
        "document_sha256": _source_hashes(), "case_count": len(results),
        "judge_mode": metadata.get("judge_mode", "not_run"), "exit_code": exit_code,
        "metadata": metadata, "aggregate": asdict(aggregate),
        "cases": [_detail_payload(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if metadata.get("judge_mode", "not_run") == "not_run":
        path.with_name(f"{path.stem}-citation-review.md").write_text(_render_manual_review(results), encoding="utf-8")


def _print_report(aggregate: AggregateResult) -> None:
    print("=" * 84)
    print(f"严格端到端评测（{aggregate.mode}；裁判={'已启用' if aggregate.faithfulness is not None else '未运行'}）")
    print("=" * 84)
    print(f"案例: {aggregate.count}; 严格通过: {aggregate.strict_passed}/{aggregate.count}")
    def rendered_rate(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    print(f"知识路由: {rendered_rate(aggregate.knowledge_routing_accuracy)}; 只读工具: {rendered_rate(aggregate.readonly_tool_routing_accuracy)}; 写确认: {rendered_rate(aggregate.write_confirmation_routing_accuracy)}")
    print(f"引用覆盖: {rendered_rate(aggregate.citation_coverage)}; 引用关键词支撑: {rendered_rate(aggregate.citation_keyword_support)}")
    if aggregate.strict_failed_case_ids:
        print("严格失败 case: " + ", ".join(aggregate.strict_failed_case_ids))
    if aggregate.faithfulness is None:
        print("Faithfulness / Answer relevancy: not_run（无付费模式）")
    else:
        print(f"Faithfulness: {aggregate.faithfulness:.3f}; Answer relevancy: {aggregate.answer_relevancy:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "cloud", "deepseek"], default="local")
    parser.add_argument("--no-judge", action="store_true", help="只运行本地严格路由/引用门禁；绝不初始化 Qwen 裁判")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-ids", type=str, default=None)
    parser.add_argument("--save-json", type=Path, required=True, help="ignored evidence 下的逐条结果 JSON")
    parser.add_argument("--max-generation-cost-cny", type=float)
    parser.add_argument("--judge-generation-report", type=Path)
    parser.add_argument("--max-total-cost-cny", type=float)
    args = parser.parse_args()
    if args.judge_generation_report is not None:
        if args.no_judge or args.case_ids or args.limit is not None:
            parser.error("report-driven judge cannot be combined with generation options")
        if args.max_total_cost_cny is None or args.max_total_cost_cny < 0:
            parser.error("--max-total-cost-cny is required and must be non-negative")
        try:
            return _judge_existing_report(
                args.judge_generation_report,
                args.save_json,
                max_total_cost_cny=args.max_total_cost_cny,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if not args.no_judge:
        parser.error("generation runs require --no-judge; judge an existing report separately")
    if args.mode in {"cloud", "deepseek"} and (
        args.max_generation_cost_cny is None or args.max_generation_cost_cny <= 0
    ):
        parser.error(f"{args.mode} generation requires a positive --max-generation-cost-cny")
    if args.mode == "local" and args.max_generation_cost_cny is not None:
        parser.error("--max-generation-cost-cny only applies to cloud generation modes")
    if args.mode in {"cloud", "deepseek"}:
        try:
            generation_prices_per_1k_tokens_cny(
                os.environ.get(
                    "QWEN_CHAT_MODEL" if args.mode == "cloud" else "DEEPSEEK_CHAT_MODEL",
                    "qwen-plus" if args.mode == "cloud" else "deepseek-v4-pro",
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
    raw_cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()} if args.case_ids else None
    cases = [case for case in raw_cases if wanted is None or case["id"] in wanted]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error("no evaluation cases selected")
    workdir = _bootstrap_env(args.mode)
    try:
        aggregate, results, metadata = _run_mode(
            args.mode,
            cases,
            generation_cost_limit_cny=args.max_generation_cost_cny,
        )
        metadata["judge_mode"] = "not_run"
        _print_report(aggregate)
        exit_code = 0 if not aggregate.strict_failed_case_ids else 2
        _write_outputs(args.save_json, args=args, aggregate=aggregate, results=results, metadata=metadata, exit_code=exit_code)
        return exit_code
    finally:
        # 真实验收保留本轮隔离工作目录，便于复核失败案例的 SQLite/Qdrant 现场。
        # 目录位于容器 /tmp，由容器生命周期统一回收。
        pass


if __name__ == "__main__":
    sys.exit(main())
