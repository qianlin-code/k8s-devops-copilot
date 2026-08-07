from app.agent.context_manager import ContextBundle
from app.agent.state_machine import AgentRunResult
from app.rag.retriever import RetrievalResult
from app.schemas.trace import (
    AgentStepTrace,
    CitationTrace,
    ContextTrace,
    ExecutionTrace,
    QueryRewriteTrace,
    RetrievalStageTrace,
    RetrievalTrace,
    RouteDecisionTrace,
    SecurityTrace,
    SufficiencyTrace,
    ToolCallTrace,
)

_CITATION_PREVIEW_CHARS = 600


def build_execution_trace(
    *,
    trace_id: str,
    total_elapsed_ms: int,
    context: ContextBundle,
    retrieval: RetrievalResult,
    agent: AgentRunResult,
    input_flags: list[str],
    output_redactions: list[str],
    agent_max_steps: int,
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        total_elapsed_ms=total_elapsed_ms,
        context=ContextTrace(
            total_turns=context.total_turns,
            windowed_turns=context.windowed_turns,
            summarized=context.summarized,
            summary_source_turns=context.summary_source_turns,
            degrade_reason=context.degrade_reason,
            summary=context.summary,
        ),
        retrieval=RetrievalTrace(
            query_rewrite=QueryRewriteTrace(
                original=retrieval.query_rewrite.original,
                rewritten=retrieval.query_rewrite.rewritten,
                applied=retrieval.query_rewrite.applied,
                keywords=retrieval.query_rewrite.keywords,
                skip_reason=retrieval.query_rewrite.skip_reason,
            ),
            hybrid_enabled=retrieval.hybrid_enabled,
            rerank_applied=retrieval.rerank_applied,
            stages=[
                RetrievalStageTrace(
                    name=s.name,
                    hit_count=s.hit_count,
                    elapsed_ms=s.elapsed_ms,
                    top_chunk_ids=s.top_chunk_ids,
                    note=s.note,
                )
                for s in retrieval.stages
            ],
            citations=[
                CitationTrace(
                    index=i,
                    chunk_id=c.chunk.chunk_id,
                    document_id=c.chunk.document_id,
                    document_title=c.chunk.document_title,
                    citation_label=c.chunk.citation_label(),
                    heading_path=c.chunk.heading_path,
                    text=c.chunk.text[:_CITATION_PREVIEW_CHARS],
                    rerank_score=round(c.rerank_score, 4),
                    rank_before=c.rank_before,
                    rank_after=c.rank_after,
                )
                for i, c in enumerate(agent.citations, start=1)
            ],
        ),
        route_decisions=[
            RouteDecisionTrace(
                round=i,
                action=d.action.value if hasattr(d.action, "value") else str(d.action),
                reasoning=d.reasoning,
                confidence=d.confidence,
                tool_name=d.tool_name,
                tool_arguments=d.tool_arguments,
            )
            for i, d in enumerate(agent.decisions, start=1)
        ],
        tool_calls=[
            ToolCallTrace(
                tool_name=inv.tool_name,
                is_write=inv.is_write,
                arguments=inv.arguments,
                result=inv.result,
                success=inv.success,
                error_code=inv.error_code,
                error_message=inv.error_message,
                cache_hit=inv.cache_hit,
                idempotent_replay=inv.idempotent_replay,
                elapsed_ms=inv.elapsed_ms,
            )
            for inv in agent.invocations
        ],
        sufficiency=(
            SufficiencyTrace(
                sufficient=agent.sufficiency.sufficient,
                reasoning=agent.sufficiency.reasoning,
                missing_information=agent.sufficiency.missing_information,
                suggested_next_step=agent.sufficiency.suggested_next_step,
            )
            if agent.sufficiency is not None
            else None
        ),
        steps=[
            AgentStepTrace(step=s.step, node=s.node, detail=s.detail) for s in agent.steps
        ],
        security=SecurityTrace(
            input_flags=input_flags, output_redactions=output_redactions
        ),
        agent_max_steps=agent_max_steps,
    )
