from typing import Any, Optional

from pydantic import Field

from app.schemas.base import StrictBaseModel


class QueryRewriteTrace(StrictBaseModel):
    original: str
    rewritten: str
    applied: bool
    keywords: list[str] = Field(default_factory=list)
    skip_reason: Optional[str] = None


class RetrievalStageTrace(StrictBaseModel):
    name: str = Field(description="阶段名：query_rewrite/vector_search/bm25_search/rrf_fusion/rerank/relevance_filter")
    hit_count: int
    elapsed_ms: int
    top_chunk_ids: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class CitationTrace(StrictBaseModel):
    index: int = Field(description="回答中 [n] 引用编号")
    chunk_id: str
    document_id: str
    document_title: str
    citation_label: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    rerank_score: float
    rank_before: int = Field(description="rerank 前在融合结果中的排名")
    rank_after: int = Field(description="rerank 后排名，与 rank_before 对比可见重排效果")


class RetrievalTrace(StrictBaseModel):
    query_rewrite: QueryRewriteTrace
    hybrid_enabled: bool
    rerank_applied: bool
    stages: list[RetrievalStageTrace]
    citations: list[CitationTrace]


class ContextTrace(StrictBaseModel):
    total_turns: int
    windowed_turns: int
    summarized: bool
    summary_source_turns: int
    degrade_reason: Optional[str] = None
    summary: Optional[str] = None


class RouteDecisionTrace(StrictBaseModel):
    round: int
    action: str = Field(description="answer / call_tool / insufficient")
    reasoning: str
    confidence: float
    tool_name: Optional[str] = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(StrictBaseModel):
    tool_name: str
    is_write: bool
    arguments: dict[str, Any]
    success: bool
    result: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    cache_hit: bool
    idempotent_replay: bool
    elapsed_ms: int


class SufficiencyTrace(StrictBaseModel):
    sufficient: bool
    reasoning: str
    missing_information: list[str] = Field(default_factory=list)
    suggested_next_step: Optional[str] = None


class AgentStepTrace(StrictBaseModel):
    step: int
    node: str = Field(
        description="route/execute_tool/execute_confirmed_write/verify_sufficiency/"
        "generate_answer/await_write_confirmation/max_steps_exceeded"
    )
    detail: dict[str, Any] = Field(default_factory=dict)


class SecurityTrace(StrictBaseModel):
    input_flags: list[str] = Field(default_factory=list)
    output_redactions: list[str] = Field(default_factory=list)


class AnswerEvidenceTrace(StrictBaseModel):
    item_index: int
    section: str = Field(description="conclusion / evidence_step")
    evidence_kind: str = Field(description="knowledge / tool")
    source_id: str
    rendered_text: str
    chunk_id: Optional[str] = None
    citation_label: Optional[str] = None
    exact_quote: Optional[str] = None
    tool_name: Optional[str] = None
    invocation_index: Optional[int] = None
    json_pointer: Optional[str] = None
    serialized_value: Optional[str] = None


class AnswerGenerationTrace(StrictBaseModel):
    status: str = Field(description="verified / fallback")
    attempts: int
    fallback_reason: Optional[str] = None


class ExecutionTrace(StrictBaseModel):
    """接口响应里的完整链路，前端直接渲染，无需翻后台日志。"""

    trace_id: str
    total_elapsed_ms: int
    context: ContextTrace
    retrieval: RetrievalTrace
    route_decisions: list[RouteDecisionTrace]
    tool_calls: list[ToolCallTrace]
    sufficiency: Optional[SufficiencyTrace] = None
    steps: list[AgentStepTrace]
    security: SecurityTrace
    agent_max_steps: int
    answer_evidence: list[AnswerEvidenceTrace] = Field(default_factory=list)
    answer_generation: Optional[AnswerGenerationTrace] = None
