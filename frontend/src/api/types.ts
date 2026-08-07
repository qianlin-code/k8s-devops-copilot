/**
 * 本文件由 backend/scripts/generate_frontend_types.py 自动生成，请勿手动修改。
 * 后端修改任何接口 schema 后，重新运行该脚本以同步类型。
 *
 * Generated from OpenAPI schema of Enterprise Support Copilot.
 */


export interface AgentStepTrace {
  step: number
  /** route/execute_tool/execute_confirmed_write/verify_sufficiency/generate_answer/await_write_confirmation/max_steps_exceeded */
  node: string
  detail?: Record<string, unknown>
}

export interface ChatRequest {
  /** 用户的自然语言问题 */
  question: string
  /** 提问用户的账号 ID */
  user_id: string
  /** 续接已有会话；留空则新建会话 */
  conversation_id?: null | string
  /** 是否在响应里返回完整执行链路 */
  include_trace?: boolean
}

export interface ChatResponse {
  conversation_id: string
  message_id: number
  /** direct_answer / tool_assisted_answer / write_confirmation_required / insufficient_information / max_steps_exceeded */
  outcome: string
  answer: string
  pending_write?: PendingWriteActionSchema | null
  trace?: ExecutionTrace | null
  created_at: string
}

/** SSE 事件的文档化载荷。 */
export interface ChatStreamEnvelope {
  /** event: progress 的载荷 */
  progress?: ProgressEvent | null
  /** event: done 的载荷 */
  done?: ChatResponse | null
  /** event: error 的载荷 */
  error?: StreamErrorEvent | null
}

export interface CitationTrace {
  /** 回答中 [n] 引用编号 */
  index: number
  chunk_id: string
  document_id: string
  document_title: string
  citation_label: string
  heading_path?: string[]
  text: string
  rerank_score: number
  /** rerank 前在融合结果中的排名 */
  rank_before: number
  /** rerank 后排名，与 rank_before 对比可见重排效果 */
  rank_after: number
}

export interface ConfirmWriteRequest {
  conversation_id: string
  user_id: string
  /** 来自 pending_write 的令牌 */
  confirmation_token: string
  /** false 表示用户拒绝执行 */
  approved: boolean
  include_trace?: boolean
}

export interface ContextTrace {
  total_turns: number
  windowed_turns: number
  summarized: boolean
  summary_source_turns: number
  degrade_reason?: null | string
  summary?: null | string
}

export interface ConversationDetailResponse {
  conversation_id: string
  user_id: string
  title?: null | string
  summary?: null | string
  created_at: string
  updated_at: string
  messages: MessageItem[]
}

export interface ConversationListResponse {
  total: number
  conversations: ConversationSummary[]
}

export interface ConversationSummary {
  conversation_id: string
  user_id: string
  title?: null | string
  message_count: number
  has_summary: boolean
  created_at: string
  updated_at: string
}

export interface DeleteDocumentResponse {
  document_id: string
  deleted: boolean
  vector_count: number
  bm25_index_size: number
}

export interface DependencyCheck {
  name: string
  ok: boolean
  detail?: null | string
  elapsed_ms: number
}

export interface DocumentListResponse {
  collection_name: string
  total: number
  vector_count: number
  bm25_index_size: number
  documents: DocumentSummary[]
}

export interface DocumentSummary {
  document_id: string
  title: string
  /** upload / sedimentation / file */
  source: string
  source_ref?: null | string
  chunk_strategy: string
  chunk_count: number
  char_count: number
  collection_name: string
  created_at: string
}

/** 全局统一错误返回格式，永不包含原生异常堆栈。 */
export interface ErrorResponse {
  /** 业务错误码，见 ErrorCode 枚举 */
  code: string
  /** 人类可读的错误说明 */
  message: string
  /** 全链路追踪 ID，用于日志关联排查 */
  trace_id: string
  /** 调用方是否可以重试该请求 */
  retryable: boolean
  /** 附加调试上下文，生产环境可能为空 */
  details?: Record<string, unknown>
}

/** 接口响应里的完整链路，前端直接渲染，无需翻后台日志。 */
export interface ExecutionTrace {
  trace_id: string
  total_elapsed_ms: number
  context: ContextTrace
  retrieval: RetrievalTrace
  route_decisions: RouteDecisionTrace[]
  tool_calls: ToolCallTrace[]
  sufficiency?: SufficiencyTrace | null
  steps: AgentStepTrace[]
  security: SecurityTrace
  agent_max_steps: number
}

/** 存活探针。无需鉴权，所以生产环境不返回内部拓扑。 */
export interface HealthResponse {
  status?: "ok"
  environment: string
  llm_provider?: null | string
  embedding_provider?: null | string
  collection_name?: null | string
}

export interface IngestResponse {
  document: DocumentSummary
  bm25_index_size: number
}

export interface IngestTextRequest {
  title: string
  /** Markdown 或纯文本正文 */
  content: string
  /** char / markdown，留空用配置默认值 */
  chunk_strategy?: null | string
}

export interface MarkSedimentationRequest {
  conversation_id: string
  marked_by: string
  proposed_title?: null | string
}

export interface MessageItem {
  message_id: number
  /** user / assistant / system */
  role: string
  content: string
  trace_id?: null | string
  /** 该轮的执行链路快照，仅 assistant 消息有 */
  trace?: Record<string, unknown> | null
  created_at: string
}

/** 需要用户确认的写操作。前端拿 confirmation_token 回传即执行。 */
export interface PendingWriteActionSchema {
  tool_name: string
  description: string
  arguments: Record<string, unknown>
  reasoning: string
  /** 确认令牌，回传到 /chat/confirm 执行该操作 */
  confirmation_token: string
}

/** 阶段进展事件。 */
export interface ProgressEvent {
  phase: "accepted" | "guarded" | "context_built" | "retrieving" | "retrieved" | "agent_step" | "generating"
  /** 面向用户的中文阶段描述 */
  label: string
  /** 自请求开始的累计耗时 */
  elapsed_ms: number
  /** 该阶段的补充信息，键随 phase 变化 */
  detail?: Record<string, unknown>
}

export interface QueryRewriteTrace {
  original: string
  rewritten: string
  applied: boolean
  keywords?: string[]
  skip_reason?: null | string
}

export interface ReadinessResponse {
  ready: boolean
  checks: DependencyCheck[]
}

export interface RetrievalStageTrace {
  /** 阶段名：query_rewrite/vector_search/bm25_search/rrf_fusion/rerank/relevance_filter */
  name: string
  hit_count: number
  elapsed_ms: number
  top_chunk_ids?: string[]
  note?: null | string
}

export interface RetrievalTrace {
  query_rewrite: QueryRewriteTrace
  hybrid_enabled: boolean
  rerank_applied: boolean
  stages: RetrievalStageTrace[]
  citations: CitationTrace[]
}

export interface ReviewSedimentationRequest {
  reviewer: string
  approved: boolean
  title_override?: null | string
  note?: null | string
}

export interface RouteDecisionTrace {
  round: number
  /** answer / call_tool / insufficient */
  action: string
  reasoning: string
  confidence: number
  tool_name?: null | string
  tool_arguments?: Record<string, unknown>
}

export interface SecurityTrace {
  input_flags?: string[]
  output_redactions?: string[]
}

export interface SedimentationEntry {
  pending_id: string
  conversation_id: string
  question: string
  answer: string
  proposed_title: string
  marked_by: string
  /** pending / approved / rejected */
  status: string
  review_note?: null | string
  kb_document_id?: null | string
  created_at: string
  reviewed_at?: null | string
  /** 审核人；自动初筛通过时为 system:auto-quality */
  reviewed_by?: null | string
  /** 是否由自动质量初筛通过，未经人工审核 */
  auto_approved?: boolean
  /** 云端小模型质量初筛分数 */
  quality_score?: null | number
  quality_reasoning?: null | string
  /** 非空表示疑似与已有知识库文档重复 */
  duplicate_of_document_id?: null | string
  duplicate_score?: null | number
}

export interface SedimentationListResponse {
  total: number
  entries: SedimentationEntry[]
}

/** SSE 的 error 事件载荷。 */
export interface StreamErrorEvent {
  /** 业务错误码，见 ErrorCode 枚举 */
  code: string
  /** 人类可读的错误说明 */
  message: string
  /** 全链路追踪 ID，用于日志关联排查 */
  trace_id: string
  /** 调用方是否可以重试该请求 */
  retryable: boolean
  /** 附加调试上下文，生产环境可能为空 */
  details?: Record<string, unknown>
  /** 非流式接口下该错误对应的 HTTP 状态码 */
  http_status: number
}

export interface SufficiencyTrace {
  sufficient: boolean
  reasoning: string
  missing_information?: string[]
  suggested_next_step?: null | string
}

export interface ToolAuditItem {
  audit_id: number
  trace_id: string
  conversation_id?: null | string
  request_id?: null | string
  tool_name: string
  is_write: boolean
  success: boolean
  cache_hit: boolean
  idempotent_replay: boolean
  error_code?: null | string
  elapsed_ms: number
  created_at: string
}

export interface ToolAuditListResponse {
  total: number
  items: ToolAuditItem[]
}

export interface ToolCallTrace {
  tool_name: string
  is_write: boolean
  arguments: Record<string, unknown>
  success: boolean
  result?: Record<string, unknown> | null
  error_code?: null | string
  error_message?: null | string
  cache_hit: boolean
  idempotent_replay: boolean
  elapsed_ms: number
}

export const API_ENDPOINTS = {
  chat_api_v1_chat_post: { method: 'POST', path: '/api/v1/chat' },
  confirm_write_api_v1_chat_confirm_post: { method: 'POST', path: '/api/v1/chat/confirm' },
  chat_stream_api_v1_chat_stream_post: { method: 'POST', path: '/api/v1/chat/stream' },
  list_conversations_api_v1_conversations_get: { method: 'GET', path: '/api/v1/conversations' },
  get_conversation_api_v1_conversations: { method: 'GET', path: '/api/v1/conversations/{conversation_id}' },
  health_api_v1_health_get: { method: 'GET', path: '/api/v1/health' },
  ingest_document_api_v1_knowledge_documents_post: { method: 'POST', path: '/api/v1/knowledge/documents' },
  list_documents_api_v1_knowledge_documents_get: { method: 'GET', path: '/api/v1/knowledge/documents' },
  delete_document_api_v1_knowledge_documents: { method: 'DELETE', path: '/api/v1/knowledge/documents/{document_id}' },
  mark_sedimentation_api_v1_knowledge_sedimentations_post: { method: 'POST', path: '/api/v1/knowledge/sedimentations' },
  list_sedimentations_api_v1_knowledge_sedimentations_get: { method: 'GET', path: '/api/v1/knowledge/sedimentations' },
  review_sedimentation_api_v1_knowledge_sedimentations: { method: 'POST', path: '/api/v1/knowledge/sedimentations/{pending_id}/review' },
  readiness_api_v1_readiness_get: { method: 'GET', path: '/api/v1/readiness' },
  list_tool_audits_api_v1_tool_audits_get: { method: 'GET', path: '/api/v1/tool-audits' },
} as const

export const API_KEY_HEADER = 'X-API-Key'
export const TRACE_ID_HEADER = 'X-Trace-Id'
