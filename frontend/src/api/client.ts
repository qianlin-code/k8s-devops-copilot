import {
  API_KEY_HEADER,
  TRACE_ID_HEADER,
  type ChatRequest,
  type ChatResponse,
  type ConfirmWriteRequest,
  type ConversationDetailResponse,
  type ConversationListResponse,
  type DeleteDocumentResponse,
  type DocumentListResponse,
  type ErrorResponse,
  type HealthResponse,
  type IngestResponse,
  type IngestTextRequest,
  type MarkSedimentationRequest,
  // 与 DOM 内置的 ProgressEvent 同名，必须显式导入，否则会静默用成 DOM 类型
  type ProgressEvent,
  type StreamErrorEvent,
  type ReadinessResponse,
  type ReviewSedimentationRequest,
  type SedimentationEntry,
  type SedimentationListResponse,
  type ToolAuditListResponse,
} from './types'

const BASE = '/api/v1'
const API_KEY_STORAGE = 'copilot.apiKey'

/** 后端统一错误格式的载体，UI 可直接展示 code 与 traceId 便于排查。 */
export class ApiError extends Error {
  readonly code: string
  readonly traceId: string
  readonly retryable: boolean
  readonly details: Record<string, unknown>
  readonly httpStatus: number

  constructor(httpStatus: number, payload: ErrorResponse) {
    super(payload.message)
    this.name = 'ApiError'
    this.httpStatus = httpStatus
    this.code = payload.code
    this.traceId = payload.trace_id
    this.retryable = payload.retryable
    this.details = payload.details ?? {}
  }
}

export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE) ?? ''
}

/**
 * 保存 API Key。空值走清除而非存空串 —— 否则后续请求会带一个空 header，
 * 服务端返回 401 但用户以为"已经填过了"。
 */
export function setApiKey(value: string): void {
  const trimmed = value.trim()
  if (trimmed) localStorage.setItem(API_KEY_STORAGE, trimmed)
  else localStorage.removeItem(API_KEY_STORAGE)
}

export function hasApiKey(): boolean {
  return getApiKey().length > 0
}

/** 普通 CRUD 请求的默认超时。 */
const DEFAULT_TIMEOUT_MS = 15_000
/** Agent 链路超时：本地 7B 模型串联多次 LLM 调用，实测 20-40s。 */
const AGENT_TIMEOUT_MS = 120_000
/**
 * SSE 空闲超时：多久收不到任何数据就判定后端挂死。
 * 服务端每 10s 发心跳，所以 45s 静默足以说明异常，
 * 同时不会误杀「慢但正常」的长链路。
 */
const STREAM_IDLE_TIMEOUT_MS = 45_000

/** 请求超时或被主动取消时抛出，便于 UI 区别于业务错误。 */
export class RequestAbortedError extends Error {
  readonly timedOut: boolean
  constructor(timedOut: boolean) {
    super(timedOut ? '请求超时' : '请求已取消')
    this.name = 'RequestAbortedError'
    this.timedOut = timedOut
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<{ data: T; traceId: string | null }> {
  const headers = new Headers(init.headers)
  headers.set(API_KEY_HEADER, getApiKey())
  if (init.body !== undefined) {
    headers.set('Content-Type', 'application/json')
  }

  // 调用方传了 signal 就尊重它，否则用超时控制器兜底，
  // 避免后端卡住时请求永远挂着。
  const controller = new AbortController()
  const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null
  let timedOut = false
  if (timer) {
    controller.signal.addEventListener('abort', () => {
      timedOut = init.signal?.aborted !== true
    })
  }
  init.signal?.addEventListener('abort', () => controller.abort())

  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, { ...init, headers, signal: controller.signal })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new RequestAbortedError(timedOut)
    }
    throw err
  } finally {
    if (timer) clearTimeout(timer)
  }

  const traceId = response.headers.get(TRACE_ID_HEADER)
  const text = await response.text()
  const body: unknown = text ? JSON.parse(text) : null

  if (!response.ok) {
    throw new ApiError(response.status, body as ErrorResponse)
  }
  return { data: body as T, traceId }
}

function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>) {
  const query = params
    ? Object.entries(params)
        .filter(([, v]) => v !== undefined && v !== '')
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
    : ''
  return request<T>(query ? `${path}?${query}` : path)
}

function post<T>(path: string, body: unknown) {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

export interface StreamHandlers {
  onProgress?: (event: ProgressEvent) => void
  signal?: AbortSignal
}

/**
 * SSE 版对话。EventSource 只支持 GET，所以用 fetch + ReadableStream 手动解析。
 *
 * 相比非流式接口：本地 7B 模型一轮要 20-40s（串联 4 次 LLM 调用），
 * 干等会被误判为卡死。流式下每个阶段即时回调，用户能看到进展。
 */
async function chatStream(
  payload: ChatRequest,
  { onProgress, signal }: StreamHandlers = {},
): Promise<ChatResponse> {
  const headers = new Headers({
    [API_KEY_HEADER]: getApiKey(),
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  })

  // 用「空闲超时」而非总时长上限：只要还在收数据（含服务端每 10s 的心跳）
  // 就说明后端活着，不该掐断一个慢但正常的请求。真挂死时才会触发。
  const controller = new AbortController()
  let timedOut = false
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, STREAM_IDLE_TIMEOUT_MS)
  }
  signal?.addEventListener('abort', () => controller.abort())

  const abortedError = () => new RequestAbortedError(timedOut)

  let response: Response
  resetIdleTimer()
  try {
    response = await fetch(`${BASE}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
  } catch (err) {
    if (idleTimer) clearTimeout(idleTimer)
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw abortedError()
    }
    throw err
  }

  // 鉴权失败、注入拦截等在流开始前就返回普通 JSON 错误
  if (!response.ok) {
    const text = await response.text()
    throw new ApiError(response.status, JSON.parse(text) as ErrorResponse)
  }
  if (!response.body) {
    throw new Error('当前环境不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: ChatResponse | null = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      resetIdleTimer() // 收到任何数据（含心跳）就说明后端活着
      buffer += decoder.decode(value, { stream: true })

      // SSE 以空行分隔事件；最后一段可能不完整，留在 buffer 里等下一块
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        const parsed = parseSseBlock(block)
        if (!parsed) continue
        if (parsed.event === 'progress') {
          onProgress?.(parsed.data as ProgressEvent)
        } else if (parsed.event === 'done') {
          result = parsed.data as ChatResponse
        } else if (parsed.event === 'error') {
          // 用后端透出的 http_status，别一律当 500：
          // 注入拦截是 422，硬编码 500 会误报成服务器故障且 retryable 判断失真
          const err = parsed.data as StreamErrorEvent
          throw new ApiError(err.http_status, err)
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw abortedError()
    }
    throw err
  } finally {
    if (idleTimer) clearTimeout(idleTimer)
    // await 而非 fire-and-forget：确保底层连接确实释放，
    // 否则异常路径下连接可能悬挂到 GC。
    await reader.cancel().catch(() => undefined)
  }

  if (result === null) {
    throw new Error('流已结束但未收到最终结果')
  }
  return result
}

function parseSseBlock(block: string): { event: string; data: unknown } | null {
  let event = ''
  const dataLines: string[] = []
  for (const line of block.split('\n')) {
    // 以 ':' 开头的是注释帧（后端用它发心跳保活），按 SSE 规范忽略
    if (line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!event || dataLines.length === 0) return null
  return { event, data: JSON.parse(dataLines.join('\n')) }
}

export const api = {
  // health 探活给短超时：后端忙时应快速判定离线，而不是把页面吊死在"连接中"
  health: () => request<HealthResponse>('/health', {}, 4_000),
  readiness: () => request<ReadinessResponse>('/readiness', {}, 8_000),

  // Agent 链路可能跑 20-40s（本地 7B 串联多次 LLM 调用），给足超时
  chat: (payload: ChatRequest) =>
    request<ChatResponse>(
      '/chat',
      { method: 'POST', body: JSON.stringify(payload) },
      AGENT_TIMEOUT_MS,
    ),
  chatStream,
  confirmWrite: (payload: ConfirmWriteRequest) =>
    request<ChatResponse>(
      '/chat/confirm',
      { method: 'POST', body: JSON.stringify(payload) },
      AGENT_TIMEOUT_MS,
    ),

  // user_id 必填：后端按它做用户隔离，不传会 422
  listConversations: (params: { user_id: string; limit?: number; offset?: number }) =>
    get<ConversationListResponse>('/conversations', params),
  getConversation: (id: string, userId: string, includeTrace = true) =>
    get<ConversationDetailResponse>(`/conversations/${encodeURIComponent(id)}`, {
      user_id: userId,
      include_trace: includeTrace,
    }),
  listToolAudits: (params: {
    user_id: string
    conversation_id?: string
    tool_name?: string
    limit?: number
    offset?: number
  }) => get<ToolAuditListResponse>('/tool-audits', params),

  ingestDocument: (payload: IngestTextRequest) =>
    post<IngestResponse>('/knowledge/documents', payload),
  listDocuments: () => get<DocumentListResponse>('/knowledge/documents'),
  deleteDocument: (id: string) =>
    request<DeleteDocumentResponse>(`/knowledge/documents/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  markSedimentation: (payload: MarkSedimentationRequest) =>
    post<SedimentationEntry>('/knowledge/sedimentations', payload),
  listSedimentations: (status?: string) =>
    get<SedimentationListResponse>('/knowledge/sedimentations', { status }),
  reviewSedimentation: (pendingId: string, payload: ReviewSedimentationRequest) =>
    post<SedimentationEntry>(
      `/knowledge/sedimentations/${encodeURIComponent(pendingId)}/review`,
      payload,
    ),
}
