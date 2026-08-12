import {
  AUTHORIZATION_HEADER,
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
  type LoginRequest,
  type LoginResponse,
  type MarkSedimentationRequest,
  // 与 DOM 内置的 ProgressEvent 同名，必须显式导入，否则会静默用成 DOM 类型
  type ProgressEvent,
  type RegisterRequest,
  type RegisterResponse,
  type StreamErrorEvent,
  type ReadinessResponse,
  type ReviewSedimentationRequest,
  type SedimentationEntry,
  type SedimentationListResponse,
  type ToolAuditListResponse,
} from './types'

const BASE = '/api/v1'
// JWT 切换后，前端改用 Bearer token 而非 API Key
const ACCESS_TOKEN_STORAGE = 'copilot.accessToken'
const USER_STORAGE = 'copilot.currentUser'
export const AUTH_EXPIRED_EVENT = 'copilot:unauthorized'

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

export function getAccessToken(): string {
  return localStorage.getItem(ACCESS_TOKEN_STORAGE) ?? ''
}

/**
 * 保存 JWT token。空值走清除而非存空串。
 */
export function setAccessToken(value: string): void {
  const trimmed = value.trim()
  if (trimmed) localStorage.setItem(ACCESS_TOKEN_STORAGE, trimmed)
  else localStorage.removeItem(ACCESS_TOKEN_STORAGE)
}

export function hasAccessToken(): boolean {
  return getAccessToken().length > 0
}

/** JWT 切换：保存登录响应信息（token + 用户信息） */
export function saveLoginInfo(response: LoginResponse): void {
  localStorage.setItem(ACCESS_TOKEN_STORAGE, response.access_token)
  localStorage.setItem(USER_STORAGE, JSON.stringify({
    user_id: response.user_id,
    username: response.username,
    role: response.role,
    organization_id: response.organization_id,
  }))
}

/** JWT 切换：获取当前登录用户信息 */
export function getCurrentUser(): { user_id: string; username: string; role: string; organization_id: string } | null {
  const data = localStorage.getItem(USER_STORAGE)
  if (!data) return null
  try {
    return JSON.parse(data)
  } catch {
    return null
  }
}

/** JWT 切换：退出登录 */
export function logout(): void {
  localStorage.removeItem(ACCESS_TOKEN_STORAGE)
  localStorage.removeItem(USER_STORAGE)
}

function notifyUnauthorized(status: number): void {
  if (status === 401) window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT))
}

/** 普通 CRUD 请求的默认超时。 */
const DEFAULT_TIMEOUT_MS = 15_000
/**
 * 非流式 Agent 链路（`/chat`、`/chat/confirm`）的总时长上限。
 *
 * 取 300s 与 `vite.config.ts` 的 proxy `timeout` 对齐 —— 客户端比代理更早掐断
 * 没有意义。原值 120s 偏小：一轮的最坏耗时是「检索 + max_steps(6) 轮 ×
 * (路由 + 充分性校验) + 生成回答」，本地 7B 每次调用 7-15s，实测
 * `/chat/confirm` 单次到过 **186s**（后端返回 200、写操作已真正执行），
 * 前端却在 120s 抛超时 —— 用户看到"请求超时"，以为写操作没生效。
 *
 * 注意这仍是总时长上限、只能缓解不能根治：真要彻底解决得像 `/chat/stream`
 * 那样改成 SSE + 心跳 + 空闲超时（见下方 STREAM_IDLE_TIMEOUT_MS 的理由）。
 */
const AGENT_TIMEOUT_MS = 300_000
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

/**
 * 把错误响应体解析成后端的统一错误结构；不是 JSON 时合成一个等价结构。
 *
 * 后端所有错误都是 `{code, message, trace_id, retryable, details}`，但错误页
 * 未必来自后端：nginx 的 502/504、WAF 拦截页、代理超时页都是 HTML。直接
 * `JSON.parse` 会抛 SyntaxError 且不被包成 ApiError，用户看到的是
 * 「请求失败：SyntaxError: Unexpected token <」，分不清是鉴权失败、
 * 服务离线还是网关问题。
 */
function toErrorResponse(status: number, text: string): ErrorResponse {
  try {
    const parsed = JSON.parse(text) as unknown
    // 只认真正带 code 的后端错误结构；`null`、数组、纯字符串 JSON 都要兜底
    if (parsed && typeof parsed === 'object' && 'code' in parsed) {
      return parsed as ErrorResponse
    }
  } catch {
    // 落到下面的兜底结构
  }
  return {
    code: status >= 502 && status <= 504 ? 'UPSTREAM_UNAVAILABLE' : 'NON_JSON_ERROR',
    message:
      status >= 502 && status <= 504
        ? `网关或后端不可用（HTTP ${status}），请确认后端服务与代理是否正常。`
        : `服务端返回了非预期的响应（HTTP ${status}）。`,
    trace_id: '',
    retryable: status >= 502 && status <= 504,
    details: { body_preview: text.slice(0, 200) },
  } as ErrorResponse
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<{ data: T; traceId: string | null }> {
  const headers = new Headers(init.headers)
  // 所有业务请求均携带 JWT Bearer token；服务端从 token 取得用户身份。
  const token = getAccessToken()
  if (token) {
    headers.set(AUTHORIZATION_HEADER, `Bearer ${token}`)
  }
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

  if (!response.ok) {
    notifyUnauthorized(response.status)
    throw new ApiError(response.status, toErrorResponse(response.status, text))
  }
  // 成功响应必须是合法 JSON；这里解析失败属于契约被破坏，照常抛出
  return { data: (text ? JSON.parse(text) : null) as T, traceId }
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
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  })
  // SSE 与普通请求使用相同的 JWT 身份来源。
  const token = getAccessToken()
  if (token) {
    headers.set(AUTHORIZATION_HEADER, `Bearer ${token}`)
  }

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

  // 鉴权失败、注入拦截等在流开始前就返回普通 JSON 错误。
  // 这两个提前退出的分支都要先清 idleTimer：函数返回后它仍会在 45s 后
  // 触发 controller.abort()，留一个已无意义的悬挂定时器。
  if (!response.ok) {
    if (idleTimer) clearTimeout(idleTimer)
    const text = await response.text()
    notifyUnauthorized(response.status)
    throw new ApiError(response.status, toErrorResponse(response.status, text))
  }
  if (!response.body) {
    if (idleTimer) clearTimeout(idleTimer)
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
          notifyUnauthorized(err.http_status)
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
  // JWT 鉴权：登录和注册不需要 token
  login: (payload: LoginRequest) => post<LoginResponse>('/auth/login', payload),
  register: (payload: RegisterRequest) => post<RegisterResponse>('/auth/register', payload),

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

  listConversations: (params: { limit?: number; offset?: number }) =>
    get<ConversationListResponse>('/conversations', params),
  getConversation: (id: string, includeTrace = true) =>
    get<ConversationDetailResponse>(`/conversations/${encodeURIComponent(id)}`, {
      include_trace: includeTrace,
    }),
  listToolAudits: (params: {
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
