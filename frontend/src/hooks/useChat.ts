import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, RequestAbortedError, api } from '@/api/client'
import type {
  ChatResponse,
  ExecutionTrace,
  MessageItem,
  PendingWriteActionSchema,
  ProgressEvent,
} from '@/api/types'
import type { ChatTurn } from '@/components/chat/MessageList'

// 用 UUID 而非模块级自增：后者在 HMR 重载后会从 0 重新计数，
// 与既有 turns 的 key 撞车导致 React 复用错误的节点。
const nextId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `t${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

/** 单轮对话的实时进展，用于在等待期间告诉用户走到哪一步了。 */
export interface ChatProgress {
  label: string
  phase: ProgressEvent['phase']
  elapsedMs: number
  /** 已走过的阶段标签，按顺序去重 */
  history: string[]
}

export function useChat() {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [confirmingToken, setConfirmingToken] = useState<string | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [progress, setProgress] = useState<ChatProgress | null>(null)

  // busy 用 ref 镜像：send 的闭包捕获的是旧值，
  // 连点两次时第二次仍看到 busy=false，会产生重复会话。
  const inFlight = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  // 回合序号：响应回来时校验是否仍属于当前回合。
  // 否则用户在等待期间点「新建会话」，旧响应会追加到新会话里。
  const turnSeq = useRef(0)

  useEffect(() => {
    // 组件卸载时取消进行中的请求，避免向已卸载组件 setState
    return () => abortRef.current?.abort()
  }, [])

  const appendResponse = useCallback((body: ChatResponse, seq: number) => {
    if (seq !== turnSeq.current) return // 已被 reset 或新请求取代，丢弃
    setConversationId(body.conversation_id)
    setTurns((prev) => [
      ...prev,
      {
        id: nextId(),
        role: 'assistant',
        content: body.answer,
        outcome: body.outcome,
        trace: body.trace ?? null,
        pendingWrite: body.pending_write ?? null,
      },
    ])
  }, [])

  const appendFailure = useCallback(
    (err: unknown, seq: number, kind: 'ask' | 'write' = 'ask') => {
      if (seq !== turnSeq.current) return
      if (err instanceof RequestAbortedError && !err.timedOut) {
        // 用户主动取消：不当作失败留痕
        return
      }
      const apiError = err instanceof ApiError ? err : null
      if (apiError) setError(apiError)
      setTurns((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          failed: true,
          content: apiError
            ? `请求失败：${apiError.message}（${apiError.code}，trace ${apiError.traceId.slice(0, 12)}）`
            : err instanceof RequestAbortedError
              ? // 写操作超时要说清两件事：一是后端很可能已经执行完（实测
                // confirm 到过 186s，前端先超时了），二是再点一次确认是安全的
                // —— 同一张卡片的 request_id 不变，会命中审计表的幂等重放，
                // 不会把一次重启做成两次。含糊的提示会让用户不敢动也不知道结果。
                kind === 'write'
                ? '确认请求超时，写操作可能已在后端执行完成。可再点一次确认（同一操作幂等，不会重复执行），或到历史记录核对结果。'
                : '请求超时，后端可能仍在处理。可稍后在历史记录中查看结果。'
              : `请求失败：${String(err)}`,
        },
      ])
    },
    [],
  )

  const send = useCallback(
    async (question: string) => {
      const text = question.trim()
      if (!text || inFlight.current) return
      inFlight.current = true

      const seq = ++turnSeq.current
      const controller = new AbortController()
      abortRef.current = controller

      setError(null)
      setTurns((prev) => [...prev, { id: nextId(), role: 'user', content: text }])
      setBusy(true)
      setProgress({ label: '已提交，等待后端响应', phase: 'accepted', elapsedMs: 0, history: [] })

      try {
        const data = await api.chatStream(
          {
            question: text,
            conversation_id: conversationId ?? undefined,
            include_trace: true,
          },
          {
            signal: controller.signal,
            onProgress: (event) => {
              if (seq !== turnSeq.current) return
              setProgress((prev) => ({
                label: event.label,
                phase: event.phase,
                elapsedMs: event.elapsed_ms,
                history:
                  prev && prev.history[prev.history.length - 1] === event.label
                    ? prev.history
                    : [...(prev?.history ?? []), event.label],
              }))
            },
          },
        )
        appendResponse(data, seq)
      } catch (err) {
        appendFailure(err, seq)
      } finally {
        // 同 confirm：已被新回合取代时不要覆盖它的 inFlight/busy/progress。
        // reset/hydrate/新 send 都会自行重置 inFlight，跳过是安全的。
        if (seq === turnSeq.current) {
          inFlight.current = false
          abortRef.current = null
          setBusy(false)
          setProgress(null)
        }
      }
    },
    [conversationId, appendResponse, appendFailure],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  /**
   * 从历史记录恢复一个旧会话，之后 send/confirm 会自然带着这个 conversation_id 续聊。
   * 只有「最后一条 assistant 消息」的 pending_write 会恢复成可点击的确认卡片——
   * 更早的 pending_write 早已被后续消息取代（确认/拒绝后是新增消息而非覆盖旧消息），
   * 重新展示会让用户对着一个其实已经处理完的操作再次点确认。
   */
  const hydrate = useCallback((id: string, messages: MessageItem[]) => {
    turnSeq.current += 1
    abortRef.current?.abort()
    inFlight.current = false
    const lastAssistantIdx = messages.reduce(
      (acc, m, i) => (m.role === 'assistant' ? i : acc),
      -1,
    )
    setTurns(
      messages.map((m, i) => ({
        id: nextId(),
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
        trace: (m.trace as unknown as ExecutionTrace) ?? null,
        pendingWrite:
          i === lastAssistantIdx
            ? ((m.trace as { pending_write?: PendingWriteActionSchema } | null)
                ?.pending_write ?? null)
            : null,
      })),
    )
    setConversationId(id)
    setError(null)
    setProgress(null)
    setBusy(false)
    setConfirmingToken(null)
  }, [])

  const confirm = useCallback(
    async (token: string, approved: boolean) => {
      if (!conversationId || confirmingToken || inFlight.current) return
      inFlight.current = true
      const seq = ++turnSeq.current
      // confirmingToken 已让按钮进入 disabled，足以防重复提交。
      // 卡片留到成功后再清：请求失败或中断时用户还能重试，
      // 提前清掉会让 token 无处可用，只能重开会话。
      setConfirmingToken(token)
      setBusy(true)
      try {
        const { data } = await api.confirmWrite({
          conversation_id: conversationId,
          confirmation_token: token,
          approved,
          include_trace: true,
        })
        if (seq === turnSeq.current) {
          setTurns((prev) =>
            prev.map((t) =>
              t.pendingWrite?.confirmation_token === token
                ? { ...t, pendingWrite: null }
                : t,
            ),
          )
        }
        appendResponse(data, seq)
      } catch (err) {
        appendFailure(err, seq, 'write')
      } finally {
        // 全部收尾都要校验 seq：本请求发出后用户若点了「新建会话」或发了新消息，
        // turnSeq 已递增，新回合可能已经设置了自己的 confirmingToken/inFlight。
        // 无条件清理会把新回合的状态擦掉，确认按钮被错误启用、可重复点击。
        // reset/hydrate/新 send 都会自行把 inFlight 置回 false，这里跳过是安全的。
        if (seq === turnSeq.current) {
          inFlight.current = false
          setConfirmingToken(null)
          setBusy(false)
        }
      }
    },
    [conversationId, confirmingToken, appendResponse, appendFailure],
  )

  const reset = useCallback(() => {
    // 递增序号让进行中的回合的响应被丢弃，而不是落进新会话
    turnSeq.current += 1
    abortRef.current?.abort()
    inFlight.current = false
    setTurns([])
    setConversationId(null)
    setError(null)
    setProgress(null)
    setBusy(false)
    setConfirmingToken(null)
  }, [])

  return {
    turns,
    conversationId,
    busy,
    progress,
    confirmingToken,
    error,
    send,
    cancel,
    confirm,
    reset,
    hydrate,
  }
}
