import { useEffect, useRef } from 'react'

import type { ExecutionTrace, PendingWriteActionSchema } from '@/api/types'
import type { ChatProgress } from '@/hooks/useChat'

import { TraceViewer } from './TraceViewer'
import './MessageList.css'

export interface ChatTurn {
  id: string
  role: 'user' | 'assistant'
  content: string
  outcome?: string
  trace?: ExecutionTrace | null
  pendingWrite?: PendingWriteActionSchema | null
  failed?: boolean
}

const OUTCOME_LABELS: Record<string, { text: string; cls: string }> = {
  direct_answer: { text: '知识问答', cls: 'ok' },
  tool_assisted_answer: { text: '工具辅助', cls: 'ok' },
  write_confirmation_required: { text: '待确认写操作', cls: 'write' },
  write_rejected: { text: '已取消', cls: '' },
  insufficient_information: { text: '信息不足', cls: 'warn' },
  max_steps_exceeded: { text: '超出步数上限', cls: 'warn' },
}

const SAMPLE_QUESTIONS = [
  'ops-demo 命名空间下 api-gateway-7f9c 这个 Pod 一直是 Pending 状态',
  '查一下 ops-demo 下 worker-queue 这个 Deployment 的副本状态',
  '配置已经修好了，请帮我重启一下 ops-demo 下的 worker-queue',
]

interface Props {
  turns: ChatTurn[]
  busy: boolean
  progress?: ChatProgress | null
  onConfirm: (token: string, approved: boolean) => void
  confirmingToken: string | null
  onPickSample?: (question: string) => void
}

export function MessageList({
  turns,
  busy,
  progress,
  onConfirm,
  confirmingToken,
  onPickSample,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns.length, busy, progress?.label])

  if (turns.length === 0 && !busy) {
    return (
      <div className="messages empty">
        <div className="hint">
          <h3>K8s 智能运维 Copilot</h3>
          <p>先在知识库页面上传文档，然后在下方提问。每次回答都会附带完整执行链路。</p>
          <ul className="samples">
            {SAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button type="button" onClick={() => onPickSample?.(q)}>
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    )
  }

  return (
    <div className="messages">
      {turns.map((turn) => (
        <article key={turn.id} className={`msg msg-${turn.role}`}>
          <div className="msg-role">{turn.role === 'user' ? '我' : 'Copilot'}</div>
          <div className="msg-body">
            {turn.outcome && (
              <span className={`tag ${OUTCOME_LABELS[turn.outcome]?.cls ?? ''}`}>
                {OUTCOME_LABELS[turn.outcome]?.text ?? turn.outcome}
              </span>
            )}
            <div className={turn.failed ? 'msg-text failed' : 'msg-text'}>{turn.content}</div>

            {turn.pendingWrite && (
              <div className="confirm-card">
                <header>
                  <span className="tag write">写操作需要确认</span>
                  <code>{turn.pendingWrite.tool_name}</code>
                </header>
                <p className="confirm-why">{turn.pendingWrite.reasoning}</p>
                <pre>{JSON.stringify(turn.pendingWrite.arguments, null, 2)}</pre>
                <div className="confirm-actions">
                  <button
                    className="primary"
                    disabled={confirmingToken !== null}
                    onClick={() => onConfirm(turn.pendingWrite!.confirmation_token, true)}
                  >
                    确认执行
                  </button>
                  <button
                    className="danger"
                    disabled={confirmingToken !== null}
                    onClick={() => onConfirm(turn.pendingWrite!.confirmation_token, false)}
                  >
                    取消
                  </button>
                </div>
              </div>
            )}

            {turn.trace && <TraceViewer trace={turn.trace} />}
          </div>
        </article>
      ))}

      {busy && (
        <article className="msg msg-assistant">
          <div className="msg-role">Copilot</div>
          <div className="msg-body">
            <div className="thinking">
              <span className="thinking-now">
                {progress?.label ?? '正在处理'}
                <span className="dots">
                  <i />
                  <i />
                  <i />
                </span>
              </span>
              {progress && (
                <span className="thinking-elapsed">
                  {(progress.elapsedMs / 1000).toFixed(1)}s
                </span>
              )}
            </div>
            {/* 本地模型首次真实调用会有冷启动开销（实测可达 100s+），
                超过阈值才提示，避免正常响应时也显示这行造成误导 */}
            {progress && progress.elapsedMs > 15_000 && (
              <p className="thinking-warmup-hint">
                模型可能正在首次加载中，请再等一下，后续对话会明显更快
              </p>
            )}
            {/* 已走过的阶段：让长等待有可见的推进感，也便于判断卡在哪一步 */}
            {progress && progress.history.length > 1 && (
              <ol className="thinking-steps">
                {progress.history.slice(0, -1).map((label, i) => (
                  <li key={`${label}-${i}`}>{label}</li>
                ))}
              </ol>
            )}
          </div>
        </article>
      )}
      <div ref={endRef} />
    </div>
  )
}
