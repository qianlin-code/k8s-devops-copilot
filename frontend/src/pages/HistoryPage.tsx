import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiError, api } from '@/api/client'
import type {
  ConversationDetailResponse,
  ConversationListResponse,
  ExecutionTrace,
  ToolAuditListResponse,
} from '@/api/types'
import { TraceViewer } from '@/components/chat/TraceViewer'
import './HistoryPage.css'

export default function HistoryPage() {
  const navigate = useNavigate()
  const [list, setList] = useState<ConversationListResponse | null>(null)
  const [detail, setDetail] = useState<ConversationDetailResponse | null>(null)
  const [audits, setAudits] = useState<ToolAuditListResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fail = (err: unknown) =>
    setError(err instanceof ApiError ? `${err.message}（${err.code}）` : String(err))

  // 记录最新一次选中的会话：快速切换时先发的请求可能后到，
  // 不校验就会把旧会话的详情写进当前选中项。
  const latestRequest = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .listConversations({ limit: 50 })
      .then(({ data }) => {
        if (!cancelled) setList(data)
      })
      .catch((err) => {
        if (!cancelled) fail(err)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const open = useCallback(
    async (id: string) => {
      setSelected(id)
      setError(null)
      latestRequest.current = id
      // 立刻清空旧内容，避免新会话短暂显示上一条的详情
      setDetail(null)
      setAudits(null)
      try {
        const [conversation, audit] = await Promise.all([
          api.getConversation(id, true),
          api.listToolAudits({ conversation_id: id, limit: 100 }),
        ])
        if (latestRequest.current !== id) return // 已切到别的会话，丢弃
        setDetail(conversation.data)
        setAudits(audit.data)
      } catch (err) {
        if (latestRequest.current === id) fail(err)
      }
    },
    [],
  )

  return (
    <div className="history-page">
      <aside className="history-list">
        <h3>会话记录 {list ? `(${list.total})` : ''}</h3>
        {!list || list.total === 0 ? (
          <p className="dim">还没有任何会话。</p>
        ) : (
          list.conversations.map((c) => (
            <button
              key={c.conversation_id}
              className={selected === c.conversation_id ? 'conv active' : 'conv'}
              onClick={() => void open(c.conversation_id)}
            >
              <span className="conv-title">{c.title ?? '(无标题)'}</span>
              <span className="conv-meta">
                {c.message_count} 条消息 · {c.user_id}
                {c.has_summary && <span className="tag">含摘要</span>}
              </span>
              <span className="conv-meta">
                {new Date(c.updated_at).toLocaleString('zh-CN')}
              </span>
            </button>
          ))
        )}
      </aside>

      <section className="history-detail">
        {error && <div className="kb-notice err">{error}</div>}
        {!detail ? (
          <p className="dim">从左侧选择一个会话查看完整执行链路。</p>
        ) : (
          <>
            <header className="detail-head">
              <h3>{detail.title ?? '(无标题)'}</h3>
              <div className="detail-head-actions">
                <span className="dim">
                  {detail.user_id} · {new Date(detail.created_at).toLocaleString('zh-CN')}
                </span>
                <button
                  className="primary"
                  onClick={() => navigate(`/chat/${encodeURIComponent(detail.conversation_id)}`)}
                >
                  继续对话
                </button>
              </div>
            </header>

            {detail.summary && (
              <div className="detail-summary">
                <span className="kb-label">历史摘要</span>
                <p>{detail.summary}</p>
              </div>
            )}

            {detail.messages.map((m) => (
              <article key={m.message_id} className={`hist-msg ${m.role}`}>
                <div className="hist-role">{m.role === 'user' ? '我' : 'Copilot'}</div>
                <div className="hist-body">
                  <div className="hist-text">{m.content}</div>
                  {m.trace && <TraceViewer trace={m.trace as unknown as ExecutionTrace} />}
                </div>
              </article>
            ))}

            {audits && audits.total > 0 && (
              <section className="audit">
                <h4>工具调用审计 ({audits.total})</h4>
                <table className="kb-table">
                  <thead>
                    <tr>
                      <th>工具</th>
                      <th>类型</th>
                      <th>结果</th>
                      <th>缓存</th>
                      <th>幂等重放</th>
                      <th>request_id</th>
                      <th>耗时</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audits.items.map((a) => (
                      <tr key={a.audit_id}>
                        <td>
                          <code>{a.tool_name}</code>
                        </td>
                        <td>
                          <span className={a.is_write ? 'tag write' : 'tag'}>
                            {a.is_write ? '写' : '只读'}
                          </span>
                        </td>
                        <td>
                          <span className={a.success ? 'tag ok' : 'tag danger'}>
                            {a.success ? '成功' : (a.error_code ?? '失败')}
                          </span>
                        </td>
                        <td>{a.cache_hit ? '命中' : '—'}</td>
                        <td>{a.idempotent_replay ? '是' : '—'}</td>
                        <td className="kb-meta">{a.request_id ?? '—'}</td>
                        <td>{a.elapsed_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </>
        )}
      </section>
    </div>
  )
}
