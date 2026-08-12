import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { ApiError, api } from '@/api/client'
import { MessageList } from '@/components/chat/MessageList'
import { useChat } from '@/hooks/useChat'
import './ChatPage.css'

export default function ChatPage() {
  const {
    turns,
    conversationId,
    busy,
    progress,
    confirmingToken,
    send,
    cancel,
    confirm,
    reset,
    hydrate,
  } = useChat()
  const [draft, setDraft] = useState('')
  const [marking, setMarking] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const { conversationId: routeConversationId } = useParams<{ conversationId?: string }>()
  const navigate = useNavigate()
  // 记录已恢复过的会话 id，避免同一个会话在渲染期间被重复拉取
  const hydratedRef = useRef<string | null>(null)

  useEffect(() => {
    if (!routeConversationId || routeConversationId === hydratedRef.current) return
    hydratedRef.current = routeConversationId
    setLoadError(null)
    api
      .getConversation(routeConversationId, true)
      .then(({ data }) => hydrate(data.conversation_id, data.messages))
      .catch((err) => {
        hydratedRef.current = null
        setLoadError(err instanceof ApiError ? `会话加载失败：${err.message}（${err.code}）` : String(err))
      })
  }, [routeConversationId, hydrate])

  const submit = () => {
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    setNotice(null)
    void send(text)
  }

  const startNewConversation = () => {
    reset()
    hydratedRef.current = null
    if (routeConversationId) navigate('/chat')
  }

  const markForSedimentation = async () => {
    if (!conversationId || marking) return
    setMarking(true)
    setNotice(null)
    try {
      await api.markSedimentation({
        conversation_id: conversationId,
      })
      setNotice('已提交到待审队列，需在知识库页面人工确认后才会写入知识库。')
    } catch (err) {
      setNotice(
        err instanceof ApiError ? `标记失败：${err.message}（${err.code}）` : String(err),
      )
    } finally {
      setMarking(false)
    }
  }

  const hasAnswer = turns.some((t) => t.role === 'assistant' && !t.failed)

  return (
    <div className="chat-page">
      <header className="chat-head">
        <div>
          <h2>对话</h2>
          {conversationId && (
            <span className="chat-cid">会话 {conversationId.slice(0, 8)}</span>
          )}
        </div>
        <div className="chat-actions">
          <button onClick={markForSedimentation} disabled={!hasAnswer || marking}>
            {marking ? '提交中…' : '标记为优质对话'}
          </button>
          <button onClick={startNewConversation} disabled={turns.length === 0 || busy}>
            新建会话
          </button>
        </div>
      </header>

      {loadError && <div className="chat-notice err">{loadError}</div>}
      {notice && <div className="chat-notice">{notice}</div>}

      <MessageList
        turns={turns}
        busy={busy}
        progress={progress}
        onConfirm={(token, approved) => void confirm(token, approved)}
        confirmingToken={confirmingToken}
        onPickSample={setDraft}
      />

      <div className="composer">
        <textarea
          rows={3}
          value={draft}
          disabled={busy}
          placeholder="描述遇到的问题，例如：ops-demo 命名空间下的 Pod 一直是 Pending 状态该怎么排查（Enter 发送，Shift+Enter 换行）"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {busy ? (
          <button className="cancel" onClick={cancel}>
            取消
          </button>
        ) : (
          <button className="primary" onClick={submit} disabled={!draft.trim()}>
            发送
          </button>
        )}
      </div>
    </div>
  )
}
