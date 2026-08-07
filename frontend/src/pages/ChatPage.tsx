import { useState } from 'react'

import { ApiError, api } from '@/api/client'
import { MessageList } from '@/components/chat/MessageList'
import { useChat } from '@/hooks/useChat'
import './ChatPage.css'

export default function ChatPage({ userId }: { userId: string }) {
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
  } = useChat(userId)
  const [draft, setDraft] = useState('')
  const [marking, setMarking] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)

  const submit = () => {
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    setNotice(null)
    void send(text)
  }

  const markForSedimentation = async () => {
    if (!conversationId || marking) return
    setMarking(true)
    setNotice(null)
    try {
      await api.markSedimentation({
        conversation_id: conversationId,
        marked_by: userId,
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
          <button onClick={reset} disabled={turns.length === 0 || busy}>
            新建会话
          </button>
        </div>
      </header>

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
          placeholder="描述遇到的问题，例如：账号 u-1001 登录提示 403 Forbidden 该怎么处理（Enter 发送，Shift+Enter 换行）"
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
