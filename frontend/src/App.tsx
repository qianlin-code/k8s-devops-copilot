import { useCallback, useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import { api, getApiKey, setApiKey } from '@/api/client'
import type { HealthResponse } from '@/api/types'
import ChatPage from '@/pages/ChatPage'
import HistoryPage from '@/pages/HistoryPage'
import KnowledgeBasePage from '@/pages/KnowledgeBasePage'
import './App.css'

const USER_STORAGE = 'copilot.userId'
const NAV = [
  { to: '/chat', label: '对话' },
  { to: '/knowledge', label: '知识库' },
  { to: '/history', label: '历史记录' },
]

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [offline, setOffline] = useState(false)
  const [userId, setUserId] = useState(
    () => localStorage.getItem(USER_STORAGE) ?? 'u-1001',
  )
  const [keyDraft, setKeyDraft] = useState(getApiKey)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const refreshHealth = useCallback(async () => {
    try {
      const { data } = await api.health()
      setHealth(data)
      setOffline(false)
    } catch {
      // 探活已带 4s 超时，失败即判离线而不是一直显示"连接中"
      setOffline(true)
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
    // 心跳轮询：后端重启或从繁忙中恢复后能自动转回在线，无需刷新页面
    const timer = setInterval(() => void refreshHealth(), 15_000)
    return () => clearInterval(timer)
  }, [refreshHealth])

  const saveSettings = () => {
    const nextUser = userId.trim() || 'u-1001'
    setApiKey(keyDraft)
    localStorage.setItem(USER_STORAGE, nextUser)
    setUserId(nextUser)
    setSettingsOpen(false)
    // 不 reload：getApiKey() 每次请求都读 localStorage，userId 是 state 直接生效。
    // 刷新页面会中断进行中的 SSE 对话，用户的提问白跑一遍。
    void refreshHealth()
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <strong>Support Copilot</strong>
          <span>RAG + Agent 闭环</span>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? 'nav-item active' : 'nav-item')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <button className="settings-btn" onClick={() => setSettingsOpen((v) => !v)}>
            设置
          </button>
          {offline ? (
            <div className="status err">后端未连接</div>
          ) : health ? (
            <div className="status">
              {/* 生产环境 /health 不返回内部拓扑（无鉴权端点不该泄露），
                  所以这些字段可能为 null，只在 dev 下有值 */}
              {health.llm_provider ? (
                <>
                  <div>
                    LLM <code>{health.llm_provider}</code>
                  </div>
                  <div>
                    Embed <code>{health.embedding_provider}</code>
                  </div>
                  <div
                    className="collection"
                    title={health.collection_name ?? undefined}
                  >
                    {health.collection_name}
                  </div>
                </>
              ) : (
                <div>
                  后端在线 <code>{health.environment}</code>
                </div>
              )}
            </div>
          ) : (
            <div className="status">连接中…</div>
          )}
        </div>
      </aside>

      <main className="main">
        {settingsOpen && (
          <div className="settings">
            <label>
              API Key
              <input
                value={keyDraft}
                placeholder="与后端 .env 的 API_KEY 一致"
                onChange={(e) => setKeyDraft(e.target.value)}
              />
            </label>
            <label>
              当前用户 ID
              <input value={userId} onChange={(e) => setUserId(e.target.value)} />
            </label>
            <button className="primary" onClick={saveSettings}>
              保存并刷新
            </button>
          </div>
        )}

        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage userId={userId} />} />
          <Route path="/chat/:conversationId" element={<ChatPage userId={userId} />} />
          <Route path="/knowledge" element={<KnowledgeBasePage userId={userId} />} />
          <Route path="/history" element={<HistoryPage userId={userId} />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  )
}
