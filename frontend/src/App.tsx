import { useCallback, useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import {
  AUTH_EXPIRED_EVENT,
  api,
  getCurrentUser,
  hasAccessToken,
  logout,
} from '@/api/client'
import type { HealthResponse } from '@/api/types'
import AuthPage from '@/pages/AuthPage'
import ChatPage from '@/pages/ChatPage'
import HistoryPage from '@/pages/HistoryPage'
import KnowledgeBasePage from '@/pages/KnowledgeBasePage'
import './App.css'

const NAV = [
  { to: '/chat', label: '对话' },
  { to: '/knowledge', label: '知识库', adminOnly: true },
  { to: '/history', label: '历史记录' },
]

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [offline, setOffline] = useState(false)
  const [currentUser, setCurrentUser] = useState(getCurrentUser)
  const authenticated = currentUser !== null && hasAccessToken()

  const refreshHealth = useCallback(async () => {
    try {
      const { data } = await api.health()
      setHealth(data)
      setOffline(false)
    } catch {
      setOffline(true)
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
    const timer = setInterval(() => void refreshHealth(), 15_000)
    return () => clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    const handleUnauthorized = () => {
      logout()
      setCurrentUser(null)
    }
    window.addEventListener(AUTH_EXPIRED_EVENT, handleUnauthorized)
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handleUnauthorized)
  }, [])

  if (!authenticated) {
    return <AuthPage onAuthenticated={() => setCurrentUser(getCurrentUser())} />
  }

  const isAdmin = currentUser.role === 'admin'
  const visibleNav = NAV.filter((item) => !item.adminOnly || isAdmin)
  const signOut = () => {
    logout()
    setCurrentUser(null)
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <strong>Ops Copilot</strong>
          <span>RAG + Agent 闭环</span>
        </div>
        <nav>
          {visibleNav.map((item) => (
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
          <div className="current-user">
            <span>{currentUser.username}</span>
            <small>{isAdmin ? '管理员' : '普通用户'}</small>
          </div>
          <button className="settings-btn" onClick={signOut}>退出登录</button>
          {offline ? (
            <div className="status err">后端未连接</div>
          ) : health ? (
            <div className="status">
              {health.llm_provider ? (
                <>
                  <div>LLM <code>{health.llm_provider}</code></div>
                  <div>Embed <code>{health.embedding_provider}</code></div>
                  <div className="collection" title={health.collection_name ?? undefined}>
                    {health.collection_name}
                  </div>
                </>
              ) : (
                <div>后端在线 <code>{health.environment}</code></div>
              )}
            </div>
          ) : (
            <div className="status">连接中…</div>
          )}
        </div>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:conversationId" element={<ChatPage />} />
          <Route
            path="/knowledge"
            element={isAdmin ? <KnowledgeBasePage /> : <Navigate to="/chat" replace />}
          />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  )
}
