import { type FormEvent, useState } from 'react'

import { ApiError, api, saveLoginInfo } from '@/api/client'
import './AuthPage.css'

type AuthMode = 'login' | 'register'

interface AuthPageProps {
  onAuthenticated: () => void
}

export default function AuthPage({ onAuthenticated }: AuthPageProps) {
  const [mode, setMode] = useState<AuthMode>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [organizationName, setOrganizationName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      if (mode === 'register') {
        await api.register({
          username: username.trim(),
          password,
          organization_name: organizationName.trim(),
        })
      }
      const { data } = await api.login({ username: username.trim(), password })
      saveLoginInfo(data)
      onAuthenticated()
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message}（${err.code}）` : String(err))
    } finally {
      setBusy(false)
    }
  }

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode)
    setError(null)
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-brand">
          <strong>K8s / DevOps Copilot</strong>
          <span>RAG + Agent 智能运维闭环</span>
        </div>
        <div className="auth-tabs" role="tablist" aria-label="认证方式">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={mode === 'login' ? 'active' : ''}
            onClick={() => switchMode('login')}
          >
            登录
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            className={mode === 'register' ? 'active' : ''}
            onClick={() => switchMode('register')}
          >
            注册
          </button>
        </div>

        <form onSubmit={(event) => void submit(event)}>
          <label>
            用户名
            <input
              value={username}
              minLength={mode === 'register' ? 3 : 1}
              maxLength={128}
              autoComplete="username"
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>
          <label>
            密码
            <input
              type="password"
              value={password}
              minLength={mode === 'register' ? 8 : 1}
              maxLength={128}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>
          {mode === 'register' && (
            <label>
              组织名称
              <input
                value={organizationName}
                maxLength={255}
                autoComplete="organization"
                onChange={(event) => setOrganizationName(event.target.value)}
                required
              />
            </label>
          )}
          {error && <div className="auth-error" role="alert">{error}</div>}
          <button className="primary" type="submit" disabled={busy}>
            {busy ? '处理中…' : mode === 'login' ? '登录' : '注册并登录'}
          </button>
        </form>
      </section>
    </main>
  )
}
