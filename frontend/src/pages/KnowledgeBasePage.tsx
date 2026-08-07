import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '@/api/client'
import type {
  DocumentListResponse,
  SedimentationEntry,
  SedimentationListResponse,
} from '@/api/types'
import './KnowledgeBasePage.css'

export default function KnowledgeBasePage({ userId }: { userId: string }) {
  const [docs, setDocs] = useState<DocumentListResponse | null>(null)
  const [pending, setPending] = useState<SedimentationListResponse | null>(null)
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [strategy, setStrategy] = useState('markdown')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  // 与后端 IngestTextRequest.max_length 保持一致
  const MAX_DOCUMENT_CHARS = 1_000_000
  const overLimit = content.length > MAX_DOCUMENT_CHARS

  const refresh = useCallback(async () => {
    try {
      const [documents, sedimentations] = await Promise.all([
        api.listDocuments(),
        api.listSedimentations('pending'),
      ])
      setDocs(documents.data)
      setPending(sedimentations.data)
    } catch (err) {
      setNotice({
        kind: 'err',
        text: err instanceof ApiError ? `${err.message}（${err.code}）` : String(err),
      })
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const run = async (task: () => Promise<string>) => {
    setBusy(true)
    setNotice(null)
    try {
      setNotice({ kind: 'ok', text: await task() })
      await refresh()
    } catch (err) {
      setNotice({
        kind: 'err',
        text:
          err instanceof ApiError
            ? `${err.message}（${err.code}，trace ${err.traceId.slice(0, 12)}）`
            : String(err),
      })
    } finally {
      setBusy(false)
    }
  }

  const upload = () =>
    run(async () => {
      const { data } = await api.ingestDocument({
        title: title.trim(),
        content,
        chunk_strategy: strategy,
      })
      setTitle('')
      setContent('')
      return `已入库「${data.document.title}」：${data.document.chunk_count} 个切片，BM25 索引 ${data.bm25_index_size} 条。`
    })

  const remove = (id: string, name: string) =>
    run(async () => {
      const { data } = await api.deleteDocument(id)
      return `已删除「${name}」，剩余向量 ${data.vector_count} 条。`
    })

  const review = (entry: SedimentationEntry, approved: boolean) =>
    run(async () => {
      const { data } = await api.reviewSedimentation(entry.pending_id, {
        reviewer: userId,
        approved,
        note: approved ? '人工审核通过' : '人工审核驳回',
      })
      return approved
        ? `已通过并写入知识库（文档 ${data.kb_document_id?.slice(0, 8)}）。`
        : '已驳回，未写入知识库。'
    })

  return (
    <div className="kb-page">
      <h2>知识库管理</h2>
      {notice && <div className={`kb-notice ${notice.kind}`}>{notice.text}</div>}

      <section className="kb-card">
        <h3>上传文档</h3>
        <div className="kb-form">
          <input
            placeholder="文档标题"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="markdown">Markdown 结构化分块</option>
            <option value="char">字符重叠分块</option>
          </select>
        </div>
        <textarea
          rows={9}
          maxLength={MAX_DOCUMENT_CHARS}
          placeholder="粘贴 Markdown 或纯文本正文。Markdown 策略会按标题层级切分并保留标题链。"
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        {/* 就地显示用量，别让用户提交后才吃 422 */}
        <div className={overLimit ? 'char-count over' : 'char-count'}>
          {content.length.toLocaleString()} / {MAX_DOCUMENT_CHARS.toLocaleString()} 字
          {overLimit && ' — 超出上限，请拆分为多篇文档'}
        </div>
        <button
          className="primary"
          onClick={upload}
          disabled={busy || !title.trim() || !content.trim() || overLimit}
        >
          {busy ? '处理中…' : '入库'}
        </button>
      </section>

      <section className="kb-card">
        <h3>
          待审沉淀队列
          {pending && pending.total > 0 && <span className="badge">{pending.total}</span>}
        </h3>
        <p className="kb-hint">
          标记的优质对话不会自动入库，必须在此人工确认。生产级方案还会加入相似度去重、
          质量评分与多级审校。
        </p>
        {!pending || pending.total === 0 ? (
          <p className="kb-empty">当前没有待审条目。</p>
        ) : (
          pending.entries.map((entry) => (
            <article key={entry.pending_id} className="pending">
              <header>
                <strong>{entry.proposed_title}</strong>
                <span className="kb-meta">由 {entry.marked_by} 标记</span>
                {entry.duplicate_of_document_id && (
                  <span className="tag warn" title={`相似度 ${entry.duplicate_score?.toFixed(2)}`}>
                    疑似重复
                  </span>
                )}
                {entry.quality_score != null && (
                  <span className={entry.quality_score >= 0.8 ? 'tag ok' : 'tag warn'}>
                    质量分 {entry.quality_score.toFixed(2)}
                  </span>
                )}
                {entry.auto_approved && <span className="tag write">已自动通过</span>}
              </header>
              {entry.quality_reasoning && (
                <p className="kb-quality-note">{entry.quality_reasoning}</p>
              )}
              <div className="pending-body">
                <div>
                  <span className="kb-label">问题</span>
                  <p>{entry.question}</p>
                </div>
                <div>
                  <span className="kb-label">回答</span>
                  <p>{entry.answer}</p>
                </div>
              </div>
              <div className="pending-actions">
                <button className="primary" disabled={busy} onClick={() => review(entry, true)}>
                  通过并入库
                </button>
                <button className="danger" disabled={busy} onClick={() => review(entry, false)}>
                  驳回
                </button>
              </div>
            </article>
          ))
        )}
      </section>

      <section className="kb-card">
        <h3>
          文档列表
          {docs && (
            <span className="kb-meta">
              集合 {docs.collection_name} · 向量 {docs.vector_count} · BM25{' '}
              {docs.bm25_index_size}
            </span>
          )}
        </h3>
        {!docs || docs.total === 0 ? (
          <p className="kb-empty">知识库为空，先上传一篇文档。</p>
        ) : (
          <table className="kb-table">
            <thead>
              <tr>
                <th>标题</th>
                <th>来源</th>
                <th>分块策略</th>
                <th>切片</th>
                <th>字符</th>
                <th>创建时间</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {docs.documents.map((doc) => (
                <tr key={doc.document_id}>
                  <td>{doc.title}</td>
                  <td>
                    <span className={doc.source === 'sedimentation' ? 'tag write' : 'tag'}>
                      {doc.source === 'sedimentation' ? '对话沉淀' : doc.source}
                    </span>
                  </td>
                  <td>
                    <code>{doc.chunk_strategy}</code>
                  </td>
                  <td>{doc.chunk_count}</td>
                  <td>{doc.char_count}</td>
                  <td className="kb-meta">
                    {new Date(doc.created_at).toLocaleString('zh-CN')}
                  </td>
                  <td>
                    <button
                      className="danger"
                      disabled={busy}
                      onClick={() => remove(doc.document_id, doc.title)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
