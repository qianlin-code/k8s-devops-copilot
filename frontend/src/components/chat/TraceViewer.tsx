import { useState } from 'react'

import type { ExecutionTrace } from '@/api/types'
import './TraceViewer.css'

// 键名必须覆盖 backend/app/agent/state_machine.py 里 record() 的全部节点名，
// 漏一个就会在「执行链路」面板里显示英文原名
const NODE_LABELS: Record<string, string> = {
  route: '路由决策',
  execute_tool: '执行工具',
  execute_confirmed_write: '执行已确认的写操作',
  verify_sufficiency: '信息充分性校验',
  generate_answer: '生成回答',
  await_write_confirmation: '等待用户确认写操作',
  skip_already_executed_call: '跳过已执行的调用',
  skip_repeated_failed_call: '跳过已失败的调用',
  max_steps_exceeded: '达到最大步数上限',
  settle_insufficient: '多次尝试无果，归纳已有信息',
}

const ACTION_LABELS: Record<string, string> = {
  answer: '直接回答',
  call_tool: '调用工具',
  insufficient: '信息不足',
}

type TabKey = 'pipeline' | 'retrieval' | 'agent' | 'tools' | 'context'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'pipeline', label: '执行链路' },
  { key: 'retrieval', label: '检索过程' },
  { key: 'agent', label: 'Agent 决策' },
  { key: 'tools', label: '工具调用' },
  { key: 'context', label: '上下文与安全' },
]

export function TraceViewer({ trace }: { trace: ExecutionTrace }) {
  const [tab, setTab] = useState<TabKey>('pipeline')

  return (
    <div className="trace">
      <div className="trace-head">
        <span className="trace-id" title="全链路追踪 ID">
          trace {trace.trace_id.slice(0, 12)}
        </span>
        <span className="trace-meta">{trace.total_elapsed_ms} ms</span>
        <span className="trace-meta">
          {trace.retrieval.hybrid_enabled ? '混合检索' : '纯向量'}
        </span>
        <span className="trace-meta">
          {trace.retrieval.rerank_applied ? 'Rerank 已应用' : 'Rerank 未应用'}
        </span>
      </div>

      <nav className="trace-tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={tab === t.key ? 'trace-tab active' : 'trace-tab'}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="trace-body">
        {tab === 'pipeline' && <PipelineTab trace={trace} />}
        {tab === 'retrieval' && <RetrievalTab trace={trace} />}
        {tab === 'agent' && <AgentTab trace={trace} />}
        {tab === 'tools' && <ToolsTab trace={trace} />}
        {tab === 'context' && <ContextTab trace={trace} />}
      </div>
    </div>
  )
}

function PipelineTab({ trace }: { trace: ExecutionTrace }) {
  return (
    <ol className="steps">
      {trace.steps.map((step) => (
        <li key={step.step} className={`step step-${step.node}`}>
          <span className="step-no">{step.step}</span>
          <div className="step-main">
            <div className="step-name">{NODE_LABELS[step.node] ?? step.node}</div>
            <StepDetail node={step.node} detail={step.detail ?? {}} />
          </div>
        </li>
      ))}
    </ol>
  )
}

function StepDetail({ node, detail }: { node: string; detail: Record<string, unknown> }) {
  if (node === 'route') {
    return (
      <div className="step-detail">
        <span className="tag">{ACTION_LABELS[String(detail.action)] ?? String(detail.action)}</span>
        {detail.tool_name ? <code>{String(detail.tool_name)}</code> : null}
        <span className="conf">置信度 {Number(detail.confidence ?? 0).toFixed(2)}</span>
        <p className="reason">{String(detail.reasoning ?? '')}</p>
      </div>
    )
  }
  if (node === 'verify_sufficiency') {
    const missing = (detail.missing_information as string[] | undefined) ?? []
    return (
      <div className="step-detail">
        <span className={detail.sufficient ? 'tag ok' : 'tag warn'}>
          {detail.sufficient ? '信息充分' : '信息不足'}
        </span>
        <p className="reason">{String(detail.reasoning ?? '')}</p>
        {missing.length > 0 && (
          <ul className="missing">
            {missing.map((m) => (
              <li key={m}>缺失：{m}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }
  if (node === 'execute_tool' || node === 'execute_confirmed_write') {
    return (
      <div className="step-detail">
        <code>{String(detail.tool_name)}</code>
        {detail.is_write ? <span className="tag write">写操作</span> : <span className="tag">只读</span>}
        <span className={detail.success ? 'tag ok' : 'tag danger'}>
          {detail.success ? '成功' : String(detail.error_code ?? '失败')}
        </span>
        {detail.cache_hit ? <span className="tag">缓存命中</span> : null}
        {detail.idempotent_replay ? <span className="tag">幂等重放</span> : null}
        <span className="conf">{String(detail.elapsed_ms ?? 0)} ms</span>
      </div>
    )
  }
  if (node === 'await_write_confirmation') {
    return (
      <div className="step-detail">
        <code>{String(detail.tool_name)}</code>
        <span className="tag write">已暂停，等待确认</span>
      </div>
    )
  }
  if (node === 'skip_already_executed_call' || node === 'skip_repeated_failed_call') {
    return (
      <div className="step-detail">
        <code>{String(detail.tool_name)}</code>
        {detail.previous_error ? (
          <span className="tag danger">上次失败：{String(detail.previous_error)}</span>
        ) : (
          <span className="tag ok">本轮已成功执行过</span>
        )}
      </div>
    )
  }
  if (node === 'max_steps_exceeded') {
    return (
      <div className="step-detail">
        <span className="tag warn">
          已用 {String(detail.rounds_used)} / 上限 {String(detail.limit)} 轮
        </span>
      </div>
    )
  }
  if (node === 'settle_insufficient') {
    return (
      <div className="step-detail">
        <span className="tag warn">路由反复给出无法执行的动作，提前收敛</span>
      </div>
    )
  }
  return null
}

function RetrievalTab({ trace }: { trace: ExecutionTrace }) {
  const { query_rewrite: rewrite, stages, citations } = trace.retrieval
  return (
    <>
      <section className="block">
        <h4>查询改写</h4>
        {rewrite.applied ? (
          <div className="rewrite">
            <div>
              <span className="label">原始</span>
              <code>{rewrite.original}</code>
            </div>
            <div>
              <span className="label">改写后</span>
              <code className="hl">{rewrite.rewritten}</code>
            </div>
            {rewrite.keywords && rewrite.keywords.length > 0 && (
              <div>
                <span className="label">关键词</span>
                {rewrite.keywords.map((k) => (
                  <span key={k} className="tag">
                    {k}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : (
          <p className="dim">未改写（{rewrite.skip_reason ?? '无需改写'}）</p>
        )}
      </section>

      <section className="block">
        <h4>检索阶段</h4>
        <table className="grid">
          <thead>
            <tr>
              <th>阶段</th>
              <th>命中</th>
              <th>耗时</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody>
            {stages.map((s) => (
              <tr key={s.name}>
                <td>
                  <code>{s.name}</code>
                </td>
                <td>{s.hit_count}</td>
                <td>{s.elapsed_ms} ms</td>
                <td className="dim">{s.note ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="block">
        <h4>引用片段 ({citations.length})</h4>
        {citations.length === 0 ? (
          <p className="dim">本轮没有命中相关片段，Agent 不会基于知识库编造答案。</p>
        ) : (
          citations.map((c) => (
            <article key={c.chunk_id} className="citation">
              <header>
                <span className="cite-no">[{c.index}]</span>
                <span className="cite-label">{c.citation_label}</span>
                <span className="cite-score">{c.rerank_score.toFixed(3)}</span>
                {c.rank_before !== c.rank_after && (
                  <span className="tag" title="Rerank 前后排名变化">
                    #{c.rank_before} → #{c.rank_after}
                  </span>
                )}
              </header>
              <p>{c.text}</p>
            </article>
          ))
        )}
      </section>
    </>
  )
}

function AgentTab({ trace }: { trace: ExecutionTrace }) {
  const evidence = trace.answer_evidence ?? []
  return (
    <>
      <section className="block">
        <h4>
          回答证据映射 ({evidence.length})
          {trace.answer_generation ? (
            <span className={trace.answer_generation.status === 'verified' ? 'tag ok' : 'tag warn'}>
              {trace.answer_generation.status === 'verified' ? '已验证' : '已降级'}
            </span>
          ) : null}
        </h4>
        {trace.answer_generation ? (
          <p className="dim">
            结构化生成 {trace.answer_generation.attempts} 次
            {trace.answer_generation.fallback_reason
              ? `，原因：${trace.answer_generation.fallback_reason}`
              : ''}
          </p>
        ) : null}
        {evidence.length === 0 ? (
          <p className="dim">该历史回答未记录逐项证据。</p>
        ) : (
          <ol className="answer-evidence">
            {evidence.map((item) => (
              <li key={`${item.item_index}-${item.source_id}`}>
                <header>
                  <span className="cite-no">[{item.source_id}]</span>
                  <span className="tag">
                    {item.section === 'conclusion' ? '结论' : '证据步骤'}
                  </span>
                  <span className="evidence-origin">
                    {item.evidence_kind === 'knowledge'
                      ? item.citation_label
                      : `${item.tool_name ?? '工具'}${item.json_pointer ?? ''}`}
                  </span>
                </header>
                <p>{item.rendered_text}</p>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section className="block">
        <h4>
          路由决策 ({trace.route_decisions.length} / 上限 {trace.agent_max_steps} 轮)
        </h4>
        {trace.route_decisions.map((d) => (
          <div key={d.round} className="decision">
            <header>
              <span className="round">第 {d.round} 轮</span>
              <span className="tag">{ACTION_LABELS[d.action] ?? d.action}</span>
              {d.tool_name ? <code>{d.tool_name}</code> : null}
              <span className="conf">置信度 {d.confidence.toFixed(2)}</span>
            </header>
            <p className="reason">{d.reasoning}</p>
            {d.tool_arguments && Object.keys(d.tool_arguments).length > 0 && (
              <pre>{JSON.stringify(d.tool_arguments, null, 2)}</pre>
            )}
          </div>
        ))}
      </section>

      {trace.sufficiency && (
        <section className="block">
          <h4>最终充分性判定</h4>
          <span className={trace.sufficiency.sufficient ? 'tag ok' : 'tag warn'}>
            {trace.sufficiency.sufficient ? '信息充分' : '信息不足'}
          </span>
          <p className="reason">{trace.sufficiency.reasoning}</p>
          {trace.sufficiency.missing_information &&
            trace.sufficiency.missing_information.length > 0 && (
              <ul className="missing">
                {trace.sufficiency.missing_information.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            )}
          {trace.sufficiency.suggested_next_step && (
            <p className="dim">建议：{trace.sufficiency.suggested_next_step}</p>
          )}
        </section>
      )}
    </>
  )
}

function ToolsTab({ trace }: { trace: ExecutionTrace }) {
  if (trace.tool_calls.length === 0) {
    return <p className="dim">本轮未调用任何工具。</p>
  }
  return (
    <>
      {trace.tool_calls.map((call, i) => (
        <section key={`${call.tool_name}-${i}`} className="block">
          <h4>
            <code>{call.tool_name}</code>
            {call.is_write ? (
              <span className="tag write">写操作</span>
            ) : (
              <span className="tag">只读</span>
            )}
            <span className={call.success ? 'tag ok' : 'tag danger'}>
              {call.success ? '成功' : call.error_code}
            </span>
            {call.cache_hit && <span className="tag">缓存命中</span>}
            {call.idempotent_replay && <span className="tag">幂等重放</span>}
            <span className="conf">{call.elapsed_ms} ms</span>
          </h4>
          <div className="io">
            <div>
              <span className="label">入参</span>
              <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
            </div>
            <div>
              <span className="label">出参</span>
              <pre>
                {call.success
                  ? JSON.stringify(call.result, null, 2)
                  : `${call.error_code}: ${call.error_message}`}
              </pre>
            </div>
          </div>
        </section>
      ))}
    </>
  )
}

function ContextTab({ trace }: { trace: ExecutionTrace }) {
  const { context, security } = trace
  return (
    <>
      <section className="block">
        <h4>多轮上下文</h4>
        <table className="grid">
          <tbody>
            <tr>
              <td>总轮次</td>
              <td>{context.total_turns}</td>
            </tr>
            <tr>
              <td>窗口内轮次</td>
              <td>{context.windowed_turns}</td>
            </tr>
            <tr>
              <td>是否已摘要</td>
              <td>{context.summarized ? `是（压缩 ${context.summary_source_turns} 轮）` : '否'}</td>
            </tr>
            {context.degrade_reason && (
              <tr>
                <td>降级原因</td>
                <td className="dim">{context.degrade_reason}</td>
              </tr>
            )}
          </tbody>
        </table>
        {context.summary && <p className="summary">{context.summary}</p>}
      </section>

      <section className="block">
        <h4>安全防护</h4>
        <p>
          输入标记：
          {security.input_flags && security.input_flags.length > 0 ? (
            security.input_flags.map((f) => (
              <span key={f} className="tag warn">
                {f}
              </span>
            ))
          ) : (
            <span className="dim">无</span>
          )}
        </p>
        <p>
          输出屏蔽：
          {security.output_redactions && security.output_redactions.length > 0 ? (
            security.output_redactions.map((r) => (
              <span key={r} className="tag warn">
                {r}
              </span>
            ))
          ) : (
            <span className="dim">无</span>
          )}
        </p>
      </section>
    </>
  )
}
