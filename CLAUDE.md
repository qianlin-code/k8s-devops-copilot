# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

用中文回答。本仓库的代码注释都是中文，且只解释**为什么**这么写，不重复代码本身在做什么。

## 目录

- [项目与当前配置](#项目与当前配置)
- [常用命令](#常用命令)
- [后端架构](#后端架构)
- [前端架构与状态约定](#前端架构与状态约定)
- [前后端契约](#前后端契约)
- [Schema 变更的标准流程](#schema-变更的标准流程)
- [测试与三层验证](#测试与三层验证)
- [故障排查手册](#故障排查手册)
- [已验证的 Anti-Patterns](#已验证的-anti-patterns)
- [生产部署检查清单](#生产部署检查清单)
- [环境注意事项](#环境注意事项)

## 项目与当前配置

企业级智能客服 Copilot —— RAG + Agent 闭环系统，面向企业内部 IT 支持场景。
后端 FastAPI（Python 3.11），前端 Vite + React 18 + TypeScript。

当前 `backend/.env` 的实际配置：

| 项 | 值 |
| --- | --- |
| `LLM_PROVIDER` / 模型 | `ollama` / `qwen2.5:7b` |
| `EMBEDDING_PROVIDER` / 模型 | `ollama` / `nomic-embed-text`（768 维） |
| Rerank | `BAAI/bge-reranker-base`，本地 GPU |
| `CHUNK_STRATEGY` | `markdown` |
| `ENVIRONMENT` | `dev` |
| Qdrant 集合名 | `kb_nomic-embed-text_768`（格式：`kb_<model>_<dim>`） |
| `QWEN_JUDGE_MODEL` | `qwen-max`，RAGAS 评估裁判，固定走云端，与 `LLM_PROVIDER` 无关 |
| `QWEN_SEDIMENTATION_MODEL` | `qwen-plus`，沉淀质量初筛，成本优先 |

> 本仓库已初始化 git（首次提交见 `d5cd9ad`）。`.gitignore` 已排除 `.env`、运行时数据库
> （`data/app.db*`、`data/probe.db`）、Qdrant 存储、日志。改 `PendingSedimentation` 这类
> 表结构前仍建议先备份 `backend/data/app.db*`，SQLite 的结构变更不受 git 保护。

## 常用命令

后端命令都在 `backend/` 目录下执行。**必须显式用 `.venv\Scripts\python.exe`** ——
虚拟环境是 Python 3.11 而系统 `python` 是 3.13。

### 测试

```bash
.venv/Scripts/python -m pytest tests/ -q                          # 全量
.venv/Scripts/python -m pytest tests/test_user_isolation.py -q    # 单个文件
.venv/Scripts/python -m pytest tests/ -q -k "write_lock"          # 按名字筛选
.venv/Scripts/python -m pytest tests/contract/ -q                 # 只跑契约测试（不需要真实模型）
```

### 启动

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
cd ../frontend && npm run dev        # 前端 :5173
```

### 代码生成（改 schema 后必跑，见「Schema 变更的标准流程」）

```bash
.venv/Scripts/python scripts/generate_frontend_types.py   # → frontend/src/api/types.ts
.venv/Scripts/python scripts/export_api_examples.py       # → backend/api_examples/*.json
```

### 真实服务验证（第三层，需服务已启动）

```bash
.venv/Scripts/python scripts/seed_kb.py            # 把 data/docs 灌进运行中的服务
.venv/Scripts/python scripts/e2e_check.py          # 全部业务分支，真实模型
.venv/Scripts/python scripts/sse_check.py          # SSE 事件时序与到达间隔
.venv/Scripts/python scripts/concurrent_check.py   # 3 路并发 SSE + 轮询，压锁竞争
.venv/Scripts/python scripts/probe_lock.py         # 确认 busy_timeout 真的在排队
```

### 检索质量

```bash
.venv/Scripts/python scripts/eval_retrieval.py             # 三组对比，30 条查询
.venv/Scripts/python scripts/eval_retrieval.py --fake      # 替身模型，只验流程
.venv/Scripts/python scripts/check_reranker.py             # 确认 Rerank 真的在重排
.venv/Scripts/python scripts/diagnose_case.py q01          # 单条查询 Rerank 前后排名
```

### 端到端生成质量评估（RAGAS 风格自研指标）

```bash
.venv/Scripts/python scripts/eval_ragas.py                        # --mode local，免费，裁判走云端
.venv/Scripts/python scripts/eval_ragas.py --mode cloud           # 云端生成+裁判，会产生计费调用
.venv/Scripts/python scripts/eval_ragas.py --mode both            # 本地/云端对比 cost/quality
.venv/Scripts/python scripts/eval_ragas.py --limit 5 --save-json out.json  # 调试用，保存逐条明细
```

不装 `ragas` 库，复用 `app/llm/client.py` 自己实现：context_precision/recall 用
`eval_set.json` 的 `expected_keywords` 关键词覆盖率算（不需要 LLM）；tool_correctness
对比实际调用工具与 `expected_tool`（纯代码）；faithfulness + answer_relevancy 合并成
一次结构化裁判调用，减半费用。裁判固定用 `QWEN_JUDGE_MODEL`（默认 `qwen-max`），
不受 `LLM_PROVIDER` 影响——本地 7B 模型评自己生成的答案没有意义，裁判必须独立于被测链路。

`data/eval_set.json` 已扩充 `gold_answer`/`expected_outcome`/`expected_tool` 三个字段。
**局限**：这 30 条全是不含账号 ID 的知识性问题，按路由规则会走 `direct_answer`，
`expected_tool` 恒为 `null`——`tool_correctness` 指标在本数据集上不能证明工具路由能力。

### 前端

```bash
npx tsc --noEmit        # 类型检查
npx vite build          # 构建
```

## 后端架构

### 闭环链路

`app/services/chat_service.py` 负责编排，`app/agent/state_machine.py` 是 Agent 核心。

```
输入防护 → 组装上下文 → 检索 → 路由决策 ─┬─ answer      → 充分性校验 → 生成回答
                                        ├─ call_tool   → 执行工具 → 充分性校验 → 回到路由
                                        └─ insufficient
```

检索链路（`app/rag/retriever.py`）：可选查询改写 → 向量 ∥ BM25 → RRF 融合 → BGE 重排 →
分数阈值过滤。融合用**排名**而非原始分数，因为余弦相似度和 BM25 分数量纲不可比。
阈值不能省：没有它，无关查询也会返回 Top-K，「检索为空」分支永远走不到，
模型就会拿着无关片段硬编答案。

路由是一次结构化 LLM 调用（Pydantic 约束输出），不是关键词匹配。
prompt 里会注入当前提问用户的真实账号 ID —— 否则模型会自己编一个。

### 工具安全

`app/agent/tools/` —— `registry.py` 注册，`base.py` 定义契约，`executor.py` 执行。

- 写工具在执行前中断，请求返回待确认动作，确认之前审计表为空。
- 启动时跑 `assert_write_contract()`：写工具必须禁缓存且声明 `request_id`。
  只有读工具可以缓存。
- 一轮之内，失败的调用不重试，成功的写操作**按工具名**去重 ——
  LLM 每轮生成的 `reason` 文本都不同，按完整参数签名去重会失效，确认卡片会反复弹出。

### 访问控制

API Key 只证明「是我们的客户端」，不代表「可以读任何人的数据」。所有以 `conversation_id`
为入口的操作都要过 `_load_owned_conversation`。历史类端点的 `user_id` 是**必填**。
跨用户访问返回 **404 而非 403**。

`/health` 无需鉴权，所以 `ENVIRONMENT=prod` 下只返回 `status` 和 `environment`。
需要拓扑信息时用 `/readiness`（要鉴权）。

### 供应商切换

Ollama 和阿里云千问都走 OpenAI 兼容协议，所以 `app/llm/client.py` 只有一个客户端，
切换只是换 `base_url` / `api_key` / `model`。不同 provider 的 Embedding 维度不同，
所以 Qdrant 集合名里嵌了模型名 + 维度 —— 切换 provider 是换一个集合，
而不是在维度不匹配上崩掉。

Rerank 只用本地模型，因为检索质量的基准必须固定，调优前后才有可比性。

### 异常分级

见 `app/llm/error_mapping.py`。可重试：LLM 超时、限流、连接中断、向量库临时不可用。
不可重试：参数校验、凭证错误、工具权限、业务规则冲突。

**Ollama 会把传输层故障包装成 HTTP 400 返回。** 若按「400 一律不重试」处理，
一次网络抖动就会让整个入库失败。所以映射逻辑会检查 400 的响应体里有没有传输层特征
（`connection reset` / `wsarecv` / `EOF` / `broken pipe`）。
`tests/test_error_mapping.py` 里 19 个测试锁定这些边界。

### 沉淀自动初筛

`app/knowledge/sedimentation.py::SedimentationService._screen`，在 `mark()` 内同步触发，
标记即初筛，不是等人工点开才评：

1. **查重优先**：用现有 embedding 对 `question+answer` 算向量，搜已有知识库 Top-1，
   相似度 ≥ `_DUPLICATE_SIMILARITY_THRESHOLD`（0.92，原始余弦相似度，不经过 Rerank）
   即判重复，写 `duplicate_of_document_id`/`duplicate_score`，**不再打质量分**——
   重复内容没有质量可言，直接留人工判断合并还是驳回。
2. **非重复才打分**：用 `QWEN_SEDIMENTATION_MODEL` 结构化裁判完整度/可回答性/敏感信息，
   含敏感信息时 `quality_score` 直接清零。分数 ≥ `_AUTO_APPROVE_QUALITY_THRESHOLD`（0.8）
   且非敏感 → 自动 `approve()`，`reviewer` 固定传 `AUTO_QUALITY_REVIEWER`
   （`"system:auto-quality"`），`entry.auto_approved=True`；否则留 `pending` 给人工看分审。
3. **降级路径**：`_embed`/`_store`/`_quality_llm` 任一为 `None`（比如没配 `QWEN_API_KEY`）
   或调用抛 `AppError`，都静默降级为纯人工审核，不阻塞 `mark()` 本身——评估服务不可用
   不该让"标记"这个动作本身失败。

留痕：`approve()`/`reject()` 都会写 `reviewed_by`；只有自动路径会把 `auto_approved` 置 `True`，
人工审核路径 `reviewed_by` 是审核人 ID。`marked_by`（谁标记的原始对话）与 `reviewed_by`
（谁批准的）语义不同，分开记录才能回答"这条内容是谁批的、为什么批"。

测试见 `tests/test_sedimentation_screening.py`，覆盖高分自动通过、低分转人工、
敏感信息强制人工、重复命中跳过打分、相似但不到阈值不误判、初筛服务不可用降级五条路径。

## 前端架构与状态约定

状态全部集中在 `useChat`，页面组件只负责渲染。这几条约定是联调核心，改动前先读懂：

**`frontend/src/hooks/useChat.ts`**

- `turnSeq` 是**权威回合序号**。send / confirm / reset 时递增。
  响应到达时先比对 `turnSeq`，不匹配就丢弃 —— 否则用户在等待期间点「新建会话」，
  旧响应会追加到新会话里。
- `inFlight.current` 是**权威的「请求进行中」标志**，`busy` state 只用于渲染。
  两者不能颠倒：`send` 闭包捕获的是旧的 `busy` 值，连点两次时第二次仍看到 `false`，
  会产生重复会话。
- 确认写操作时，`pendingWrite` 卡片留到**请求成功后**再清。提前清掉的话，
  请求失败或中断后 token 无处可用，用户只能重开会话。

**`frontend/src/api/client.ts`**

- SSE parser 必须跳过以 `:` 开头的注释帧（那是心跳），否则会被当成空事件。
- `event: error` 用后端透出的 `http_status`，**不能写死 500**。
- 流式请求用**空闲超时**（45s 收不到任何数据）而非总时长上限 ——
  总时长上限会误杀「慢但正常」的长链路。
- `setApiKey('')` 走清除而非存空串，否则后续请求会带一个空 header，
  服务端返回 401 但用户以为「已经填过了」。

**`frontend/src/api/types.ts` 是生成物，禁止手写。**

**`frontend/src/pages/HistoryPage.tsx`** —— 切换会话时用 `latestRequest` ref 校验，
先发的请求可能后到，不校验会把旧会话的详情写进当前选中项。

## 前后端契约

后端 schema 是唯一事实来源：

1. 全部 Request/Response 继承 `StrictBaseModel`（`extra="forbid"`，时间统一 UTC ISO）。
2. `tests/contract/` 打掉 LLM / Qdrant / Rerank，只校验响应 JSON 与 schema 一致。
3. 所有错误统一为 `{code, message, trace_id, retryable, details}`，堆栈只进日志。

SSE 特有：`error` 事件的载荷多一个 `http_status`，因为帧发出时 HTTP 状态码已经是 200 了，
客户端无从得知这个错误本该是几百。心跳是 SSE 注释帧（`: keep-alive`），每 10s 一个。

## Schema 变更的标准流程

改动 `app/schemas/*.py` 后按顺序全部走完，漏一步就会出现契约漂移：

```bash
# 1. 改 app/schemas/*.py

# 2-3. 重新生成两个产物（在 backend/ 下）
.venv/Scripts/python scripts/generate_frontend_types.py
.venv/Scripts/python scripts/export_api_examples.py

# 4. 契约测试
.venv/Scripts/python -m pytest tests/contract/ -q

# 5. 前端类型检查（在 frontend/ 下）
npx tsc --noEmit

# 6. 若改了 SSE 事件结构，同步检查 frontend/src/api/client.ts 的 parser
```

第 5 步经常能抓到连带问题：比如给 `/health` 的字段加上 `Optional` 后，
`tsc` 立刻报出侧边栏没处理 null。

### 数据库表结构变更（没有 Alembic）

项目体量还没到需要迁移框架的程度，`init_db()` 只会 `CREATE TABLE IF NOT EXISTS`，
不会给已存在的表加新列。给 `app/storage/models.py` 里的表加字段时：

- **全新环境**（`data/app.db` 不存在）：什么都不用做，`init_db()` 建表时字段就是全的。
- **已有数据要保留**：**先备份** `backend/data/app.db`（连同同目录的 `-wal`/`-shm`，
  如果存在），再跑一次性迁移脚本（如 `scripts/migrate_sedimentation.py`）用
  `ALTER TABLE ADD COLUMN` 就地加列，可重复执行、已存在的列会跳过。
- 图省事也可以直接删库重建（删 `data/app.db*` 后 `init_db()` 重新建表），
  但会丢光对话/工单/审计等本地数据，只适合真不在乎这些数据的场景。

## 测试与三层验证

`tests/conftest.py` 在导入 `app.config` **之前**设置环境变量，并给每个测试独立的库和
向量集合。`tests/fakes.py` 里的 `ScriptedLLMClient` 按**意图**分派
（router / sufficiency / rewrite / answer / summary）而非按调用顺序 ——
这样上下文摘要之类的附带调用不会让测试声明的路由脚本错位。

只跑契约测试不够。真实模型的行为和替身不同，并发缺陷在单发请求下完全看不见。

| 层 | 抓什么 | 靠它发现的问题 |
| --- | --- | --- |
| 契约测试（替身） | schema 漂移、分支覆盖 | — |
| 真实模型联调 | 幻觉 ID、过度拒答、Rerank 打分异常 | Rerank 把正确答案挤出 Top-5 |
| 并发 / 故障注入 | 锁竞争、超时、断连 | `tool_call_audits` 撞锁 |

**任何涉及写库或调用外部服务的改动，必须过第三层才算完成。**

## 故障排查手册

| 现象 | 排查位置 | 常见根因 |
| --- | --- | --- |
| `python --version` 是 3.13 而非 3.11 | `py -0p`、`$env:PATH.Split(';')` | 机器 PATH 优先于用户 PATH；本项目一律显式用 `.venv\Scripts\python.exe`，不依赖 PATH |
| 后端启动慢 / 检索阶段卡 ~300s | `run.log` 搜 `huggingface_hub` | `HF_HUB_OFFLINE` 设晚了，模块常量已固化。必须在 import 前设 |
| 前端请求 20s 后 `net::ERR_ABORTED` | `run.log` 看 `elapsed_ms` | 走了非流式接口且代理超时；对话应走 `/chat/stream` |
| 流式对话返回 `INTERNAL_ERROR` | `run.log` 搜 `chat_stream_unhandled` | 多半是 `database is locked`，检查是否有 `flush()` 后夹了慢操作 |
| 健康检查 200 但业务接口 401 | DevTools Network 看 `X-API-Key` | localStorage 没保存，或保存了空字符串 |
| 历史记录里出现别人的会话 | 请求是否带 `user_id` | 可选过滤参数默认「不过滤」 |
| SSE 事件全部在最后一起到达 | nginx / vite 代理配置 | 缺 `proxy_buffering off`，或 gzip 压了 `text/event-stream` |
| 确认写操作后卡片反复弹出 | trace 里的 `node` 序列 | 写操作去重用了完整参数签名而非工具名 |
| Rerank 后命中率反而下降 | `scripts/diagnose_case.py <id>` | Rerank 用了裸 `text` 而非 `contextual_text` |
| Rerank 增量指标为 0 | `scripts/eval_retrieval.py` 输出 | 召回宽度等于输出宽度，重排没有筛选空间；或评估集太易，基线已到天花板 |

日志位置：`backend/run.log`（stdout，结构化 JSON）、`backend/err.log`（stderr）。
按 `trace_id` 可串联一次请求的全部步骤。

## 已验证的 Anti-Patterns

这些都是本项目真实踩过并修复的，改动时不要退回去：

**1. 不要在长耗时段前用 `flush()`。** 它会开启写事务并持 SQLite 写锁到 `commit()`。
凡是检索、Agent 循环、LLM 调用之前，必须先 `commit()`。

**2. `commit()` 放在 `_resolve_conversation` 之后不够。** 那个方法内部新建会话时
已经 `flush()` 开了事务。开场写入要收成一个独立短事务（见 `_open_turn`）。

**3. 中段写入也要 `commit()`。** 工具审计在 Agent 循环中段写入，后面还有多次 LLM 调用。
只 `flush()` 等于又把锁攥到整轮结束。

**4. SSE `error` 事件必须带 `http_status`。** SSE 帧发出时 HTTP 状态码已经是 200，
客户端无法从状态码判断业务错误类型。

**5. 过滤参数默认「不过滤」最危险。** `user_id: Optional[str] = None` 会让不传参数时
返回全量数据。历史类接口一律必填。

**6. 越权访问返回 404 而非 403。** 403 等于确认「这个 id 存在但不属于你」，方便枚举。

**7. 文本表示必须三处一致。** 向量嵌入、BM25 索引、Rerank 都用 `contextual_text`
（正文前缀标题链），不能用裸 `text`。Markdown 分块把「现象 / 根因 / 处理步骤」切成独立
chunk 后，「根因」那段既不含 "403" 也不含 "登录" —— Rerank 只看裸正文时给它打 0.056，
把正确答案从第 2 位挤到第 12 位。

**8. 评估时召回宽度必须显著大于输出宽度。** 召回 5 条输出 5 条时，重排只调整这 5 条的
顺序、不改变集合成员，Hit@5 必然不动 —— 看起来像「Rerank 无效」，实则是评估设计问题。

**9. 环境变量对已 import 的模块可能无效。** `huggingface_hub` 在 import 时就把
`HF_HUB_OFFLINE` 固化成模块常量。这类变量要么在 import 前设，要么同时改模块常量。

## 生产部署检查清单

`config.py` 的 `_validate_production()` 会强制校验前三项，不满足直接拒绝启动：

- [ ] `ENVIRONMENT=prod`
- [ ] `API_KEY` 不是开发默认值，且长度 ≥ 24
- [ ] `CORS_ALLOW_ORIGINS` 不含 `*` 和 localhost
- [ ] 数据卷已挂载（SQLite 与 Qdrant 存储不能只在容器内）
- [ ] readiness 探针用 `/readiness`（`/health` 在 prod 下不返回依赖状态）
- [ ] 如需 Rerank，镜像构建时传 `--build-arg WITH_RERANK=true`
- [ ] 确认 `/health` 的响应里没有 provider / 集合名

## 环境注意事项

- **不要用 `pip install -e`。** 可编辑安装会把项目路径写进 `.pth` 文件，
  而本仓库路径含中文，Python 3.11 在 GBK 环境的中文 Windows 上启动解释器时会
  `UnicodeDecodeError`。导入问题由 `pyproject.toml` 里的 `pythonpath = ["."]` 解决。
- FlagEmbedding 在可选依赖组 `[rerank]` 里（会拉 ~2.5 GB 的 torch）。不装也能跑，
  检索会降级为 RRF 融合顺序，`trace.retrieval.rerank_applied` 为 `false`。
- 这台机器同时装了 pyenv、winget 版 Python 3.11、以及 `E:\Python` 下的 3.13。
  **不要依赖 PATH 解析**，一律显式写解释器路径或用 `py -3.11`。
- PowerShell 是 5.1：没有 `&&`、没有 `||`、没有三元运算符。用 `;` 和 `if ($?) { }`。
  含空格的路径在链式命令里会让某些 cmdlet 解析失败（`Remove-Item` 尤其明显），
  这类命令要拆成多条。
