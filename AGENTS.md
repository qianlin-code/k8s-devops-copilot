# K8s / DevOps 智能运维 Copilot 开发约定

本文件是 Codex 在本项目中的开发入口。对外项目名统一使用“K8s / DevOps 智能运维 Copilot”；目录名保留旧客服名称仅为了延续 Git 历史。

## 事实来源

| 内容 | 位置 |
| --- | --- |
| 对外介绍与快速开始 | `README.md` |
| 当前架构 | `docs/架构设计.md` |
| 指标与失败案例 | `docs/评测与失败案例.md` |
| 数据许可与来源 | `docs/数据来源.md` |
| 发布门槛与完成状态 | `项目验收标准.md` |
| 后端 Schema | `backend/app/schemas/` |
| 前端 API 类型 | `frontend/src/api/types.ts`（生成物） |

代码与文档冲突时，以当前代码、配置和可复现测试为准。不要在 README、架构和验收文档中重复维护同一组指标。

## 环境与命令

- 后端要求 Python 3.11；系统 Python 3.13 不能替代项目虚拟环境。
- 后端命令在 `backend/` 目录执行，并显式使用 `.venv/Scripts/python.exe`。
- 前端是 React 18 + TypeScript + Vite，命令在 `frontend/` 目录执行。
- 模型和 provider 以 `backend/.env` 与 `app/config.py` 为准；不得把本机密钥或私有 `.env` 内容写入文档。

```powershell
# 后端离线测试
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests\contract -q

# 启动后端
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# 数据与真实链路验证（会写入目标服务的数据库）
.venv\Scripts\python.exe scripts\seed_kb.py --docs-dir data\docs_k8s
.venv\Scripts\python.exe scripts\e2e_check.py
.venv\Scripts\python.exe scripts\sse_check.py

# 检索与生成评测
.venv\Scripts\python.exe scripts\eval_retrieval.py
.venv\Scripts\python.exe scripts\eval_rerank_threshold.py
.venv\Scripts\python.exe scripts\eval_ragas.py --mode local

# Schema 变更后的生成物
.venv\Scripts\python.exe scripts\generate_frontend_types.py
.venv\Scripts\python.exe scripts\export_api_examples.py

# 前端
npm run typecheck
npm run build
npm run e2e
```

真实 HTTP 脚本默认指向 `http://localhost:8000`，会产生会话、消息或审计数据。需要隔离时，用独立 `DATABASE_URL` 启动服务，并通过 `COPILOT_BASE` 指向该实例。

## 核心架构不变量

### RAG

- 检索顺序是查询改写 → 向量与 BM25 双路召回 → RRF → BGE Rerank → 阈值过滤。
- 向量、BM25 和 Rerank 必须使用相同的带标题上下文文本表示；不得只在某一阶段使用裸正文。
- RRF 融合排名而不是直接混合不同量纲的分数。
- 相关性阈值必须保留，确保“检索为空”分支可达；修改阈值前运行敏感性评测。
- Embedding 模型与维度绑定 Qdrant 集合名，切换 provider 使用新集合，不能向旧集合混写不同维度向量。

### Agent 与工具

- `app/services/chat_service.py` 编排全链路，`app/agent/state_machine.py` 管理有限步状态流转。
- 路由结果只能是直接回答、调用工具或信息不足，并由 Pydantic 结构化校验。
- 写工具必须禁用缓存、声明 `request_id`、确认后执行，并记录审计；只读工具才允许缓存。
- 写操作幂等作用域是 `(conversation_id, request_id)`，不能退回全局 `request_id` 唯一。
- 同一轮失败工具不重试，成功写工具按工具身份去重，避免确认循环。
- 工具白名单不得扩展为任意 shell 或 kubectl 执行。

### 鉴权与数据隔离

- 当前业务端点使用 JWT；登录/注册路由签发身份，`sub` 是用户事实来源。
- 普通用户只能访问自己的会话、消息和审计；跨用户访问统一表现为资源不存在，避免泄露存在性。
- 知识库写入、删除和审核属于 admin 能力；不要用客户端传入的 `user_id` 代替服务端身份。
- `/health` 公开且生产环境隐藏内部拓扑；`/readiness` 需要 JWT。
- 生产环境必须拒绝默认弱密钥、本地 CORS 和通配来源。

### Schema、SSE 与前端状态

- 所有请求/响应继承 `StrictBaseModel`，`extra="forbid"`；后端 Schema 是唯一接口事实来源。
- 修改 Schema 后必须依次重新生成前端类型和 API 样例，再运行契约测试与前端类型检查。
- `frontend/src/api/types.ts` 禁止手写。
- SSE 心跳是 `: keep-alive` 注释帧；错误事件必须携带原始 `http_status`；前端使用空闲超时而非总时长上限。
- 同步 Agent 链路在线程中执行；线程内使用独立 SQLAlchemy Session。
- `useChat` 的 `turnSeq` 与 `inFlight` 分别负责丢弃过期响应和阻止重复提交，不得只依赖 React state 闭包。

### 存储与知识沉淀

- SQLite 使用 WAL、busy timeout 和有限事务范围；不要在持有写锁时调用 LLM、Embedding 或 Rerank。
- 会话最终响应与 trace 快照一起持久化，历史读取不能依赖当前配置重新解释旧请求。
- 沉淀先做向量去重，再做质量初筛；初筛服务不可用时降级为人工审核，不能阻止用户标记。
- `marked_by` 与 `reviewed_by` 语义不同；自动批准必须明确标记系统审核者。

## 验证层级

1. 离线单元/契约测试：不访问真实 LLM、Qdrant 网络或云服务。
2. 真实服务联调：验证模型输出、SSE 时序、工具审计和端到端持久化。
3. 并发与故障注入：验证 SQLite 锁、断连、超时、重试和降级。
4. 量化评测：检索、路由、生成和引用指标，结果写入 `docs/评测与失败案例.md`。

不得用一层通过替代另一层。自动化测试必须 100% 通过才能在验收文档标记完成；历史测试数和历史指标不能当作当前结论。

## 文档维护

- 新架构决策更新 `docs/架构设计.md`；开发命令或不变量更新本文件。
- 指标和失败案例只更新 `docs/评测与失败案例.md`；README 仅同步一行摘要和链接。
- 数据来源只更新 `docs/数据来源.md`；发布状态只更新 `项目验收标准.md`。
- 注释优先解释不变量本身；需要外链时引用文档名称和概念，不引用易漂移的章节号。
