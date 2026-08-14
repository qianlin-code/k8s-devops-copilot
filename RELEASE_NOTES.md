# K8s / DevOps 智能运维 Copilot v1.0.0

v1.0.0 是首个可复现的作品集发布版，覆盖 Kubernetes 知识问答、演示状态查询、受控写操作和知识沉淀闭环。

## 主要能力

- 查询改写、BGE-M3 向量与 BM25 双路召回、RRF、BGE Rerank 和阈值过滤组成的 RAG 链路。
- 直接回答、只读工具、写操作确认和信息不足四类 Agent 路由。
- JWT 用户隔离、admin 知识审核、写操作幂等与审计记录。
- React 前端展示 SSE 进度、引用、Trace、历史会话和待审知识。
- Docker Compose 启动前后端，并持久化 SQLite、Qdrant 和模型缓存。

## 验证摘要

- Python 3.11 离线全量与契约测试通过，前端 typecheck/build 通过。
- 固定 39 条生成与路由门禁完成；固定 30 条检索题与 17 条 hard case 通过修正后的八项发布门禁。
- 独立干净克隆已验证 Docker 主链路、Compose 前端交互、重启持久化和知识沉淀语义。
- 发布前源码、运行证据和 Git 历史完成脱敏扫描。

## 快速开始

请按 [README](README.md) 配置 `backend/.env`，然后运行：

```bash
docker compose up --build
```

演示用户和知识库初始化需要显式传入运行时密码，详见 README 的 Docker 演示章节。

## 已知限制

- SQLite 与 embedded Qdrant 面向单机演示，不是高可用生产架构。
- Kubernetes 工具操作可复现 mock 数据，不连接真实集群。
- `qwen2.5:7b` 仅作为无付费的本地兼容方案；发布质量基线使用 DeepSeek V4 Pro。
- DeepSeek 需要用户自行配置密钥，项目不提供或保存共享凭据。
- v1.0.0 没有公网托管实例，也不声称已通过生产集群验证。
