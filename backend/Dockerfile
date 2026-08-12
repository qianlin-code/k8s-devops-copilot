FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.5.11

# Rerank 会拉入 torch（约 2 GB），默认不装以保持镜像精简。
# 需要时: docker compose build --build-arg WITH_RERANK=true
# 未装时检索链路自动降级为 RRF 融合顺序，功能不中断。
ARG WITH_RERANK=false

# 先只拷锁定依赖声明，让依赖层可被缓存。容器不 editable 安装当前项目，
# 因为应用代码会在后续 COPY 后直接由 uvicorn 导入。
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && if [ "$WITH_RERANK" = "true" ]; then \
         uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu \
           "FlagEmbedding==1.3.3"; \
       fi

# 让容器内的 `python scripts/*.py` 与服务入口都能导入 /app/app。
ENV PYTHONPATH=/app

COPY app ./app
COPY scripts ./scripts
COPY data/docs_k8s ./data/docs_k8s
COPY data/eval_set.json ./data/eval_set.json

# 数据卷挂载点：SQLite 与 Qdrant 存储不进镜像
RUN mkdir -p /app/data/qdrant_storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
