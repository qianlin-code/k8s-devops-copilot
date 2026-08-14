import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_auth, routes_chat, routes_health, routes_history, routes_kb
from app.auth.dependencies import require_jwt
from app.config import Environment, get_settings
from app.exception_handlers import register_exception_handlers
from app.middleware import TraceMiddleware
from app.rag.reranker import preload_reranker
from app.schemas.common import DependencyCheck, ReadinessResponse
from app.startup_checks import run_startup_checks
from app.tracing.logger import configure_logging, get_logger, log_event

logger = get_logger(__name__)
_readiness: list[DependencyCheck] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    outcomes = run_startup_checks(strict=settings.environment is Environment.PROD)
    _readiness.clear()
    for o in outcomes:
        _readiness.append(
            DependencyCheck(
                name=o.name, ok=o.ok, detail=o.detail, elapsed_ms=o.elapsed_ms
            )
        )
        log_event(
            logger,
            logging.INFO if o.ok else logging.WARNING,
            "startup_check",
            check=o.name,
            ok=o.ok,
            detail=o.detail,
            elapsed_ms=o.elapsed_ms,
        )

    # Rerank 模型冷启动要十几秒（含 HuggingFace 版本校验）。不预热的话
    # 这笔开销会落在第一个用户请求上——实测国内网络下 HF 校验重试让首个
    # 请求多等 300s。后台加载：服务立即可用，模型就绪前检索自动降级为
    # RRF 融合顺序。
    warmup_tasks = [
        asyncio.create_task(_warmup_reranker()),
        asyncio.create_task(_warmup_llm()),
    ]
    try:
        yield
    finally:
        for task in warmup_tasks:
            task.cancel()


async def _warmup_reranker() -> None:
    if not get_settings().warmup_reranker:
        return
    started = time.perf_counter()
    try:
        # 同步阻塞的模型加载放线程，别卡住事件循环
        await asyncio.to_thread(preload_reranker)
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            logging.WARNING,
            "reranker_warmup_failed",
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )
        return
    log_event(
        logger,
        logging.INFO,
        "reranker_warmed_up",
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


async def _warmup_llm() -> None:
    """发一次最小的 chat 调用把模型权重提前加载，避免首个真实用户请求
    背上冷启动开销（本地 Ollama 实测首次对话耗时 115s，预热后降到 1s 量级）。
    """
    if not get_settings().warmup_llm:
        return
    from app.llm.factory import get_llm_client

    started = time.perf_counter()
    try:
        client = get_llm_client()
        # chat() 是同步阻塞调用，放线程里跑，别卡住事件循环
        await asyncio.to_thread(
            client.chat,
            [{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=1,
        )
    except Exception as exc:  # noqa: BLE001
        # 预热失败不影响服务可用性：第一个真实请求会自然触发加载，
        # 只是失去了"提前"的收益，不该因为探测失败就拒绝启动。
        log_event(
            logger,
            logging.WARNING,
            "llm_warmup_failed",
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )
        return
    log_event(
        logger,
        logging.INFO,
        "llm_warmed_up",
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Support Copilot",
        version="1.0.0",
        description=(
            "RAG + Agent closed-loop copilot for enterprise IT support. "
            "Every response carries the full execution trace."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(TraceMiddleware)
    app.add_middleware(
        CORSMiddleware,
        # 生产必须显式配置真实来源；带 localhost 或 '*' 会在启动校验时被拒
        allow_origins=get_settings().cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Trace-Id"],
    )
    register_exception_handlers(app)

    prefix = "/api/v1"
    app.include_router(routes_health.router, prefix=prefix)
    app.include_router(routes_auth.router, prefix=prefix)
    app.include_router(routes_chat.router, prefix=prefix)
    app.include_router(routes_kb.router, prefix=prefix)
    app.include_router(routes_history.router, prefix=prefix)

    @app.get(
        f"{prefix}/readiness",
        response_model=ReadinessResponse,
        tags=["health"],
        summary="依赖可用性明细",
        dependencies=[Depends(require_jwt)],
    )
    async def readiness() -> ReadinessResponse:
        return ReadinessResponse(
            ready=all(c.ok for c in _readiness), checks=list(_readiness)
        )

    return app


app = create_app()
