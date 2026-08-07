import asyncio
import json
import logging
from collections.abc import AsyncIterator
from concurrent.futures import Future

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.dependencies import get_agent, get_context_manager, get_retriever
from app.errors import AppError, ErrorCode
from app.schemas.chat import ChatRequest, ChatResponse, ConfirmWriteRequest
from app.schemas.common import ERROR_RESPONSES
from app.schemas.progress import ChatStreamEnvelope, StreamErrorEvent
from app.services.chat_service import ChatService
from app.storage.db import get_db, session_scope
from app.tracing.context import get_trace_id

logger = logging.getLogger(__name__)

# 无新事件时的心跳间隔。取值需明显小于常见代理的读超时（nginx 默认 60s）。
_HEARTBEAT_SECONDS = 10.0

router = APIRouter(
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
    responses=ERROR_RESPONSES,
)


def _log_worker_failure(trace_id: str):
    """给后台线程的 Future 挂回调，避免异常被静默吞掉。

    run_blocking 内部已捕获所有异常并转成 error 事件，走到这里说明是
    捕获逻辑本身出了问题，属于需要暴露的 bug。
    """

    def _callback(future: "Future[None]") -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.error(
                "chat_stream_worker_crashed",
                exc_info=exc,
                extra={"extra_fields": {"trace_id": trace_id}},
            )

    return _callback


def _sse(event: str, data: object) -> str:
    """按 SSE 规范序列化一个事件。

    data 必须是单行 —— JSON 里的换行会被转义成 \\n，不会破坏协议。
    """
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _service() -> ChatService:
    return ChatService(
        retriever=get_retriever(),
        agent=get_agent(),
        context_manager=get_context_manager(),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="提问，返回回答与完整执行链路",
)
async def chat(
    payload: ChatRequest, session: Session = Depends(get_db)
) -> ChatResponse:
    return _service().ask(
        session,
        question=payload.question,
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        trace_id=get_trace_id(),
        include_trace=payload.include_trace,
    )


@router.post(
    "/chat/stream",
    summary="提问，以 SSE 流式推送阶段进展与最终结果",
    response_class=StreamingResponse,
    responses={
        200: {
            # 声明 model 让三类载荷进入 OpenAPI，前端类型才能自动生成
            "model": ChatStreamEnvelope,
            "content": {"text/event-stream": {}},
            "description": (
                "SSE 事件流。event 取值 progress | done | error，"
                "对应载荷见 ChatStreamEnvelope 的同名字段。"
            ),
        },
        **ERROR_RESPONSES,
    },
)
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """流式版 /chat。

    本地 7B 模型一轮对话串联 4 次 LLM 调用（改写/路由/校验/生成），耗时 20-40s。
    非流式接口下前端只能干等，会被误判为卡死并重复提交。

    实现要点：Agent 全链路是同步阻塞代码（SQLAlchemy + openai 同步客户端），
    放进线程池跑，用 call_soon_threadsafe 把阶段事件送回事件循环。
    """
    trace_id = get_trace_id()
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push(kind: str, data: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def run_blocking() -> None:
        # 线程内独立开 session：Session 非线程安全，不能复用请求线程注入的那个
        try:
            with session_scope() as session:
                response = _service().ask_streaming(
                    session,
                    question=payload.question,
                    user_id=payload.user_id,
                    conversation_id=payload.conversation_id,
                    trace_id=trace_id,
                    include_trace=payload.include_trace,
                    emit=lambda event: push("progress", event),
                )
            push("done", response)
        except AppError as exc:
            logger.warning(
                "chat_stream_app_error",
                extra={"extra_fields": {"code": exc.code.value, "trace_id": trace_id}},
            )
            push(
                "error",
                StreamErrorEvent(
                    code=exc.code.value,
                    message=exc.message,
                    trace_id=trace_id,
                    retryable=exc.retryable,
                    details=exc.details or {},
                    http_status=exc.http_status,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # 堆栈只进日志，不进响应体
            logger.error("chat_stream_unhandled", exc_info=exc)
            push(
                "error",
                StreamErrorEvent(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message="An unexpected internal error occurred",
                    trace_id=trace_id,
                    retryable=False,
                    details={},
                    http_status=500,
                ),
            )
        finally:
            push("eof", None)

    async def event_source() -> AsyncIterator[str]:
        future = loop.run_in_executor(None, run_blocking)
        # 不 await：客户端断连后仍让它跑完并落库。但要挂回调，
        # 否则线程内的意外异常会被 Future 静默吞掉。
        future.add_done_callback(_log_worker_failure(trace_id))
        while True:
            try:
                kind, data = await asyncio.wait_for(
                    queue.get(), timeout=_HEARTBEAT_SECONDS
                )
            except TimeoutError:
                # 单个 LLM 调用可能十几秒不产生新阶段。期间必须发心跳，
                # 否则空闲连接会被代理/客户端按读超时回收。
                # SSE 注释帧（以 ':' 开头）会被规范要求的解析器忽略。
                yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    logger.info(
                        "chat_stream_client_disconnected",
                        extra={"extra_fields": {"trace_id": trace_id}},
                    )
                    break
                continue

            if kind == "eof":
                break
            # 先发再判断断连：反过来会在连接尚未完全建立时误判，丢掉首个事件
            yield _sse(kind, data)
            if kind in ("done", "error"):
                break
            if await request.is_disconnected():
                # 客户端已断开就停止推送；后台线程仍会跑完并落库，
                # 用户重开历史记录能看到这轮结果
                logger.info(
                    "chat_stream_client_disconnected",
                    extra={"extra_fields": {"trace_id": trace_id}},
                )
                break

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲，否则 SSE 会被攒着一起发
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/chat/confirm",
    response_model=ChatResponse,
    summary="确认或拒绝执行写操作",
)
async def confirm_write(
    payload: ConfirmWriteRequest, session: Session = Depends(get_db)
) -> ChatResponse:
    return _service().confirm_write(
        session,
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
        confirmation_token=payload.confirmation_token,
        approved=payload.approved,
        trace_id=get_trace_id(),
        include_trace=payload.include_trace,
    )
