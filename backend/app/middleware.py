import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.tracing.context import new_trace_id, set_trace_id
from app.tracing.logger import get_logger, log_event
import logging

logger = get_logger(__name__)
TRACE_HEADER = "X-Trace-Id"


class TraceMiddleware(BaseHTTPMiddleware):
    """为每个请求分配 trace_id，贯穿日志与响应头。"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        trace_id = request.headers.get(TRACE_HEADER) or new_trace_id()
        set_trace_id(trace_id)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.headers[TRACE_HEADER] = trace_id
        # 流式响应下 call_next 在响应头就绪时即返回，此处记的是首字节耗时
        # 而非全流时长。不标注的话日志会显示 SSE 请求"只用了几毫秒"，误导排查。
        streaming = response.headers.get("content-type", "").startswith(
            "text/event-stream"
        )
        log_event(
            logger,
            logging.INFO,
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
            elapsed_meaning="time_to_first_byte" if streaming else "total",
        )
        return response
