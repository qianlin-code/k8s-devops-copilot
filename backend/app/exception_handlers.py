import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Environment, get_settings
from app.errors import AppError, ErrorCode
from app.schemas.common import ErrorResponse
from app.tracing.context import get_trace_id
from app.tracing.logger import get_logger

logger = get_logger(__name__)


def _emit(
    code: ErrorCode | str,
    message: str,
    http_status: int,
    retryable: bool,
    details: dict | None = None,
) -> JSONResponse:
    trace_id = get_trace_id()
    body = ErrorResponse(
        code=code.value if isinstance(code, ErrorCode) else str(code),
        message=message,
        trace_id=trace_id,
        retryable=retryable,
        details=details or {},
    )
    return JSONResponse(
        status_code=http_status, content=body.model_dump(mode="json")
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "app_error",
            extra={"extra_fields": {"code": exc.code.value, "details": exc.details}},
        )
        return _emit(exc.code, exc.message, exc.http_status, exc.retryable, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # extra="forbid" 触发的字段漂移会走到这里，details 保留字段定位信息
        return _emit(
            ErrorCode.VALIDATION_FAILED,
            "Request payload does not match the API contract",
            422,
            False,
            {"violations": exc.errors()[:10]},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = (
            ErrorCode.RESOURCE_NOT_FOUND
            if exc.status_code == 404
            else ErrorCode.INTERNAL_ERROR
        )
        return _emit(code, str(exc.detail), exc.status_code, False)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # 堆栈只进日志，绝不出现在响应体里
        logger.error("unhandled_exception", exc_info=exc)
        try:
            is_dev = get_settings().environment is Environment.DEV
        except Exception:
            # 配置本身损坏时不能让处理器再炸一次，退化成不暴露细节
            is_dev = False
        details = {"exception_type": type(exc).__name__} if is_dev else {}
        return _emit(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected internal error occurred",
            500,
            False,
            details,
        )
