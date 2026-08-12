from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    # 鉴权 / 输入 (4xx, 不可重试)
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

    # 工具 / 业务 (不可重试)
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
    TOOL_ARGS_INVALID = "TOOL_ARGS_INVALID"
    WRITE_CONFIRMATION_REQUIRED = "WRITE_CONFIRMATION_REQUIRED"
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"

    # 外部依赖 (可重试)
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    RERANKER_UNAVAILABLE = "RERANKER_UNAVAILABLE"

    # Agent
    AGENT_MAX_STEPS_EXCEEDED = "AGENT_MAX_STEPS_EXCEEDED"

    # 兜底
    INTERNAL_ERROR = "INTERNAL_ERROR"
    STARTUP_CHECK_FAILED = "STARTUP_CHECK_FAILED"


class AppError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.details = details or {}


class RetryableError(AppError):
    """外部依赖临时故障，tenacity 会对其重试。"""

    retryable = True
    http_status = 503
    code = ErrorCode.LLM_UNAVAILABLE


class NonRetryableError(AppError):
    """参数、权限、业务逻辑错误，重试无意义。"""

    retryable = False
    http_status = 400
    code = ErrorCode.VALIDATION_FAILED


class UnauthorizedError(NonRetryableError):
    code = ErrorCode.UNAUTHORIZED
    http_status = 401


class ForbiddenError(NonRetryableError):
    code = ErrorCode.FORBIDDEN
    http_status = 403


class NotFoundError(NonRetryableError):
    code = ErrorCode.RESOURCE_NOT_FOUND
    http_status = 404


class InputGuardError(NonRetryableError):
    http_status = 422


class ToolError(NonRetryableError):
    code = ErrorCode.TOOL_ARGS_INVALID


class ToolPermissionDeniedError(ToolError):
    code = ErrorCode.TOOL_PERMISSION_DENIED
    http_status = 403


class LLMTimeoutError(RetryableError):
    code = ErrorCode.LLM_TIMEOUT
    http_status = 504


class LLMUnavailableError(RetryableError):
    code = ErrorCode.LLM_UNAVAILABLE


class EmbeddingUnavailableError(RetryableError):
    code = ErrorCode.EMBEDDING_UNAVAILABLE


class VectorStoreUnavailableError(RetryableError):
    code = ErrorCode.VECTOR_STORE_UNAVAILABLE


class RerankerUnavailableError(RetryableError):
    code = ErrorCode.RERANKER_UNAVAILABLE


class StartupCheckError(AppError):
    code = ErrorCode.STARTUP_CHECK_FAILED
