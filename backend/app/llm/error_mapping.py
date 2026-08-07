import re

import openai

from app.errors import (
    EmbeddingUnavailableError,
    LLMTimeoutError,
    LLMUnavailableError,
    NonRetryableError,
)

# Ollama 会把后端进程的传输层故障包装成 HTTP 400 返回，
# 若一律按"参数错误"处理，一次网络抖动就会让整条链路失败。
_TRANSIENT_IN_400 = re.compile(
    r"connection\s+(was\s+)?(forcibly\s+)?(closed|reset|refused|aborted)"
    r"|wsarecv|wsasend|broken pipe|EOF|unexpected end"
    r"|read tcp|write tcp|dial tcp"
    r"|timeout|timed out|deadline exceeded"
    r"|temporarily unavailable|no such host",
    re.IGNORECASE,
)


def is_transient_bad_request(message: str) -> bool:
    return bool(_TRANSIENT_IN_400.search(message))


def map_chat_error(exc: Exception) -> Exception:
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError("LLM request timed out")
    if isinstance(exc, openai.RateLimitError):
        return LLMUnavailableError("LLM rate limit exceeded")
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
        return LLMUnavailableError(f"LLM endpoint unreachable: {type(exc).__name__}")
    if isinstance(exc, openai.AuthenticationError):
        return NonRetryableError("LLM credentials rejected")
    if isinstance(exc, openai.BadRequestError):
        text = str(exc)
        if is_transient_bad_request(text):
            return LLMUnavailableError(f"LLM backend dropped the request: {text[:200]}")
        return NonRetryableError(f"LLM rejected the request: {text[:300]}")
    return LLMUnavailableError(f"Unexpected LLM failure: {type(exc).__name__}")


def map_embedding_error(exc: Exception) -> Exception:
    if isinstance(exc, openai.AuthenticationError):
        return NonRetryableError("Embedding credentials rejected")
    if isinstance(exc, openai.BadRequestError):
        text = str(exc)
        if is_transient_bad_request(text):
            return EmbeddingUnavailableError(
                f"Embedding backend dropped the request: {text[:200]}"
            )
        return NonRetryableError(f"Embedding request rejected: {text[:300]}")
    if isinstance(exc, openai.APITimeoutError):
        return EmbeddingUnavailableError("Embedding request timed out")
    return EmbeddingUnavailableError(
        f"Embedding endpoint failure: {type(exc).__name__}"
    )
