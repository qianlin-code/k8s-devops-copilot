"""异常分级测试：可重试与不可重试的边界必须准确，否则要么白重试要么误放弃。"""

import httpx
import openai
import pytest

from app.errors import (
    EmbeddingUnavailableError,
    LLMTimeoutError,
    LLMUnavailableError,
    NonRetryableError,
    RetryableError,
)
from app.llm.error_mapping import map_chat_error, map_embedding_error

_REQUEST = httpx.Request("POST", "http://localhost:11434/v1/embeddings")


def _bad_request(message: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=_REQUEST, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body=None)


# Ollama 把后端断连包装成 HTTP 400，这类必须判为可重试
TRANSIENT_400 = [
    'do embedding request: Post "http://127.0.0.1:61505/v1/embeddings": '
    "read tcp 127.0.0.1:56730->127.0.0.1:61505: wsarecv: "
    "An existing connection was forcibly closed by the remote host.",
    "connection reset by peer",
    "dial tcp 127.0.0.1:11434: connect: connection refused",
    "context deadline exceeded",
    "unexpected EOF",
    "broken pipe",
]

GENUINE_400 = [
    "invalid model name 'does-not-exist'",
    "input must not be empty",
    "context length 8192 exceeded",
]


@pytest.mark.parametrize("message", TRANSIENT_400)
def test_transient_400_is_retryable_for_embedding(message: str) -> None:
    mapped = map_embedding_error(_bad_request(message))
    assert isinstance(mapped, EmbeddingUnavailableError)
    assert isinstance(mapped, RetryableError)
    assert mapped.retryable is True


@pytest.mark.parametrize("message", TRANSIENT_400)
def test_transient_400_is_retryable_for_chat(message: str) -> None:
    mapped = map_chat_error(_bad_request(message))
    assert isinstance(mapped, LLMUnavailableError)
    assert mapped.retryable is True


@pytest.mark.parametrize("message", GENUINE_400)
def test_genuine_400_is_not_retryable(message: str) -> None:
    for mapped in (
        map_embedding_error(_bad_request(message)),
        map_chat_error(_bad_request(message)),
    ):
        assert isinstance(mapped, NonRetryableError)
        assert mapped.retryable is False


def test_auth_error_is_never_retried() -> None:
    response = httpx.Response(401, request=_REQUEST, json={"error": {"message": "bad key"}})
    exc = openai.AuthenticationError("bad key", response=response, body=None)
    for mapped in (map_embedding_error(exc), map_chat_error(exc)):
        assert isinstance(mapped, NonRetryableError)


def test_timeout_is_retryable() -> None:
    exc = openai.APITimeoutError(request=_REQUEST)
    assert isinstance(map_chat_error(exc), LLMTimeoutError)
    assert isinstance(map_embedding_error(exc), EmbeddingUnavailableError)
    assert map_chat_error(exc).retryable is True


def test_connection_error_is_retryable() -> None:
    exc = openai.APIConnectionError(request=_REQUEST)
    assert map_chat_error(exc).retryable is True
    assert map_embedding_error(exc).retryable is True


def test_rate_limit_is_retryable() -> None:
    response = httpx.Response(429, request=_REQUEST, json={"error": {"message": "slow down"}})
    exc = openai.RateLimitError("slow down", response=response, body=None)
    assert map_chat_error(exc).retryable is True
