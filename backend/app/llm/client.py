import json
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.errors import NonRetryableError, RetryableError
from app.llm.error_mapping import map_chat_error

T = TypeVar("T", bound=BaseModel)

# 只对外部依赖的瞬时故障重试；参数/业务错误立即抛出
_retry_policy = retry(
    retry=retry_if_exception_type(RetryableError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


_wrap_openai_error = map_chat_error


class LLMClient:
    """Ollama 与千问都走 OpenAI 兼容协议，切换只是换 base_url/api_key/model。"""

    def __init__(
        self, *, base_url: str, api_key: str, model: str, timeout: float | None = None
    ):
        if timeout is None:
            timeout = get_settings().llm_timeout_seconds
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        # 累计用量：给评估脚本这类需要估算云端调用成本的场景用，
        # 不影响 chat()/structured() 的返回签名，其他调用点无需感知。
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    @_retry_policy
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc
        if resp.usage is not None:
            self.total_prompt_tokens += resp.usage.prompt_tokens
            self.total_completion_tokens += resp.usage.completion_tokens
        self.total_calls += 1
        return resp.choices[0].message.content or ""

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_repairs: int = 1,
    ) -> T:
        """强制 JSON 输出并用 Pydantic 校验，失败后带错误信息重试一次。"""
        convo = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Respond with a single JSON object only, no markdown fences, "
                    f"matching this JSON schema: {json.dumps(schema.model_json_schema())}"
                ),
            },
        ]
        last_error = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(convo, temperature=temperature)
            try:
                return schema.model_validate_json(_strip_fences(raw))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:400]
                if attempt >= max_repairs:
                    break
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw[:1000]},
                    {
                        "role": "user",
                        "content": (
                            f"That output failed schema validation: {last_error}. "
                            "Return corrected JSON only."
                        ),
                    },
                ]
        raise NonRetryableError(
            "LLM failed to produce schema-valid structured output",
            details={"validation_error": last_error},
        )


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        cleaned = cleaned.removeprefix("json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned
