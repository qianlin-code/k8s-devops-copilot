import json
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Provider, get_settings
from app.errors import NonRetryableError, RetryableError
from app.llm.error_mapping import map_chat_error

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LLMCostPolicy:
    max_cost_cny: float
    price_per_1k_tokens_cny: float | None = None
    prompt_cache_miss_price_per_1k_cny: float | None = None
    prompt_cache_hit_price_per_1k_cny: float | None = None
    completion_price_per_1k_cny: float | None = None
    structured_max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_cost_cny < 0:
            raise ValueError("max_cost_cny must be non-negative")
        fallback = self.price_per_1k_tokens_cny
        prices = {
            "prompt_cache_miss_price_per_1k_cny": self.prompt_cache_miss_price_per_1k_cny,
            "prompt_cache_hit_price_per_1k_cny": self.prompt_cache_hit_price_per_1k_cny,
            "completion_price_per_1k_cny": self.completion_price_per_1k_cny,
        }
        for name, value in prices.items():
            resolved = fallback if value is None else value
            if resolved is None or resolved <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, resolved)
        if self.structured_max_tokens <= 0:
            raise ValueError("structured_max_tokens must be positive")

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
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider: Provider | None = None,
        timeout: float | None = None,
        cost_policy: LLMCostPolicy | None = None,
    ):
        if timeout is None:
            timeout = get_settings().llm_timeout_seconds
        self.model = model
        self.provider = provider
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        # 累计用量：给评估脚本这类需要估算云端调用成本的场景用，
        # 不影响 chat()/structured() 的返回签名，其他调用点无需感知。
        self.total_prompt_tokens = 0
        self.total_cached_prompt_tokens = 0
        self.total_uncached_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_request_attempts = 0
        self.cost_policy = cost_policy
        self.total_accounted_tokens = 0
        self._accounted_cost_cny = 0.0
        self.usage_estimated = False

    @property
    def estimated_cost_cny(self) -> float | None:
        if self.cost_policy is None:
            return None
        return self._accounted_cost_cny

    def _cost_cny(
        self, *, uncached_prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int
    ) -> float:
        if self.cost_policy is None:
            return 0.0
        return (
            uncached_prompt_tokens
            * self.cost_policy.prompt_cache_miss_price_per_1k_cny
            + cached_prompt_tokens
            * self.cost_policy.prompt_cache_hit_price_per_1k_cny
            + completion_tokens * self.cost_policy.completion_price_per_1k_cny
        ) / 1000

    def _check_cost_budget(
        self, messages: list[dict[str, str]], max_tokens: int | None
    ) -> tuple[int, float]:
        if self.cost_policy is None:
            return 0, 0.0
        if max_tokens is None:
            raise NonRetryableError(
                "A bounded max_tokens value is required for budgeted LLM calls",
                details={"reason": "unbounded_output"},
            )
        # Conservative upper bound: count every UTF-8-independent character as one
        # token, then reserve the full output limit. This intentionally overestimates
        # Chinese prompts and avoids relying on a provider-specific tokenizer.
        prompt_upper_bound = len(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        )
        request_tokens = prompt_upper_bound + max_tokens
        request_cost = self._cost_cny(
            uncached_prompt_tokens=prompt_upper_bound,
            cached_prompt_tokens=0,
            completion_tokens=max_tokens,
        )
        upper_bound_cost = self._accounted_cost_cny + request_cost
        if upper_bound_cost > self.cost_policy.max_cost_cny:
            raise NonRetryableError(
                "LLM generation cost budget would be exceeded",
                details={
                    "reason": "cost_budget_exceeded",
                    "model": self.model,
                    "used_tokens": self.total_accounted_tokens,
                    "next_request_tokens_upper_bound": request_tokens,
                    "cost_upper_bound_cny": round(upper_bound_cost, 6),
                    "max_cost_cny": self.cost_policy.max_cost_cny,
                },
            )
        return request_tokens, request_cost

    @_retry_policy
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_output: bool = False,
    ) -> str:
        request_upper_bound, request_cost_upper_bound = self._check_cost_budget(
            messages, max_tokens
        )
        if self.cost_policy is not None:
            self.total_accounted_tokens += request_upper_bound
            self._accounted_cost_cny += request_cost_upper_bound
        try:
            self.total_request_attempts += 1
            request_options: dict[str, Any] = {}
            if self.provider is Provider.DEEPSEEK:
                request_options["extra_body"] = {"thinking": {"type": "disabled"}}
                if json_output:
                    request_options["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **request_options,
            )
        except Exception as exc:
            if self.cost_policy is not None:
                self.usage_estimated = True
            raise _wrap_openai_error(exc) from exc
        if resp.usage is not None:
            prompt_tokens = int(resp.usage.prompt_tokens)
            completion_tokens = int(resp.usage.completion_tokens)
            cached_tokens = _cached_prompt_tokens(resp.usage, prompt_tokens)
            uncached_tokens = prompt_tokens - cached_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_cached_prompt_tokens += cached_tokens
            self.total_uncached_prompt_tokens += uncached_tokens
            self.total_completion_tokens += completion_tokens
            if self.cost_policy is not None:
                self.total_accounted_tokens -= request_upper_bound
                self.total_accounted_tokens += prompt_tokens + completion_tokens
                self._accounted_cost_cny -= request_cost_upper_bound
                self._accounted_cost_cny += self._cost_cny(
                    uncached_prompt_tokens=uncached_tokens,
                    cached_prompt_tokens=cached_tokens,
                    completion_tokens=completion_tokens,
                )
        elif self.cost_policy is not None:
            self.usage_estimated = True
        self.total_calls += 1
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise NonRetryableError(
                "LLM output was truncated at the configured token limit",
                details={"reason": "output_truncated"},
            )
        content = choice.message.content or ""
        if not content.strip():
            raise NonRetryableError(
                "LLM returned an empty response",
                details={"reason": "empty_output"},
            )
        return content

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_repairs: int = 1,
        max_tokens: int | None = None,
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
        if max_tokens is None and self.cost_policy is not None:
            max_tokens = self.cost_policy.structured_max_tokens
        last_error = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(
                convo,
                temperature=temperature,
                max_tokens=max_tokens,
                json_output=self.provider is Provider.DEEPSEEK,
            )
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


def _cached_prompt_tokens(usage: Any, prompt_tokens: int) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    if cached is None:
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is None:
        return 0
    try:
        return min(max(int(cached), 0), prompt_tokens)
    except (TypeError, ValueError):
        return 0
