import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Provider
from app.errors import NonRetryableError
from app.llm.client import LLMClient, LLMCostPolicy


class _Completions:
    def __init__(self, usage=None, *, content: str | None = None, finish_reason: str = "stop") -> None:
        self.calls: list[dict] = []
        self.usage = usage or SimpleNamespace(prompt_tokens=12, completion_tokens=4)
        self.content = content if content is not None else json.dumps({"ok": True})
        self.finish_reason = finish_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            usage=self.usage,
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=self.content),
                )
            ],
        )


def _client(policy: LLMCostPolicy) -> tuple[LLMClient, _Completions]:
    client = LLMClient(
        base_url="http://unused.invalid/v1",
        api_key="unused",
        model="qwen-plus",
        cost_policy=policy,
    )
    completions = _Completions()
    client._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_budget_rejects_before_provider_request() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=0, price_per_1k_tokens_cny=0.004)
    )

    with pytest.raises(NonRetryableError) as exc_info:
        client.chat([{"role": "user", "content": "probe"}], max_tokens=1)

    assert exc_info.value.details["reason"] == "cost_budget_exceeded"
    assert completions.calls == []
    assert client.total_calls == 0


def test_budgeted_call_requires_bounded_output() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=1, price_per_1k_tokens_cny=0.004)
    )

    with pytest.raises(NonRetryableError) as exc_info:
        client.chat([{"role": "user", "content": "probe"}])

    assert exc_info.value.details["reason"] == "unbounded_output"
    assert completions.calls == []


def test_usage_and_cost_are_recorded_after_success() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=1, price_per_1k_tokens_cny=0.004)
    )

    client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert len(completions.calls) == 1
    assert completions.calls[0]["max_tokens"] == 32
    assert client.total_prompt_tokens == 12
    assert client.total_completion_tokens == 4
    assert client.total_calls == 1
    assert client.estimated_cost_cny == pytest.approx(0.000064)


def test_missing_provider_usage_is_charged_at_conservative_upper_bound() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=1, price_per_1k_tokens_cny=0.004)
    )
    original_create = completions.create

    def without_usage(**kwargs):
        response = original_create(**kwargs)
        response.usage = None
        return response

    completions.create = without_usage
    messages = [{"role": "user", "content": "probe"}]
    expected = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"))) + 32

    client.chat(messages, max_tokens=32)

    assert client.usage_estimated is True
    assert client.total_accounted_tokens == expected
    assert client.estimated_cost_cny == pytest.approx(expected / 1000 * 0.004)


def test_asymmetric_prices_account_for_cached_prompt_tokens() -> None:
    policy = LLMCostPolicy(
        max_cost_cny=1,
        prompt_cache_miss_price_per_1k_cny=0.003,
        prompt_cache_hit_price_per_1k_cny=0.000025,
        completion_price_per_1k_cny=0.006,
    )
    client, completions = _client(policy)
    completions.usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
    )

    client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert client.total_cached_prompt_tokens == 5
    assert client.total_uncached_prompt_tokens == 7
    assert client.estimated_cost_cny == pytest.approx(
        (7 * 0.003 + 5 * 0.000025 + 4 * 0.006) / 1000
    )


def test_cache_detail_absence_prices_all_prompt_as_cache_miss() -> None:
    policy = LLMCostPolicy(
        max_cost_cny=1,
        prompt_cache_miss_price_per_1k_cny=0.003,
        prompt_cache_hit_price_per_1k_cny=0.000025,
        completion_price_per_1k_cny=0.006,
    )
    client, _ = _client(policy)

    client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert client.total_cached_prompt_tokens == 0
    assert client.total_uncached_prompt_tokens == 12
    assert client.estimated_cost_cny == pytest.approx((12 * 0.003 + 4 * 0.006) / 1000)


def test_deepseek_chat_disables_thinking_without_forcing_json() -> None:
    client = LLMClient(
        base_url="http://unused.invalid/v1",
        api_key="unused",
        model="deepseek-v4-pro",
        provider=Provider.DEEPSEEK,
    )
    completions = _Completions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # noqa: SLF001

    client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "response_format" not in completions.calls[0]


def test_deepseek_structured_enables_json_output() -> None:
    client = LLMClient(
        base_url="http://unused.invalid/v1",
        api_key="unused",
        model="deepseek-v4-pro",
        provider=Provider.DEEPSEEK,
    )
    completions = _Completions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # noqa: SLF001

    result = client.structured(
        [{"role": "user", "content": "return json"}],
        _OkSchema,
        max_repairs=0,
        max_tokens=32,
    )

    assert result.ok is True
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_other_providers_do_not_receive_deepseek_request_options() -> None:
    client = LLMClient(
        base_url="http://unused.invalid/v1",
        api_key="unused",
        model="qwen-plus",
        provider=Provider.QWEN,
    )
    completions = _Completions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # noqa: SLF001

    client.structured(
        [{"role": "user", "content": "return json"}],
        _OkSchema,
        max_repairs=0,
        max_tokens=32,
    )

    assert "response_format" not in completions.calls[0]
    assert "extra_body" not in completions.calls[0]


def test_truncated_response_is_rejected_after_usage_is_accounted() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=1, price_per_1k_tokens_cny=0.004)
    )
    completions.finish_reason = "length"
    completions.content = '{"ok":'

    with pytest.raises(NonRetryableError) as exc_info:
        client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert exc_info.value.details["reason"] == "output_truncated"
    assert client.total_calls == 1
    assert client.total_prompt_tokens == 12
    assert client.total_completion_tokens == 4


def test_empty_response_is_rejected_after_usage_is_accounted() -> None:
    client, completions = _client(
        LLMCostPolicy(max_cost_cny=1, price_per_1k_tokens_cny=0.004)
    )
    completions.content = "   "

    with pytest.raises(NonRetryableError) as exc_info:
        client.chat([{"role": "user", "content": "probe"}], max_tokens=32)

    assert exc_info.value.details["reason"] == "empty_output"
    assert client.total_calls == 1


class _OkSchema(BaseModel):
    ok: bool
