import pytest

from app import startup_checks
from app.errors import NonRetryableError


def test_llm_probe_requests_short_deterministic_reply(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class ProbeLLM:
        model = "probe-model"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            calls.append({"messages": messages, **kwargs})
            return "OK"

    monkeypatch.setattr(startup_checks, "get_llm_client", lambda: ProbeLLM())

    assert startup_checks._check_llm() == "model=probe-model replied=2chars"
    assert calls == [
        {
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
    ]


def test_llm_probe_does_not_waive_truncated_output(monkeypatch) -> None:
    class TruncatedLLM:
        model = "probe-model"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            raise NonRetryableError(
                "LLM output was truncated at the configured token limit",
                details={"reason": "output_truncated"},
            )

    monkeypatch.setattr(startup_checks, "get_llm_client", lambda: TruncatedLLM())

    with pytest.raises(NonRetryableError) as exc_info:
        startup_checks._check_llm()

    assert exc_info.value.details["reason"] == "output_truncated"
