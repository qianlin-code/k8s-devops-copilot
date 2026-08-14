import pytest

from app.config import Settings
from app.llm.factory import _chat_credentials, get_llm_client, reset_clients


def test_deepseek_requires_api_key() -> None:
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        Settings(
            llm_provider="deepseek",
            deepseek_api_key="",
            embedding_provider="ollama",
        )


def test_deepseek_chat_credentials_use_official_endpoint() -> None:
    settings = Settings(
        llm_provider="deepseek",
        deepseek_api_key="test-only-key",
        embedding_provider="ollama",
    )

    assert _chat_credentials(settings) == (
        "https://api.deepseek.com",
        "test-only-key",
        "deepseek-v4-pro",
    )


def test_deepseek_cannot_be_selected_for_embedding() -> None:
    with pytest.raises(ValueError, match="does not support deepseek"):
        Settings(
            llm_provider="ollama",
            embedding_provider="deepseek",
        )


def test_factory_preserves_deepseek_provider_identity(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    from app.config import get_settings

    get_settings.cache_clear()
    reset_clients()
    try:
        assert get_llm_client().provider.value == "deepseek"
    finally:
        get_settings.cache_clear()
        reset_clients()
