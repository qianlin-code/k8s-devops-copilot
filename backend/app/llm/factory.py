from functools import lru_cache

from app.config import Provider, Settings, get_settings
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient


def _chat_credentials(settings: Settings) -> tuple[str, str, str]:
    if settings.llm_provider is Provider.QWEN:
        return settings.qwen_base_url, settings.qwen_api_key, settings.qwen_chat_model
    return settings.ollama_base_url, settings.ollama_api_key, settings.ollama_chat_model


def _embedding_credentials(settings: Settings) -> tuple[str, str, str, int]:
    if settings.embedding_provider is Provider.QWEN:
        return (
            settings.qwen_base_url,
            settings.qwen_api_key,
            settings.qwen_embedding_model,
            settings.qwen_embedding_dim,
        )
    return (
        settings.ollama_base_url,
        settings.ollama_api_key,
        settings.ollama_embedding_model,
        settings.ollama_embedding_dim,
    )


@lru_cache
def get_llm_client() -> LLMClient:
    settings = get_settings()
    base_url, api_key, model = _chat_credentials(settings)
    return LLMClient(base_url=base_url, api_key=api_key, model=model)


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    base_url, api_key, model, dim = _embedding_credentials(settings)
    return EmbeddingClient(base_url=base_url, api_key=api_key, model=model, dim=dim)


def reset_clients() -> None:
    """测试或配置热切换后需要重建客户端。测试里工厂可能被替身覆盖，故容忍无缓存。"""
    for fn in (get_llm_client, get_embedding_client):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
