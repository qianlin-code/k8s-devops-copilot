from functools import lru_cache

from app.config import Provider, Settings, get_settings
from app.errors import ErrorCode, NonRetryableError
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


@lru_cache
def get_judge_client() -> LLMClient:
    """RAGAS 风格评估的裁判客户端：始终走百炼云端，不管 LLM_PROVIDER 是什么。

    裁判必须独立于被测链路，否则本地小模型评自己生成的答案没有意义。
    调用前需要 QWEN_API_KEY，缺失时在此处抛错而非启动时校验，
    因为不跑评估脚本的用户不该被这个要求挡住。
    """
    settings = get_settings()
    if not settings.qwen_api_key:
        raise NonRetryableError(
            "QWEN_API_KEY is required to run judge/evaluation calls",
            code=ErrorCode.VALIDATION_FAILED,
        )
    return LLMClient(
        base_url=settings.qwen_base_url,
        api_key=settings.qwen_api_key,
        model=settings.qwen_judge_model,
    )


@lru_cache
def get_sedimentation_client() -> LLMClient:
    """沉淀质量初筛客户端：云端小模型，成本优先于裁判级准确度。"""
    settings = get_settings()
    if not settings.qwen_api_key:
        raise NonRetryableError(
            "QWEN_API_KEY is required for sedimentation quality screening",
            code=ErrorCode.VALIDATION_FAILED,
        )
    return LLMClient(
        base_url=settings.qwen_base_url,
        api_key=settings.qwen_api_key,
        model=settings.qwen_sedimentation_model,
    )


def reset_clients() -> None:
    """测试或配置热切换后需要重建客户端。测试里工厂可能被替身覆盖，故容忍无缓存。"""
    for fn in (
        get_llm_client,
        get_embedding_client,
        get_judge_client,
        get_sedimentation_client,
    ):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
