from functools import lru_cache

from app.agent.answerer import Answerer
from app.agent.context_manager import ConversationContextManager
from app.agent.router import Router
from app.agent.state_machine import AgentStateMachine
from app.agent.sufficiency import SufficiencyChecker
from app.agent.tools.cache import ToolResultCache
from app.agent.tools.executor import ToolExecutor
from app.agent.tools.registry import get_tool_registry
from app.config import get_settings
from app.knowledge.ingest import KnowledgeIngestor
from app.knowledge.sedimentation import SedimentationService
from app.llm.factory import get_embedding_client, get_llm_client
from app.rag.bm25_index import get_bm25_index
from app.rag.reranker import get_reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import get_vector_store


@lru_cache
def get_tool_cache() -> ToolResultCache:
    return ToolResultCache(get_settings().tool_cache_ttl_seconds)


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        vector_store=get_vector_store(),
        embedding_client=get_embedding_client(),
        bm25_index=get_bm25_index(),
        reranker=get_reranker(),
        llm_client=get_llm_client(),
    )


@lru_cache
def get_context_manager() -> ConversationContextManager:
    return ConversationContextManager(get_llm_client())


@lru_cache
def get_agent() -> AgentStateMachine:
    llm = get_llm_client()
    registry = get_tool_registry()
    return AgentStateMachine(
        router=Router(llm),
        checker=SufficiencyChecker(llm),
        answerer=Answerer(llm),
        executor=ToolExecutor(registry, get_tool_cache()),
        registry=registry,
    )


@lru_cache
def get_ingestor() -> KnowledgeIngestor:
    return KnowledgeIngestor(
        vector_store=get_vector_store(),
        embedding_client=get_embedding_client(),
        bm25_index=get_bm25_index(),
    )


@lru_cache
def get_sedimentation_service() -> SedimentationService:
    return SedimentationService(get_ingestor())


def reset_dependencies() -> None:
    """测试或配置变更后重建全部单例。测试里部分工厂会被替身覆盖，故容忍无缓存。"""
    for fn in (
        get_tool_cache,
        get_retriever,
        get_context_manager,
        get_agent,
        get_ingestor,
        get_sedimentation_service,
    ):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
