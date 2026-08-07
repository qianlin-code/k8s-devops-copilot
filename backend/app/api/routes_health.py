from fastapi import APIRouter, Depends

from app.config import Environment, Settings, get_settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="存活探针，无需鉴权")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """该端点无需鉴权，因此生产环境只回 status/environment。

    模型 provider 与集合名属于内部拓扑，未鉴权就暴露等于给攻击者做侦察。
    需要这些信息时用 /readiness（要 API Key）。
    """
    if settings.environment is Environment.PROD:
        return HealthResponse(status="ok", environment=settings.environment.value)
    return HealthResponse(
        status="ok",
        environment=settings.environment.value,
        llm_provider=settings.llm_provider.value,
        embedding_provider=settings.embedding_provider.value,
        collection_name=settings.collection_name,
    )
