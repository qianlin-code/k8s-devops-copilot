from typing import Any, Literal, Optional

from pydantic import Field

from app.schemas.base import StrictBaseModel

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "DependencyCheck",
    "ReadinessResponse",
    "ERROR_RESPONSES",
]


class ErrorResponse(StrictBaseModel):
    """全局统一错误返回格式，永不包含原生异常堆栈。

    所有非 2xx 响应都是这个形状，OpenAPI 里通过 ERROR_RESPONSES 声明给前端。
    """

    code: str = Field(description="业务错误码，见 ErrorCode 枚举")
    message: str = Field(description="人类可读的错误说明")
    trace_id: str = Field(description="全链路追踪 ID，用于日志关联排查")
    retryable: bool = Field(description="调用方是否可以重试该请求")
    details: dict[str, Any] = Field(
        default_factory=dict, description="附加调试上下文，生产环境可能为空"
    )


class HealthResponse(StrictBaseModel):
    """存活探针。无需鉴权，所以生产环境不返回内部拓扑。

    provider / collection 这些字段只在 dev 下填充（前端侧边栏展示用），
    生产下为 None —— 未鉴权端点不该告诉外部我们用了什么模型和集合名。
    """

    status: Literal["ok"] = "ok"
    environment: str
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    collection_name: Optional[str] = None


class DependencyCheck(StrictBaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None
    elapsed_ms: int


class ReadinessResponse(StrictBaseModel):
    ready: bool
    checks: list[DependencyCheck]


# 挂在路由上，让统一错误格式进入 OpenAPI，前端类型才能生成出来
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "业务规则或参数错误，不可重试"},
    401: {"model": ErrorResponse, "description": "API Key 缺失或错误"},
    403: {"model": ErrorResponse, "description": "越权访问"},
    404: {"model": ErrorResponse, "description": "资源不存在"},
    422: {"model": ErrorResponse, "description": "请求体不符合契约，或触发输入安全拦截"},
    500: {"model": ErrorResponse, "description": "内部错误，响应体不含堆栈"},
    503: {"model": ErrorResponse, "description": "外部依赖暂时不可用，可重试"},
}
