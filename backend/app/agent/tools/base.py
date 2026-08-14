from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from sqlalchemy.orm import Session

from app.errors import ErrorCode, ToolError

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)
NonEmptyToolString = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class ToolArgs(BaseModel):
    """工具入参基类。extra=forbid 让 LLM 幻觉出的多余字段直接失败而非静默忽略。"""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ToolContext:
    """工具执行所需的运行时依赖。"""

    def __init__(
        self,
        *,
        session: Session,
        trace_id: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> None:
        self.session = session
        self.trace_id = trace_id
        self.user_id = user_id
        self.conversation_id = conversation_id


class Tool(ABC, Generic[TArgs, TResult]):
    name: ClassVar[str]
    description: ClassVar[str]
    is_write: ClassVar[bool] = False
    args_schema: ClassVar[type[BaseModel]]
    # 只读工具可缓存；写工具永不缓存，靠 request_id 幂等
    cacheable: ClassVar[bool] = False
    # 这些字段不能由 Router 自行补写，必须能在当前问题或用户历史中定位。
    # namespace 对所有声明该字段的工具默认强制校验。
    user_grounded_fields: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(self, args: TArgs, ctx: ToolContext) -> TResult:
        ...

    def parse_args(self, raw: dict[str, Any]) -> TArgs:
        try:
            return self.args_schema.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as exc:
            raise ToolError(
                f"Invalid arguments for tool '{self.name}'",
                code=ErrorCode.TOOL_ARGS_INVALID,
                details={
                    "tool": self.name,
                    "violations": [
                        {"field": ".".join(str(p) for p in e["loc"]), "error": e["msg"]}
                        for e in exc.errors()[:8]
                    ],
                },
            ) from exc

    def spec(self) -> dict[str, Any]:
        """给 Router 看的工具描述。"""
        return {
            "name": self.name,
            "description": self.description,
            "is_write": self.is_write,
            "parameters": self.args_schema.model_json_schema(),
        }


class WriteToolArgs(ToolArgs):
    """写操作强制携带 request_id，用于幂等去重。"""

    request_id: str
    # 子类可覆盖：决定"是否是同一个操作目标"的字段名。默认 None 表示用全部
    # 字段（除 request_id）参与去重签名。像 RestartDeploymentArgs.reason 这种
    # 解释性自由文本应该被排除在外——否则 LLM 每轮换一种说法描述原因，
    # 签名就会跟着变，导致同一个目标被误判成"新操作"，重复弹确认卡片；
    # 反过来若粗暴地按工具名整体去重（不看参数），又会把 restart_deployment(A)
    # 和 restart_deployment(B) 这两个不同目标误判成同一个操作，静默跳过 B。
    idempotency_fields: ClassVar[Optional[tuple[str, ...]]] = None


def assert_write_contract(tool: Tool) -> None:
    """启动期自检：写工具必须禁缓存且入参带 request_id。"""
    if not tool.is_write:
        return
    if tool.cacheable:
        raise ToolError(
            f"Write tool '{tool.name}' must not be cacheable",
            code=ErrorCode.BUSINESS_RULE_VIOLATION,
        )
    if "request_id" not in tool.args_schema.model_fields:
        raise ToolError(
            f"Write tool '{tool.name}' must declare request_id for idempotency",
            code=ErrorCode.BUSINESS_RULE_VIOLATION,
        )
