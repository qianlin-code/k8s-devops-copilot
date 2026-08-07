from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from app.errors import ErrorCode, ToolError

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)


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
