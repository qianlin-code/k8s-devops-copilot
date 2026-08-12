from typing import Any, Optional

from pydantic import Field

from app.schemas.base import StrictBaseModel
from app.schemas.trace import ExecutionTrace


class PendingWriteActionSchema(StrictBaseModel):
    """需要用户确认的写操作。前端拿 confirmation_token 回传即执行。"""

    tool_name: str
    description: str
    arguments: dict[str, Any]
    reasoning: str
    confirmation_token: str = Field(
        description="确认令牌，回传到 /chat/confirm 执行该操作"
    )


class ChatRequest(StrictBaseModel):
    # question 的长度上限故意不在这里加：业务层 input_guard.py 按
    # settings.max_input_length 校验并返回统一错误码 INPUT_TOO_LONG，
    # 若 schema 层也加 max_length，会被 Pydantic 提前拦截成标准 422 校验错误，
    # 绕过业务层的错误码，破坏契约（见 test_input_too_long_rejected）。
    question: str = Field(min_length=1, description="用户的自然语言问题")
    # user_id 已从 JWT token 的 sub claim 获取，不再接受客户端传入
    conversation_id: Optional[str] = Field(
        default=None,
        max_length=36,
        description="续接已有会话；留空则新建会话",
    )
    include_trace: bool = Field(
        default=True, description="是否在响应里返回完整执行链路"
    )


class ChatResponse(StrictBaseModel):
    conversation_id: str
    message_id: int
    outcome: str = Field(
        description="direct_answer / tool_assisted_answer / write_confirmation_required "
        "/ insufficient_information / max_steps_exceeded"
    )
    answer: str
    pending_write: Optional[PendingWriteActionSchema] = None
    trace: Optional[ExecutionTrace] = None
    created_at: str


class ConfirmWriteRequest(StrictBaseModel):
    conversation_id: str = Field(max_length=36)
    # user_id 已从 JWT token 的 sub claim 获取，不再接受客户端传入
    confirmation_token: str = Field(description="来自 pending_write 的令牌")
    approved: bool = Field(description="false 表示用户拒绝执行")
    include_trace: bool = True
