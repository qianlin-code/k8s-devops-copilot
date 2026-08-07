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
    question: str = Field(min_length=1, description="用户的自然语言问题")
    user_id: str = Field(min_length=1, description="提问用户的账号 ID")
    conversation_id: Optional[str] = Field(
        default=None, description="续接已有会话；留空则新建会话"
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
    conversation_id: str
    user_id: str
    confirmation_token: str = Field(description="来自 pending_write 的令牌")
    approved: bool = Field(description="false 表示用户拒绝执行")
    include_trace: bool = True
