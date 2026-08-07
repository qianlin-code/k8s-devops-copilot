from typing import Any, Optional

from pydantic import Field

from app.schemas.base import StrictBaseModel


class MessageItem(StrictBaseModel):
    message_id: int
    role: str = Field(description="user / assistant / system")
    content: str
    trace_id: Optional[str] = None
    trace: Optional[dict[str, Any]] = Field(
        default=None, description="该轮的执行链路快照，仅 assistant 消息有"
    )
    created_at: str


class ConversationSummary(StrictBaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str] = None
    message_count: int
    has_summary: bool
    created_at: str
    updated_at: str


class ConversationListResponse(StrictBaseModel):
    total: int
    conversations: list[ConversationSummary]


class ConversationDetailResponse(StrictBaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    created_at: str
    updated_at: str
    messages: list[MessageItem]


class ToolAuditItem(StrictBaseModel):
    audit_id: int
    trace_id: str
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    tool_name: str
    is_write: bool
    success: bool
    cache_hit: bool
    idempotent_replay: bool
    error_code: Optional[str] = None
    elapsed_ms: int
    created_at: str


class ToolAuditListResponse(StrictBaseModel):
    total: int
    items: list[ToolAuditItem]
