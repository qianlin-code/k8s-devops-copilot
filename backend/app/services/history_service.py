from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.errors import NotFoundError
from app.schemas.base import to_utc_iso
from app.schemas.history import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummary,
    MessageItem,
    ToolAuditItem,
    ToolAuditListResponse,
)
from app.storage.models import Conversation, Message, MessageRole, ToolCallAudit


class HistoryService:
    def list_conversations(
        self, session: Session, *, user_id: str, limit: int, offset: int
    ) -> ConversationListResponse:
        """会话列表按用户隔离。

        `user_id` 必填而非 Optional：可选参数默认「不过滤」会在漏传时静默
        返回全量数据（越权风险见 docs/评测与失败案例.md）。路由层已强制必填，
        这里同步收紧，避免其他调用方直接调 service 时绕过隔离。
        """
        counts = dict(
            session.execute(
                select(Message.conversation_id, func.count(Message.id)).group_by(
                    Message.conversation_id
                )
            ).all()
        )
        stmt = select(Conversation).where(Conversation.user_id == user_id)
        total = session.scalar(
            select(func.count()).select_from(stmt.subquery())
        ) or 0
        rows = list(
            session.scalars(
                stmt.order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)
            )
        )
        return ConversationListResponse(
            total=total,
            conversations=[
                ConversationSummary(
                    conversation_id=c.id,
                    user_id=c.user_id,
                    title=c.title,
                    message_count=counts.get(c.id, 0),
                    has_summary=bool(c.summary),
                    created_at=to_utc_iso(c.created_at),
                    updated_at=to_utc_iso(c.updated_at),
                )
                for c in rows
            ],
        )

    def get_conversation(
        self,
        session: Session,
        conversation_id: str,
        *,
        include_trace: bool,
        user_id: str,
    ) -> ConversationDetailResponse:
        """user_id 必填：不校验归属的话，遍历 conversation_id 就能读到
        任意用户的对话内容与完整执行链路（含工具入参出参）。
        """
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(
                f"Conversation '{conversation_id}' not found",
                details={"conversation_id": conversation_id},
            )
        if conversation.user_id != user_id:
            # 复用 404 而非 403：403 会告诉调用方"这个 id 存在但不属于你"，
            # 等于确认了 id 的有效性，方便枚举。
            raise NotFoundError(
                f"Conversation '{conversation_id}' not found",
                details={"conversation_id": conversation_id},
            )
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            )
        )
        return ConversationDetailResponse(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            summary=conversation.summary,
            created_at=to_utc_iso(conversation.created_at),
            updated_at=to_utc_iso(conversation.updated_at),
            messages=[
                MessageItem(
                    message_id=m.id,
                    role=m.role,
                    content=m.content,
                    trace_id=m.trace_id,
                    trace=m.trace_payload
                    if include_trace and m.role == MessageRole.ASSISTANT.value
                    else None,
                    created_at=to_utc_iso(m.created_at),
                )
                for m in messages
            ],
        )

    def list_tool_audits(
        self,
        session: Session,
        *,
        conversation_id: str | None,
        tool_name: str | None,
        limit: int,
        offset: int,
        user_id: str,
    ) -> ToolAuditListResponse:
        """审计记录按用户隔离。

        工具入参出参可能含账号状态、订单信息，不隔离等于把别人的
        业务数据摊开给任何持有 API Key 的调用方。
        """
        # 用子查询限定到该用户名下的会话，而不是信任传入的 conversation_id
        owned = select(Conversation.id).where(Conversation.user_id == user_id)
        stmt = select(ToolCallAudit).where(ToolCallAudit.conversation_id.in_(owned))
        if conversation_id:
            stmt = stmt.where(ToolCallAudit.conversation_id == conversation_id)
        if tool_name:
            stmt = stmt.where(ToolCallAudit.tool_name == tool_name)
        total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = list(
            session.scalars(
                stmt.order_by(ToolCallAudit.id.desc()).limit(limit).offset(offset)
            )
        )
        return ToolAuditListResponse(
            total=total,
            items=[
                ToolAuditItem(
                    audit_id=a.id,
                    trace_id=a.trace_id,
                    conversation_id=a.conversation_id,
                    request_id=a.request_id,
                    tool_name=a.tool_name,
                    is_write=a.is_write,
                    success=a.success,
                    cache_hit=a.cache_hit,
                    idempotent_replay=a.idempotent_replay,
                    error_code=a.error_code,
                    elapsed_ms=a.elapsed_ms,
                    created_at=to_utc_iso(a.created_at),
                )
                for a in rows
            ],
        )
