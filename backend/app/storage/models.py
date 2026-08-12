from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IncidentStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SedimentationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


AUTO_QUALITY_REVIEWER = "system:auto-quality"


class Organization(Base):
    """组织/租户。本阶段不做数据物理隔离，只是预留多租户字段。"""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class User(Base):
    """用户账号。password_hash 存 bcrypt 摘要，不存明文。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default=UserRole.USER.value)
    organization_id: Mapped[str] = mapped_column(String(36), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(String(255), default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    trace_payload: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class KBDocument(Base):
    """知识库文档元数据，向量本体存在 Qdrant。"""

    __tablename__ = "kb_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(512), default=None)
    collection_name: Mapped[str] = mapped_column(String(128), index=True)
    chunk_strategy: Mapped[str] = mapped_column(String(32))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Incident(Base):
    """模拟告警事件工单，写工具的落地对象。"""

    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default=IncidentStatus.OPEN.value)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    conversation_id: Mapped[str | None] = mapped_column(String(36), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class MockPod(Base):
    """模拟 Pod 状态，只读工具的查询对象。

    复合主键 (namespace, name)：不同命名空间下可以有同名 Pod，
    这与账号场景里 user_id 全局唯一不同，是 K8s 资源模型的真实约束。
    """

    __tablename__ = "mock_pods"

    namespace: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    phase: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(String(255), default=None)
    # Pending 状态的 Pod 尚未被调度到任何节点，必须允许为空，不能给假节点名
    node_name: Mapped[str | None] = mapped_column(String(128), default=None)
    restart_count: Mapped[int] = mapped_column(Integer, default=0)
    last_transition_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)


class MockDeployment(Base):
    __tablename__ = "mock_deployments"

    namespace: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    image: Mapped[str] = mapped_column(String(255))
    replicas: Mapped[int] = mapped_column(Integer, default=1)
    available_replicas: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ToolCallAudit(Base):
    """全量工具调用审计。写操作靠 request_id 唯一约束实现幂等。"""

    __tablename__ = "tool_call_audits"
    __table_args__ = (
        # 幂等键按会话隔离，不是全局唯一：`request_id` 由 LLM 生成，实测会出现
        # "123456" 这类极易碰撞的值。全局唯一时，B 会话用了 A 会话已用过的
        # request_id 会命中 A 的审计行并重放 A 的结果（跨会话串数据）。
        UniqueConstraint(
            "conversation_id", "request_id", name="uq_tool_call_conversation_request"
        ),
        Index("ix_audit_tool_created", "tool_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), default=None)
    tool_name: Mapped[str] = mapped_column(String(64))
    is_write: Mapped[bool] = mapped_column(Boolean, default=False)
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotent_replay: Mapped[bool] = mapped_column(Boolean, default=False)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PendingSedimentation(Base):
    """半自动知识沉淀：标记后进待审队列，人工确认才入库。"""

    __tablename__ = "pending_sedimentations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    proposed_title: Mapped[str] = mapped_column(String(255))
    marked_by: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(
        String(16), default=SedimentationStatus.PENDING.value, index=True
    )
    review_note: Mapped[str | None] = mapped_column(Text, default=None)
    kb_document_id: Mapped[str | None] = mapped_column(String(36), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    # 自动初筛结果：云端小模型打分 + 向量相似度去重，均在 mark() 时同步产生。
    quality_score: Mapped[float | None] = mapped_column(Float, default=None)
    quality_reasoning: Mapped[str | None] = mapped_column(Text, default=None)
    duplicate_of_document_id: Mapped[str | None] = mapped_column(String(36), default=None)
    duplicate_score: Mapped[float | None] = mapped_column(Float, default=None)
    # 高分且非重复时自动通过，reviewed_by 记 AUTO_QUALITY_REVIEWER；人工审核时记审核人 ID。
    # 与 marked_by（谁标记的原始对话）语义不同，必须分开记录才能复盘"谁批准的"。
    reviewed_by: Mapped[str | None] = mapped_column(String(128), default=None)
    auto_approved: Mapped[bool] = mapped_column(Boolean, default=False)
