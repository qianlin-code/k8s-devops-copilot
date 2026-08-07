import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ErrorCode, NonRetryableError, NotFoundError
from app.knowledge.ingest import KnowledgeIngestor
from app.storage.models import (
    Conversation,
    Message,
    MessageRole,
    PendingSedimentation,
    SedimentationStatus,
)

_KB_TEMPLATE = """# {title}

## 问题现象
{question}

## 解决方案
{answer}

## 来源
由对话 {conversation_id} 沉淀，审核人 {reviewer}。
"""


class SedimentationService:
    """半自动沉淀：标记进待审队列，人工确认后才写入知识库。

    MVP 不做自动触发。生产级方案还需相似度去重、质量评分、多级审校。
    """

    def __init__(self, ingestor: KnowledgeIngestor) -> None:
        self._ingestor = ingestor

    def mark(
        self,
        session: Session,
        *,
        conversation_id: str,
        marked_by: str,
        proposed_title: str | None = None,
    ) -> PendingSedimentation:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(
                f"Conversation '{conversation_id}' not found",
                details={"conversation_id": conversation_id},
            )

        question, answer = self._latest_exchange(session, conversation_id)
        existing = session.scalar(
            select(PendingSedimentation).where(
                PendingSedimentation.conversation_id == conversation_id,
                PendingSedimentation.status == SedimentationStatus.PENDING.value,
            )
        )
        if existing is not None:
            raise NonRetryableError(
                "This conversation already has a pending sedimentation entry",
                code=ErrorCode.BUSINESS_RULE_VIOLATION,
                details={"pending_id": existing.id},
            )

        entry = PendingSedimentation(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            proposed_title=(proposed_title or question)[:255],
            marked_by=marked_by,
        )
        session.add(entry)
        session.flush()
        return entry

    def list_pending(
        self, session: Session, status: str | None = None
    ) -> list[PendingSedimentation]:
        stmt = select(PendingSedimentation)
        if status:
            stmt = stmt.where(PendingSedimentation.status == status)
        return list(
            session.scalars(stmt.order_by(PendingSedimentation.created_at.desc()))
        )

    def approve(
        self,
        session: Session,
        pending_id: str,
        *,
        reviewer: str,
        title_override: str | None = None,
        note: str | None = None,
    ) -> PendingSedimentation:
        entry = self._get_pending(session, pending_id)
        title = (title_override or entry.proposed_title).strip()
        document = _KB_TEMPLATE.format(
            title=title,
            question=entry.question,
            answer=entry.answer,
            conversation_id=entry.conversation_id,
            reviewer=reviewer,
        )
        result = self._ingestor.ingest_text(
            session,
            title=title,
            content=document,
            source="sedimentation",
            source_ref=entry.conversation_id,
        )
        entry.status = SedimentationStatus.APPROVED.value
        entry.kb_document_id = result.document_id
        entry.review_note = note
        entry.reviewed_at = datetime.now(timezone.utc)
        session.flush()
        return entry

    def reject(
        self, session: Session, pending_id: str, *, reviewer: str, note: str | None = None
    ) -> PendingSedimentation:
        entry = self._get_pending(session, pending_id)
        entry.status = SedimentationStatus.REJECTED.value
        entry.review_note = note or f"rejected by {reviewer}"
        entry.reviewed_at = datetime.now(timezone.utc)
        session.flush()
        return entry

    def _get_pending(self, session: Session, pending_id: str) -> PendingSedimentation:
        entry = session.get(PendingSedimentation, pending_id)
        if entry is None:
            raise NotFoundError(
                f"Sedimentation entry '{pending_id}' not found",
                details={"pending_id": pending_id},
            )
        if entry.status != SedimentationStatus.PENDING.value:
            raise NonRetryableError(
                f"Entry already reviewed with status '{entry.status}'",
                code=ErrorCode.BUSINESS_RULE_VIOLATION,
                details={"pending_id": pending_id, "status": entry.status},
            )
        return entry

    def _latest_exchange(
        self, session: Session, conversation_id: str
    ) -> tuple[str, str]:
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(20)
            )
        )
        answer = next(
            (m.content for m in messages if m.role == MessageRole.ASSISTANT.value), None
        )
        question = next(
            (m.content for m in messages if m.role == MessageRole.USER.value), None
        )
        if not question or not answer:
            raise NonRetryableError(
                "Conversation must contain at least one question and one answer",
                code=ErrorCode.BUSINESS_RULE_VIOLATION,
                details={"conversation_id": conversation_id},
            )
        return question, answer
