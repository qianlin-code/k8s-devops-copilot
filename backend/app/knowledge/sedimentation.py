import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import AppError, ErrorCode, NonRetryableError, NotFoundError
from app.knowledge.ingest import KnowledgeIngestor
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient
from app.rag.vector_store import VectorStore
from app.storage.models import (
    AUTO_QUALITY_REVIEWER,
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

# 与检索侧 min_rerank_score 是不同量纲（这里是原始余弦相似度，不经过 Rerank），
# 0.92 是较保守的阈值：宁可漏判重复也不要误删非重复内容。
_DUPLICATE_SIMILARITY_THRESHOLD = 0.92
# 质量分达到此阈值且非重复才自动通过，否则留给人工看分数决定
_AUTO_APPROVE_QUALITY_THRESHOLD = 0.8

_QUALITY_SYSTEM = """你是知识库沉淀内容的质量初筛员。给定一段来自客服对话的问答，评估它是否适合直接写入知识库。

评分维度：
- 完整度: 问题现象与解决方案是否说清楚了，而不是"请联系管理员"这类空泛话术
- 可回答性: 未来别的用户问类似问题时，这段内容能否独立作为答案使用
- 敏感信息: 是否包含真实账号密码、密钥、身份证号等不该沉淀的敏感数据（若有，quality_score 直接判 0）

quality_score 综合以上给 0-1 分，reasoning 简述理由。"""


class _QualityVerdict(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="简述评分理由，若发现敏感信息需明确指出")
    contains_sensitive_info: bool = Field(default=False)


class SedimentationService:
    """半自动沉淀：标记后自动初筛（去重 + 质量打分），高分且非重复自动入库，
    其余留待人工审核。

    自动初筛依赖云端小模型与向量检索，两者任一不可用时降级为纯人工审核
    （不因初筛失败阻塞标记动作）。
    """

    def __init__(
        self,
        ingestor: KnowledgeIngestor,
        *,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        quality_client: LLMClient | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._embed = embedding_client
        self._store = vector_store
        self._quality_llm = quality_client

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
        # 归属校验：`conversation_id` 是入口参数，少了这层任何持有 API Key 的人
        # 都能把别人的会话推进待审队列——队列条目会完整带出原对话的 question/answer，
        # 等于顺带泄露别人的对话内容。返回 404 而非 403，避免确认 id 存在（见
        # 评测与失败案例文档记录的越权枚举风险）。
        if conversation.user_id != marked_by:
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

        self._screen(session, entry)
        return entry

    def _screen(self, session: Session, entry: PendingSedimentation) -> None:
        """自动初筛：先查重，重复则不再打分（重复内容没有质量可言）；
        非重复再打质量分，达标则自动 approve。

        任一步骤依赖缺失或调用失败都不应阻塞标记动作——初筛失败时
        entry 保持 pending 状态，退化为纯人工审核，这是有意的降级路径。
        """
        try:
            dup_id, dup_score = self._check_duplicate(entry)
            entry.duplicate_of_document_id = dup_id
            entry.duplicate_score = dup_score
            session.flush()

            if dup_id is not None:
                return  # 疑似重复：不打分，留给人工判断是否合并或驳回

            score, reasoning, sensitive = self._check_quality(entry)
            entry.quality_score = score
            entry.quality_reasoning = reasoning
            session.flush()

            if sensitive or score < _AUTO_APPROVE_QUALITY_THRESHOLD:
                return  # 低分或含敏感信息：留给人工审核

            self.approve(
                session,
                entry.id,
                reviewer=AUTO_QUALITY_REVIEWER,
                note=f"自动初筛通过(quality_score={score:.2f})，{reasoning}",
                auto=True,
            )
        except AppError as exc:
            # 初筛服务不可用（未配置 QWEN_API_KEY 等）时静默降级为人工审核，
            # 不让评估调用的失败连带影响"标记"这个动作本身。
            entry.quality_reasoning = f"自动初筛不可用({exc.code.value})，转人工审核。"
            session.flush()

    def _check_duplicate(
        self, entry: PendingSedimentation
    ) -> tuple[str | None, float | None]:
        if self._embed is None or self._store is None:
            return None, None
        text = f"{entry.question}\n{entry.answer}"
        vector = self._embed.embed_one(text)
        hits = self._store.search(vector, top_k=1)
        if not hits:
            return None, None
        top = hits[0]
        if top.score >= _DUPLICATE_SIMILARITY_THRESHOLD:
            return top.document_id, top.score
        return None, top.score

    def _check_quality(
        self, entry: PendingSedimentation
    ) -> tuple[float, str, bool]:
        if self._quality_llm is None:
            raise NonRetryableError(
                "Quality screening client is not configured",
                code=ErrorCode.VALIDATION_FAILED,
            )
        payload = f"[问题]\n{entry.question}\n\n[回答]\n{entry.answer}"
        verdict = self._quality_llm.structured(
            [
                {"role": "system", "content": _QUALITY_SYSTEM},
                {"role": "user", "content": payload},
            ],
            _QualityVerdict,
        )
        score = 0.0 if verdict.contains_sensitive_info else verdict.quality_score
        return score, verdict.reasoning, verdict.contains_sensitive_info

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
        auto: bool = False,
    ) -> PendingSedimentation:
        """auto=True 表示自动初筛通过（reviewer 固定传 AUTO_QUALITY_REVIEWER），
        留痕在 reviewed_by/auto_approved，与人工审核区分，方便复盘谁批准的。
        """
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
        entry.reviewed_by = reviewer
        entry.auto_approved = auto
        session.flush()
        return entry

    def reject(
        self, session: Session, pending_id: str, *, reviewer: str, note: str | None = None
    ) -> PendingSedimentation:
        entry = self._get_pending(session, pending_id)
        entry.status = SedimentationStatus.REJECTED.value
        entry.review_note = note or f"rejected by {reviewer}"
        entry.reviewed_at = datetime.now(timezone.utc)
        entry.reviewed_by = reviewer
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
