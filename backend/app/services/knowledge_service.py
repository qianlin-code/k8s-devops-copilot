from sqlalchemy.orm import Session

from app.config import ChunkStrategyName, get_settings
from app.errors import ErrorCode, NonRetryableError
from app.knowledge.ingest import KnowledgeIngestor
from app.knowledge.sedimentation import SedimentationService
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import VectorStore
from app.schemas.base import to_utc_iso
from app.schemas.knowledge import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummary,
    IngestResponse,
    SedimentationEntry,
    SedimentationListResponse,
)
from app.storage.models import KBDocument, PendingSedimentation


class KnowledgeService:
    def __init__(
        self,
        *,
        ingestor: KnowledgeIngestor,
        sedimentation: SedimentationService,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self._ingestor = ingestor
        self._sedimentation = sedimentation
        self._store = vector_store
        self._bm25 = bm25_index

    def ingest(
        self,
        session: Session,
        *,
        title: str,
        content: str,
        chunk_strategy: str | None,
        source: str = "upload",
        source_ref: str | None = None,
    ) -> IngestResponse:
        strategy = _parse_strategy(chunk_strategy)
        result = self._ingestor.ingest_text(
            session,
            title=title,
            content=content,
            source=source,
            source_ref=source_ref,
            strategy=strategy,
        )
        doc = session.get(KBDocument, result.document_id)
        assert doc is not None
        return IngestResponse(
            document=_to_summary(doc), bm25_index_size=result.bm25_size
        )

    def list_documents(self, session: Session) -> DocumentListResponse:
        docs = self._ingestor.list_documents(session)
        return DocumentListResponse(
            collection_name=get_settings().collection_name,
            total=len(docs),
            vector_count=self._store.count(),
            bm25_index_size=self._bm25.size,
            documents=[_to_summary(d) for d in docs],
        )

    def delete_document(
        self, session: Session, document_id: str
    ) -> DeleteDocumentResponse:
        self._ingestor.delete_document(session, document_id)
        return DeleteDocumentResponse(
            document_id=document_id,
            deleted=True,
            vector_count=self._store.count(),
            bm25_index_size=self._bm25.size,
        )

    def mark_sedimentation(
        self,
        session: Session,
        *,
        conversation_id: str,
        marked_by: str,
        proposed_title: str | None,
    ) -> SedimentationEntry:
        entry = self._sedimentation.mark(
            session,
            conversation_id=conversation_id,
            marked_by=marked_by,
            proposed_title=proposed_title,
        )
        return _to_entry(entry)

    def list_sedimentations(
        self, session: Session, status: str | None
    ) -> SedimentationListResponse:
        rows = self._sedimentation.list_pending(session, status)
        return SedimentationListResponse(
            total=len(rows), entries=[_to_entry(r) for r in rows]
        )

    def review_sedimentation(
        self,
        session: Session,
        pending_id: str,
        *,
        reviewer: str,
        approved: bool,
        title_override: str | None,
        note: str | None,
    ) -> SedimentationEntry:
        if approved:
            entry = self._sedimentation.approve(
                session,
                pending_id,
                reviewer=reviewer,
                title_override=title_override,
                note=note,
            )
        else:
            entry = self._sedimentation.reject(
                session, pending_id, reviewer=reviewer, note=note
            )
        return _to_entry(entry)


def _parse_strategy(value: str | None) -> ChunkStrategyName | None:
    if value is None:
        return None
    try:
        return ChunkStrategyName(value)
    except ValueError as exc:
        raise NonRetryableError(
            f"chunk_strategy must be one of {[s.value for s in ChunkStrategyName]}",
            code=ErrorCode.VALIDATION_FAILED,
            details={"chunk_strategy": value},
        ) from exc


def _to_summary(doc: KBDocument) -> DocumentSummary:
    return DocumentSummary(
        document_id=doc.id,
        title=doc.title,
        source=doc.source,
        source_ref=doc.source_ref,
        chunk_strategy=doc.chunk_strategy,
        chunk_count=doc.chunk_count,
        char_count=doc.char_count,
        collection_name=doc.collection_name,
        created_at=to_utc_iso(doc.created_at),
    )


def _to_entry(entry: PendingSedimentation) -> SedimentationEntry:
    return SedimentationEntry(
        pending_id=entry.id,
        conversation_id=entry.conversation_id,
        question=entry.question,
        answer=entry.answer,
        proposed_title=entry.proposed_title,
        marked_by=entry.marked_by,
        status=entry.status,
        review_note=entry.review_note,
        kb_document_id=entry.kb_document_id,
        created_at=to_utc_iso(entry.created_at),
        reviewed_at=to_utc_iso(entry.reviewed_at) if entry.reviewed_at else None,
        reviewed_by=entry.reviewed_by,
        auto_approved=entry.auto_approved,
        quality_score=entry.quality_score,
        quality_reasoning=entry.quality_reasoning,
        duplicate_of_document_id=entry.duplicate_of_document_id,
        duplicate_score=entry.duplicate_score,
    )
