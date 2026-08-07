from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.dependencies import get_ingestor, get_sedimentation_service
from app.rag.bm25_index import get_bm25_index
from app.rag.vector_store import get_vector_store
from app.schemas.common import ERROR_RESPONSES
from app.schemas.knowledge import (
    DeleteDocumentResponse,
    DocumentListResponse,
    IngestResponse,
    IngestTextRequest,
    MarkSedimentationRequest,
    ReviewSedimentationRequest,
    SedimentationEntry,
    SedimentationListResponse,
)
from app.services.knowledge_service import KnowledgeService
from app.storage.db import get_db

router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(require_api_key)],
    responses=ERROR_RESPONSES,
)


def _service() -> KnowledgeService:
    return KnowledgeService(
        ingestor=get_ingestor(),
        sedimentation=get_sedimentation_service(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
    )


@router.post("/documents", response_model=IngestResponse, summary="文本入库")
async def ingest_document(
    payload: IngestTextRequest, session: Session = Depends(get_db)
) -> IngestResponse:
    return _service().ingest(
        session,
        title=payload.title,
        content=payload.content,
        chunk_strategy=payload.chunk_strategy,
    )


@router.get("/documents", response_model=DocumentListResponse, summary="文档列表")
async def list_documents(session: Session = Depends(get_db)) -> DocumentListResponse:
    return _service().list_documents(session)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="删除文档及其向量",
)
async def delete_document(
    document_id: str, session: Session = Depends(get_db)
) -> DeleteDocumentResponse:
    return _service().delete_document(session, document_id)


@router.post(
    "/sedimentations",
    response_model=SedimentationEntry,
    summary="标记优质对话，进入待审队列（不自动入库）",
)
async def mark_sedimentation(
    payload: MarkSedimentationRequest, session: Session = Depends(get_db)
) -> SedimentationEntry:
    return _service().mark_sedimentation(
        session,
        conversation_id=payload.conversation_id,
        marked_by=payload.marked_by,
        proposed_title=payload.proposed_title,
    )


@router.get(
    "/sedimentations",
    response_model=SedimentationListResponse,
    summary="待审/已审沉淀条目列表",
)
async def list_sedimentations(
    status: str | None = Query(default=None, description="pending/approved/rejected"),
    session: Session = Depends(get_db),
) -> SedimentationListResponse:
    return _service().list_sedimentations(session, status)


@router.post(
    "/sedimentations/{pending_id}/review",
    response_model=SedimentationEntry,
    summary="人工审核：通过则写入知识库，驳回则仅标记",
)
async def review_sedimentation(
    pending_id: str,
    payload: ReviewSedimentationRequest,
    session: Session = Depends(get_db),
) -> SedimentationEntry:
    return _service().review_sedimentation(
        session,
        pending_id,
        reviewer=payload.reviewer,
        approved=payload.approved,
        title_override=payload.title_override,
        note=payload.note,
    )
