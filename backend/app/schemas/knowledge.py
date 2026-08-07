from typing import Optional

from pydantic import Field

from app.schemas.base import StrictBaseModel


# 单篇文档正文上限。按默认 500 字/块算约 2000 块，一次入库要跑同样多次
# embedding 调用 —— 再大就该走离线批量入库而不是同步接口。
MAX_DOCUMENT_CHARS = 1_000_000


class IngestTextRequest(StrictBaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(
        min_length=1,
        max_length=MAX_DOCUMENT_CHARS,
        description="Markdown 或纯文本正文",
    )
    chunk_strategy: Optional[str] = Field(
        default=None, description="char / markdown，留空用配置默认值"
    )


class DocumentSummary(StrictBaseModel):
    document_id: str
    title: str
    source: str = Field(description="upload / sedimentation / file")
    source_ref: Optional[str] = None
    chunk_strategy: str
    chunk_count: int
    char_count: int
    collection_name: str
    created_at: str


class IngestResponse(StrictBaseModel):
    document: DocumentSummary
    bm25_index_size: int


class DocumentListResponse(StrictBaseModel):
    collection_name: str
    total: int
    vector_count: int
    bm25_index_size: int
    documents: list[DocumentSummary]


class DeleteDocumentResponse(StrictBaseModel):
    document_id: str
    deleted: bool
    vector_count: int
    bm25_index_size: int


class MarkSedimentationRequest(StrictBaseModel):
    conversation_id: str
    marked_by: str
    proposed_title: Optional[str] = Field(default=None, max_length=255)


class ReviewSedimentationRequest(StrictBaseModel):
    reviewer: str
    approved: bool
    title_override: Optional[str] = Field(default=None, max_length=255)
    note: Optional[str] = None


class SedimentationEntry(StrictBaseModel):
    pending_id: str
    conversation_id: str
    question: str
    answer: str
    proposed_title: str
    marked_by: str
    status: str = Field(description="pending / approved / rejected")
    review_note: Optional[str] = None
    kb_document_id: Optional[str] = None
    created_at: str
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = Field(
        default=None, description="审核人；自动初筛通过时为 system:auto-quality"
    )
    auto_approved: bool = Field(default=False, description="是否由自动质量初筛通过，未经人工审核")
    quality_score: Optional[float] = Field(default=None, description="云端小模型质量初筛分数")
    quality_reasoning: Optional[str] = None
    duplicate_of_document_id: Optional[str] = Field(
        default=None, description="非空表示疑似与已有知识库文档重复"
    )
    duplicate_score: Optional[float] = None


class SedimentationListResponse(StrictBaseModel):
    total: int
    entries: list[SedimentationEntry]
