import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ChunkStrategyName, get_settings
from app.errors import ErrorCode, NonRetryableError, NotFoundError
from app.llm.embedding import EmbeddingClient
from app.rag.bm25_index import BM25Index
from app.rag.chunking import build_chunker
from app.rag.vector_store import VectorStore
from app.storage.models import KBDocument


@dataclass(slots=True)
class IngestResult:
    document_id: str
    title: str
    chunk_count: int
    char_count: int
    chunk_strategy: str
    collection_name: str
    bm25_size: int


class KnowledgeIngestor:
    """文档入库：分块 → 向量化 → Qdrant → 重建 BM25 索引。"""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        bm25_index: BM25Index,
    ) -> None:
        self._store = vector_store
        self._embed = embedding_client
        self._bm25 = bm25_index

    def ingest_text(
        self,
        session: Session,
        *,
        title: str,
        content: str,
        source: str = "upload",
        source_ref: str | None = None,
        strategy: ChunkStrategyName | None = None,
    ) -> IngestResult:
        text = content.strip()
        if not text:
            raise NonRetryableError(
                "Document content is empty", code=ErrorCode.VALIDATION_FAILED
            )

        chunker = build_chunker(strategy)
        chunks = chunker.split(text)
        if not chunks:
            raise NonRetryableError(
                "Chunking produced no segments", code=ErrorCode.VALIDATION_FAILED
            )

        settings = get_settings()
        document_id = str(uuid.uuid4())
        # 标题链拼进向量文本，避免切片脱离上下文
        vectors = self._embed.embed([c.contextual_text for c in chunks])
        self._store.upsert_chunks(
            document_id=document_id,
            document_title=title,
            chunks=chunks,
            vectors=vectors,
        )

        session.add(
            KBDocument(
                id=document_id,
                title=title,
                source=source,
                source_ref=source_ref,
                collection_name=settings.collection_name,
                chunk_strategy=chunker.name,
                chunk_count=len(chunks),
                char_count=len(text),
            )
        )
        session.flush()
        bm25_size = self.rebuild_bm25()
        return IngestResult(
            document_id=document_id,
            title=title,
            chunk_count=len(chunks),
            char_count=len(text),
            chunk_strategy=chunker.name,
            collection_name=settings.collection_name,
            bm25_size=bm25_size,
        )

    def delete_document(self, session: Session, document_id: str) -> None:
        doc = session.get(KBDocument, document_id)
        if doc is None:
            raise NotFoundError(
                f"Document '{document_id}' not found",
                details={"document_id": document_id},
            )
        self._store.delete_document(document_id)
        session.delete(doc)
        session.flush()
        self.rebuild_bm25()

    def rebuild_bm25(self) -> int:
        """Qdrant 是唯一数据源，BM25 索引从中重建。"""
        return self._bm25.rebuild(self._store.iter_all_chunks())

    def reconcile(self, session: Session) -> dict[str, int]:
        """清理元数据已不存在的孤儿向量。

        SQLite 存元数据、Qdrant 存向量，两者可能因单边重置而漂移；
        残留向量会永久污染检索结果，所以启动时对账一次。
        """
        settings = get_settings()
        known = {
            doc_id
            for (doc_id,) in session.execute(
                select(KBDocument.id).where(
                    KBDocument.collection_name == settings.collection_name
                )
            )
        }
        orphans = self._store.known_document_ids() - known
        for document_id in orphans:
            self._store.delete_document(document_id)
        return {
            "orphan_documents": len(orphans),
            "indexed_chunks": self.rebuild_bm25(),
        }

    def list_documents(self, session: Session) -> list[KBDocument]:
        settings = get_settings()
        return list(
            session.scalars(
                select(KBDocument)
                .where(KBDocument.collection_name == settings.collection_name)
                .order_by(KBDocument.created_at.desc())
            )
        )
