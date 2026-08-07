import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.errors import VectorStoreUnavailableError
from app.rag.chunking.base import Chunk


@dataclass(slots=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    heading_path: list[str]
    chunk_index: int
    score: float

    def citation_label(self) -> str:
        # Markdown 的 H1 常与文档标题同名，去掉避免 "手册 / 手册 > 章节" 这种重复
        path = [h for h in self.heading_path if h != self.document_title]
        return f"{self.document_title} / {' > '.join(path)}" if path else self.document_title

    @property
    def contextual_text(self) -> str:
        """与入库时 Chunk.contextual_text 保持同一种文本表示。

        必须一致：向量嵌入的是带标题链的文本，Rerank 若只看裸正文，
        「根因」这类段落会因为不含上文的关键词（如 403）而被判为不相关，
        导致重排把正确片段挤出 Top-N。
        """
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


class VectorStore:
    """Qdrant embedded 封装。集合名绑定 Embedding 模型与维度，切换模型即换集合。"""

    def __init__(self, *, path: str, collection: str, dim: int) -> None:
        self.collection = collection
        self.dim = dim
        Path(path).mkdir(parents=True, exist_ok=True)
        try:
            self._client = QdrantClient(path=path)
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Cannot open Qdrant storage: {type(exc).__name__}"
            ) from exc
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            if self._client.collection_exists(self.collection):
                info = self._client.get_collection(self.collection)
                actual = info.config.params.vectors.size  # type: ignore[union-attr]
                if actual != self.dim:
                    raise VectorStoreUnavailableError(
                        "Existing collection dimension does not match the configured "
                        "embedding model",
                        details={
                            "collection": self.collection,
                            "expected": self.dim,
                            "actual": actual,
                        },
                    )
                return
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant collection setup failed: {type(exc).__name__}"
            ) from exc

    def upsert_chunks(
        self,
        *,
        document_id: str,
        document_title: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> list[str]:
        if len(chunks) != len(vectors):
            raise VectorStoreUnavailableError(
                "Chunk and vector counts differ",
                details={"chunks": len(chunks), "vectors": len(vectors)},
            )
        points: list[qm.PointStruct] = []
        ids: list[str] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "document_title": document_title,
                        "text": chunk.text,
                        "heading_path": chunk.heading_path,
                        "chunk_index": chunk.index,
                    },
                )
            )
        if not points:
            return []
        try:
            self._client.upsert(collection_name=self.collection, points=points)
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant upsert failed: {type(exc).__name__}"
            ) from exc
        return ids

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        try:
            hits = self._client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant search failed: {type(exc).__name__}"
            ) from exc
        return [_to_scored(hit.id, hit.payload or {}, hit.score) for hit in hits]

    def iter_all_chunks(self) -> list[ScoredChunk]:
        """BM25 需要全量语料建倒排索引。"""
        results: list[ScoredChunk] = []
        offset: Any = None
        try:
            while True:
                points, offset = self._client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                results.extend(
                    _to_scored(p.id, p.payload or {}, 0.0) for p in points
                )
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant scroll failed: {type(exc).__name__}"
            ) from exc
        return results

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="document_id",
                                match=qm.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant delete failed: {type(exc).__name__}"
            ) from exc

    def known_document_ids(self) -> set[str]:
        return {c.document_id for c in self.iter_all_chunks() if c.document_id}

    def count(self) -> int:
        try:
            return self._client.count(self.collection, exact=True).count
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant count failed: {type(exc).__name__}"
            ) from exc

    def close(self) -> None:
        self._client.close()


def _to_scored(point_id: Any, payload: dict[str, Any], score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=str(point_id),
        document_id=str(payload.get("document_id", "")),
        document_title=str(payload.get("document_title", "")),
        text=str(payload.get("text", "")),
        heading_path=list(payload.get("heading_path") or []),
        chunk_index=int(payload.get("chunk_index", 0)),
        score=float(score),
    )


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = VectorStore(
            path=settings.qdrant_path,
            collection=settings.collection_name,
            dim=settings.embedding_dim,
        )
    return _store


def reset_vector_store() -> None:
    global _store
    if _store is not None:
        _store.close()
    _store = None
