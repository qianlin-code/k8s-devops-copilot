import re
import threading

from rank_bm25 import BM25Okapi

from app.rag.vector_store import ScoredChunk

# 中文按字切分 + 英文数字按词切分，覆盖 "403"、"permission_level" 这类精确术语
_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        # 中文补充 bigram，缓解单字召回噪声
    chars = [t for t in tokens if len(t) == 1 and "一" <= t <= "鿿"]
    for a, b in zip(chars, chars[1:], strict=False):
        tokens.append(a + b)
    return tokens


class BM25Index:
    """轻量本地倒排索引。语料变更后需 rebuild，不做增量。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[ScoredChunk] = []

    def rebuild(self, chunks: list[ScoredChunk]) -> int:
        # 索引 contextual_text：标题里的关键词（如 "403 Forbidden"）本该可被检索到，
        # 且需与向量嵌入、Rerank 使用同一种文本表示。
        corpus = [tokenize(c.contextual_text) for c in chunks]
        with self._lock:
            self._chunks = list(chunks)
            self._bm25 = BM25Okapi(corpus) if corpus else None
        return len(chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        with self._lock:
            bm25, chunks = self._bm25, self._chunks
        if bm25 is None or not chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        results: list[ScoredChunk] = []
        for chunk, score in ranked[:top_k]:
            if score <= 0:
                break
            results.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    text=chunk.text,
                    heading_path=chunk.heading_path,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                )
            )
        return results


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
    return _index


def reset_bm25_index() -> None:
    global _index
    _index = None
