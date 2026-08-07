from app.config import ChunkStrategyName, get_settings
from app.rag.chunking.base import Chunk, ChunkStrategy
from app.rag.chunking.char_chunker import CharOverlapChunker
from app.rag.chunking.markdown_chunker import MarkdownHeaderChunker

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "CharOverlapChunker",
    "MarkdownHeaderChunker",
    "build_chunker",
]


def build_chunker(name: ChunkStrategyName | None = None) -> ChunkStrategy:
    settings = get_settings()
    chosen = name or settings.chunk_strategy
    if chosen is ChunkStrategyName.CHAR:
        return CharOverlapChunker(settings.chunk_size, settings.chunk_overlap)
    return MarkdownHeaderChunker(settings.chunk_size, settings.chunk_overlap)
