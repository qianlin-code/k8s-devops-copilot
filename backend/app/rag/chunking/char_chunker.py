from app.rag.chunking.base import Chunk, ChunkStrategy

_BOUNDARIES = ("\n\n", "\n", "。", "；", ". ", "! ", "? ")


class CharOverlapChunker(ChunkStrategy):
    """基线策略：定长滑窗 + 重叠，切点尽量落在自然断句上。"""

    name = "char"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [Chunk(text=cleaned, index=0)]

        chunks: list[Chunk] = []
        start = 0
        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            if end < len(cleaned):
                end = self._snap_to_boundary(cleaned, start, end)
            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, index=len(chunks)))
            if end >= len(cleaned):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _snap_to_boundary(self, text: str, start: int, end: int) -> int:
        """在窗口后 30% 区间内找断句符，找不到就硬切。"""
        floor = start + int(self.chunk_size * 0.7)
        for marker in _BOUNDARIES:
            found = text.rfind(marker, floor, end)
            if found > start:
                return found + len(marker)
        return end
