import re

from app.rag.chunking.base import Chunk, ChunkStrategy
from app.rag.chunking.char_chunker import CharOverlapChunker

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


class MarkdownHeaderChunker(ChunkStrategy):
    """按标题层级切分，保留标题链；超长小节再套字符分块。

    Markitdown 转出的文档自带标题结构，按结构切能避免把一个解决方案腰斩。
    """

    name = "markdown"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self._fallback = CharOverlapChunker(chunk_size, chunk_overlap)

    def split(self, text: str) -> list[Chunk]:
        cleaned = text.strip()
        if not cleaned:
            return []

        sections = self._collect_sections(cleaned)
        if not sections:
            return self._fallback.split(cleaned)

        chunks: list[Chunk] = []
        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue
            if len(body) <= self.chunk_size:
                chunks.append(
                    Chunk(text=body, index=len(chunks), heading_path=list(heading_path))
                )
                continue
            for part in self._fallback.split(body):
                chunks.append(
                    Chunk(
                        text=part.text,
                        index=len(chunks),
                        heading_path=list(heading_path),
                    )
                )
        return chunks

    def _collect_sections(self, text: str) -> list[tuple[list[str], str]]:
        sections: list[tuple[list[str], str]] = []
        stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_code_block = False

        def flush() -> None:
            if buffer:
                sections.append(([title for _, title in stack], "\n".join(buffer)))
                buffer.clear()

        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                buffer.append(line)
                continue
            match = None if in_code_block else _HEADING.match(line)
            if match is None:
                buffer.append(line)
                continue
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        flush()
        return [s for s in sections if s[1].strip()]
