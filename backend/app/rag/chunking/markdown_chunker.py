import re

from app.rag.chunking.base import Chunk, ChunkStrategy
from app.rag.chunking.char_chunker import CharOverlapChunker

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# 处理步骤类小节标题关键词。K8s 语料统一遵循"现象/根因/处理步骤"三段式结构
# （docs_k8s/ 下 7 份文档全部如此），按最后一级标题关键词判断足够稳定，
# 不需要引入分类模型。
_PROCEDURAL_HEADING_KEYWORDS = ("处理步骤", "解决方案", "解决方法", "修复步骤")


def _infer_chunk_type(heading_path: list[str]) -> tuple[str | None, bool]:
    """按标题链最后一级推断 chunk 类型，返回 (chunk_type, is_procedural)。"""
    if not heading_path:
        return None, False
    last = heading_path[-1]
    if any(kw in last for kw in _PROCEDURAL_HEADING_KEYWORDS):
        return "procedural", True
    if "现象" in last:
        return "symptom", False
    if "根因" in last:
        return "root_cause", False
    return None, False


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
            chunk_type, is_procedural = _infer_chunk_type(heading_path)
            if len(body) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        text=body,
                        index=len(chunks),
                        heading_path=list(heading_path),
                        chunk_type=chunk_type,
                        is_procedural=is_procedural,
                    )
                )
                continue
            for part in self._fallback.split(body):
                chunks.append(
                    Chunk(
                        text=part.text,
                        index=len(chunks),
                        heading_path=list(heading_path),
                        chunk_type=chunk_type,
                        is_procedural=is_procedural,
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
