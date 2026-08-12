"""Markdown 分块的 chunk_type / is_procedural 标记测试。

K8s 语料统一遵循"现象/根因/处理步骤"三段式结构，路由降权依赖这个标记
把"处理步骤"文本和"现象/根因"文本区分开，这里验证标记本身推断正确。
"""

from app.rag.chunking.markdown_chunker import MarkdownHeaderChunker

DOC = """# RBAC 权限故障排查

## Forbidden 权限不足

### 现象
用户执行 kubectl 命令返回 Forbidden。

### 根因
RBAC 权限规则不匹配。

### 处理步骤
1. 执行 kubectl auth can-i 确认权限
2. 检查 RoleBinding
"""


def _chunk_by_last_heading(chunks, title):
    return next(c for c in chunks if c.heading_path[-1] == title)


def test_procedural_section_is_marked() -> None:
    chunker = MarkdownHeaderChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(DOC)

    step_chunk = _chunk_by_last_heading(chunks, "处理步骤")
    assert step_chunk.is_procedural is True
    assert step_chunk.chunk_type == "procedural"


def test_symptom_and_cause_sections_are_not_procedural() -> None:
    chunker = MarkdownHeaderChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(DOC)

    symptom_chunk = _chunk_by_last_heading(chunks, "现象")
    cause_chunk = _chunk_by_last_heading(chunks, "根因")

    assert symptom_chunk.is_procedural is False
    assert symptom_chunk.chunk_type == "symptom"
    assert cause_chunk.is_procedural is False
    assert cause_chunk.chunk_type == "root_cause"


def test_section_without_recognized_heading_has_no_chunk_type() -> None:
    chunker = MarkdownHeaderChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split("# 手册\n\n## 概述\n\n这是一段说明文字。\n")

    overview = _chunk_by_last_heading(chunks, "概述")
    assert overview.chunk_type is None
    assert overview.is_procedural is False
