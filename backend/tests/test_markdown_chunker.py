"""Markdown 分块的结构边界与 chunk 类型测试。

K8s 语料统一遵循"现象/根因/处理步骤"三段式结构，路由降权依赖这个标记
把"处理步骤"文本和"现象/根因"文本区分开，这里验证标记本身推断正确。
"""

from pathlib import Path

from app.rag.chunking.markdown_chunker import MarkdownHeaderChunker

_BACKEND_ROOT = Path(__file__).resolve().parents[1]

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


def test_real_q09_and_q14_procedures_keep_multiline_items_intact() -> None:
    chunker = MarkdownHeaderChunker(chunk_size=500, chunk_overlap=50)
    cases = (
        (
            "rbac_and_quota_troubleshooting.md",
            "4. 若操作的是命名空间级资源但只找到了集群范围的 `ClusterRoleBinding`，\n"
            "   或反过来只有命名空间内的 `RoleBinding` 却想操作集群范围资源，需要确认\n"
            "   绑定类型和资源类型的范围是否匹配——范围不匹配时即使角色定义本身没问题，\n"
            "   权限也不会生效",
        ),
        (
            "dns_troubleshooting.md",
            "4. 查看 CoreDNS 日志排查是否有异常（`kubectl logs --namespace=kube-system -l k8s-app=kube-dns`），\n"
            "   必要时给 CoreDNS 的 Corefile 临时加上 `log` 插件，确认查询确实被接收到",
        ),
    )

    for filename, expected_item in cases:
        source = (_BACKEND_ROOT / "data" / "docs_k8s" / filename).read_text(
            encoding="utf-8"
        )
        procedural = [
            chunk
            for chunk in chunker.split(source)
            if chunk.heading_path and chunk.heading_path[-1] == "处理步骤"
        ]
        assert any(expected_item in chunk.text for chunk in procedural)


def test_procedural_items_are_packed_without_splitting_or_overlap() -> None:
    source = """# 手册

## 处理步骤
1. 第一项包含一段较长说明
   第一项的续行必须留在第一项
2. 第二项也包含说明
   第二项的续行必须留在第二项
3. 第三项保持原始顺序
"""
    chunker = MarkdownHeaderChunker(chunk_size=55, chunk_overlap=20)

    chunks = chunker.split(source)

    assert [chunk.text for chunk in chunks] == [
        "1. 第一项包含一段较长说明\n   第一项的续行必须留在第一项",
        "2. 第二项也包含说明\n   第二项的续行必须留在第二项\n3. 第三项保持原始顺序",
    ]


def test_unordered_item_keeps_continuation_and_fenced_code_block() -> None:
    source = """# 手册

## 排查步骤
- 检查当前配置
  说明文字属于当前列表项
  ```yaml
  policyTypes:
    - Ingress
  ```
- 验证修复结果
"""
    chunker = MarkdownHeaderChunker(chunk_size=70, chunk_overlap=20)

    chunks = chunker.split(source)

    assert [chunk.text for chunk in chunks] == [
        "- 检查当前配置\n"
        "  说明文字属于当前列表项\n"
        "  ```yaml\n"
        "  policyTypes:\n"
        "    - Ingress\n"
        "  ```",
        "- 验证修复结果",
    ]


def test_single_oversized_procedural_item_is_preserved_as_one_chunk() -> None:
    oversized_item = "1. " + "完整说明" * 40
    source = f"# 手册\n\n## 处理步骤\n{oversized_item}\n2. 后续验证\n"
    chunker = MarkdownHeaderChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.split(source)

    assert [chunk.text for chunk in chunks] == [oversized_item, "2. 后续验证"]
    assert len(chunks[0].text) > chunker.chunk_size
