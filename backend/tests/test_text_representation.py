"""文本表示一致性测试。

向量嵌入、BM25 索引、Rerank 必须用同一种文本表示（带标题链的 contextual_text）。
若不一致，「根因」这类段落会因为脱离上文关键词而被 Rerank 误判为不相关，
把正确片段挤出 Top-N —— 这是实测中真实踩到的问题。
"""

from app.rag.chunking.base import Chunk
from app.rag.vector_store import ScoredChunk

SECTION_BODY = "账号的 permission_level 字段为 restricted，未被授予目标应用的访问权限。"
HEADING = ["登录与认证故障排查手册", "403 Forbidden 权限不足", "根因"]


def _scored(text: str, heading: list[str]) -> ScoredChunk:
    return ScoredChunk("c1", "d1", "手册", text, heading, 0, 0.0)


def test_chunk_and_scored_chunk_use_same_format() -> None:
    """入库端与检索端的 contextual_text 必须逐字符一致。"""
    chunk = Chunk(text=SECTION_BODY, index=0, heading_path=list(HEADING))
    scored = _scored(SECTION_BODY, list(HEADING))
    assert chunk.contextual_text == scored.contextual_text


def test_contextual_text_carries_heading_keywords() -> None:
    """裸正文不含 403，带上标题链后才含 —— 这正是 Rerank 需要的上下文。"""
    scored = _scored(SECTION_BODY, list(HEADING))
    assert "403" not in scored.text
    assert "403" in scored.contextual_text
    assert SECTION_BODY in scored.contextual_text


def test_contextual_text_without_heading_is_plain_text() -> None:
    scored = _scored(SECTION_BODY, [])
    assert scored.contextual_text == SECTION_BODY


def test_bm25_corpus_includes_heading_tokens() -> None:
    """BM25 语料必须含标题分词，否则标题里的术语无法参与检索。

    这里断言语料内容而非搜索结果：BM25 的 IDF 在极小语料下会把
    只出现在少数文档中的词打成非正分，用搜索结果断言会得到与实现无关的假失败。
    """
    from app.rag.bm25_index import BM25Index, tokenize

    index = BM25Index()
    assert index.rebuild([_scored(SECTION_BODY, list(HEADING))]) == 1

    indexed = set(tokenize(_scored(SECTION_BODY, list(HEADING)).contextual_text))
    bare = set(tokenize(SECTION_BODY))

    assert "403" in indexed, "标题里的 403 必须进入索引"
    assert "403" not in bare, "裸正文本就不含 403，对比才有意义"
    assert "forbidden" in indexed
    assert bare < indexed, "带标题链的分词应是裸正文分词的超集"


def test_bm25_ranks_heading_match_above_unrelated() -> None:
    """标题命中的片段应排在完全无关片段之前。"""
    from app.rag.bm25_index import BM25Index

    index = BM25Index()
    target = _scored(SECTION_BODY, list(HEADING))
    index.rebuild(
        [
            target,
            _scored("订阅账期逾期未付款，系统自动暂停服务。", ["手册", "服务暂停"]),
            _scored("连接 VPN 后每隔几分钟就掉线需要重新登录。", ["手册", "VPN 掉线"]),
        ]
    )
    hits = index.search("permission_level restricted 403", top_k=3)
    assert hits, "正文关键词应能命中"
    assert hits[0].chunk_id == target.chunk_id


def test_reranker_receives_contextual_text() -> None:
    """守住 Rerank 的输入是 contextual_text，防止重构时退回裸 text。"""
    from app.rag.reranker import BGEReranker

    captured: list[list[str]] = []

    class FakeModel:
        def compute_score(self, pairs, normalize=True):  # noqa: ANN001
            captured.extend(pairs)
            return [0.5] * len(pairs)

    reranker = BGEReranker("stub", use_fp16=False)
    reranker._model = FakeModel()  # 跳过真实模型加载
    result = reranker.rerank(
        "登录 403 什么原因", [_scored(SECTION_BODY, list(HEADING))], top_n=1
    )

    assert captured, "reranker 应把候选送进模型"
    _, passed_text = captured[0]
    assert "403" in passed_text, "Rerank 必须看到标题链，否则会误判不相关"
    assert result[0].rerank_score == 0.5
