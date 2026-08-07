from pydantic import BaseModel, Field

from app.errors import AppError
from app.llm.client import LLMClient

_SYSTEM = (
    "你是企业 IT 支持系统的检索查询改写器。"
    "把用户口语化的问题改写成更适合向量检索与关键词检索的标准查询："
    "补全省略的主语、展开缩写、保留原文中的错误码和专有名词、去掉情绪化表达。"
    "不要回答问题，只输出改写后的查询。"
)


class RewrittenQuery(BaseModel):
    rewritten: str = Field(description="改写后的检索查询")
    keywords: list[str] = Field(default_factory=list, description="用于关键词检索的核心词")


class QueryRewriteResult(BaseModel):
    original: str
    rewritten: str
    keywords: list[str] = Field(default_factory=list)
    applied: bool
    skip_reason: str | None = None


def rewrite_query(
    query: str,
    llm: LLMClient,
    *,
    enabled: bool,
    history_snippet: str | None = None,
) -> QueryRewriteResult:
    """改写失败不阻断主链路，退回原查询即可。"""
    if not enabled:
        return QueryRewriteResult(
            original=query, rewritten=query, applied=False, skip_reason="disabled"
        )

    user_content = query if not history_snippet else (
        f"对话上下文:\n{history_snippet}\n\n当前问题: {query}"
    )
    try:
        result = llm.structured(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            RewrittenQuery,
        )
    except AppError as exc:
        return QueryRewriteResult(
            original=query,
            rewritten=query,
            applied=False,
            skip_reason=f"llm_failure:{exc.code.value}",
        )

    cleaned = result.rewritten.strip()
    if not cleaned:
        return QueryRewriteResult(
            original=query, rewritten=query, applied=False, skip_reason="empty_output"
        )
    return QueryRewriteResult(
        original=query,
        rewritten=cleaned,
        keywords=[k.strip() for k in result.keywords if k.strip()][:8],
        applied=cleaned != query,
    )
