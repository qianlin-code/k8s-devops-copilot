"""检索阈值策略：知识问答可放宽，实时查询与写操作保持生产阈值。"""

import re


KNOWLEDGE_QUERY_MIN_SCORE = 0.03

_LIVE_QUERY_PREFIXES = ("查一下", "帮我查", "帮我看看", "帮忙查", "查询")
_WRITE_ACTION_PREFIXES = (
    "帮我重启",
    "请重启",
    "帮我创建",
    "帮我提",
    "提交工单",
    "创建工单",
    "更新工单",
)
_RESOURCE_TARGET = re.compile(
    r"(?:[a-z0-9][a-z0-9-]*\s*下\s*)?.*?\b(?:Pod|Deployment)\b",
    re.IGNORECASE,
)


def is_knowledge_only_question(question: str) -> bool:
    """判断问题能否只依赖知识库回答，而无需实时资源或写操作。"""
    normalized = question.strip()
    if normalized.startswith(_LIVE_QUERY_PREFIXES) or normalized.startswith(
        _WRITE_ACTION_PREFIXES
    ):
        return False
    if any(
        marker in normalized
        for marker in ("帮我重启", "请重启", "提个告警", "创建工单", "更新工单", "删掉")
    ):
        return False
    if "工单" in normalized and any(
        marker in normalized
        for marker in ("提", "创建", "提交", "更新", "有没有", "已创建", "处理进度")
    ):
        return False
    if _RESOURCE_TARGET.search(normalized) and any(
        marker in normalized for marker in ("当前", "现在", "状态", "是不是", "副本", "有没有")
    ):
        return False
    if _RESOURCE_TARGET.search(normalized) and any(
        marker in normalized for marker in ("重启", "创建", "更新", "删除")
    ):
        return False
    return True


def min_rerank_score_for_query(
    question: str,
    *,
    production_score: float,
    knowledge_score: float = KNOWLEDGE_QUERY_MIN_SCORE,
) -> float:
    """为问题选择阈值，禁止实时查询和写操作继承低分知识上下文。"""
    return knowledge_score if is_knowledge_only_question(question) else production_score
