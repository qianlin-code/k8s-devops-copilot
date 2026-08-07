from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.errors import AppError
from app.llm.client import LLMClient
from app.rag.reranker import RerankedChunk

_SYSTEM = """你是企业 IT 支持 Copilot 的决策路由器。给定用户问题、检索到的知识片段、可用工具列表，判断下一步该做什么。

决策规则:
- answer: 知识片段已足够回答用户问题，且无需查询用户的实时数据。
- call_tool: 需要用户账号/订单/工单的实时状态才能回答，或用户明确要求执行某项操作。
- insufficient: 知识片段不相关且没有合适工具能获取所需信息。

注意:
- 只能从给定工具列表中选择，不要发明工具名。
- 工具入参必须来自用户问题或对话上下文中已明确出现的信息，不要凭空编造账号 ID。
- 缺少必要参数(比如不知道账号 ID)时，选 answer 并在 reasoning 中说明需要向用户追问什么。
- 标注为"写操作"的工具会在执行前要求用户确认，你只需正常选择它。"""


class RouteAction(str, Enum):
    ANSWER = "answer"
    CALL_TOOL = "call_tool"
    INSUFFICIENT = "insufficient"


class RouteDecision(BaseModel):
    action: RouteAction = Field(description="下一步动作")
    reasoning: str = Field(description="做出该判断的理由，会展示给用户便于追溯")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="决策置信度")
    tool_name: Optional[str] = Field(default=None, description="action=call_tool 时必填")
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict, description="工具入参，需符合该工具的 schema"
    )
    followup_question: Optional[str] = Field(
        default=None, description="缺少信息时想向用户追问的内容"
    )


class Router:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def decide(
        self,
        question: str,
        chunks: list[RerankedChunk],
        tool_catalog: str,
        *,
        context_messages: list[dict[str, str]] | None = None,
        prior_steps: list[str] | None = None,
        current_user_id: str | None = None,
        forbidden_calls: list[str] | None = None,
    ) -> RouteDecision:
        knowledge = _format_chunks(chunks)
        parts = [
            f"[可用工具]\n{tool_catalog}",
            f"[检索到的知识片段]\n{knowledge}",
            f"[用户问题]\n{question}",
        ]
        if current_user_id:
            # 显式给出真实账号 ID，避免模型凭空编造
            parts.insert(
                0,
                f"[当前提问用户的账号 ID]\n{current_user_id}\n"
                "涉及本人账号的工具调用请直接使用该 ID；"
                "若用户问的是其他账号，只能使用问题中明确出现的 ID。",
            )
        if forbidden_calls:
            parts.insert(
                0,
                "[禁止重复的调用]\n以下调用本轮已经尝试过且失败，不要再次提交完全相同的调用：\n"
                + "\n".join(f"- {c}" for c in forbidden_calls),
            )
        if prior_steps:
            parts.insert(
                0, "[本轮已执行的步骤]\n" + "\n".join(f"- {s}" for s in prior_steps)
            )
        messages = [{"role": "system", "content": _SYSTEM}]
        if context_messages:
            messages.extend(context_messages)
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        try:
            decision = self._llm.structured(messages, RouteDecision)
        except AppError as exc:
            # 路由失败不能整体崩，退化成保守分支交由后续节点兜底
            return RouteDecision(
                action=RouteAction.INSUFFICIENT,
                reasoning=f"路由决策失败({exc.code.value})，转人工兜底。",
                confidence=0.0,
            )
        return _sanitize(decision)


def _sanitize(decision: RouteDecision) -> RouteDecision:
    if decision.action is RouteAction.CALL_TOOL and not decision.tool_name:
        return decision.model_copy(
            update={
                "action": RouteAction.INSUFFICIENT,
                "reasoning": decision.reasoning + "（未指定工具名，按信息不足处理）",
            }
        )
    if decision.action is not RouteAction.CALL_TOOL:
        return decision.model_copy(update={"tool_name": None, "tool_arguments": {}})
    return decision


def _format_chunks(chunks: list[RerankedChunk]) -> str:
    if not chunks:
        return "(无相关片段)"
    return "\n\n".join(
        f"[{i}] 来源: {c.chunk.citation_label()}\n{c.chunk.text}"
        for i, c in enumerate(chunks, start=1)
    )
