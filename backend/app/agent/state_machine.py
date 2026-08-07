import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent.answerer import Answerer, format_knowledge
from app.agent.router import RouteAction, RouteDecision, Router
from app.agent.sufficiency import SufficiencyChecker, SufficiencyVerdict
from app.agent.tools.base import ToolContext
from app.agent.tools.executor import ToolExecutor, ToolInvocation
from app.agent.tools.registry import ToolRegistry
from app.config import get_settings
from app.rag.reranker import RerankedChunk


class AgentOutcome(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    TOOL_ASSISTED_ANSWER = "tool_assisted_answer"
    WRITE_CONFIRMATION_REQUIRED = "write_confirmation_required"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


@dataclass(slots=True)
class PendingWriteAction:
    tool_name: str
    arguments: dict[str, Any]
    description: str
    reasoning: str


@dataclass(slots=True)
class AgentStep:
    step: int
    node: str
    detail: dict[str, Any]


@dataclass(slots=True)
class AgentRunResult:
    outcome: AgentOutcome
    answer: str
    steps: list[AgentStep]
    decisions: list[RouteDecision] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    sufficiency: SufficiencyVerdict | None = None
    pending_write: PendingWriteAction | None = None
    citations: list[RerankedChunk] = field(default_factory=list)


class AgentStateMachine:
    """手写状态机：route → (execute → verify) 循环 → answer。

    max_steps 兜底防死循环；写操作在执行前中断并返回待确认动作。
    """

    def __init__(
        self,
        *,
        router: Router,
        checker: SufficiencyChecker,
        answerer: Answerer,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> None:
        self._router = router
        self._checker = checker
        self._answerer = answerer
        self._executor = executor
        self._registry = registry

    def run(
        self,
        question: str,
        chunks: list[RerankedChunk],
        ctx: ToolContext,
        *,
        context_messages: list[dict[str, str]] | None = None,
        confirmed_write: PendingWriteAction | None = None,
        max_steps: int | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentRunResult:
        """on_step: 每走过一个节点即回调，供 SSE 实时推送阶段进展。

        回调里的异常不会中断状态机——推送失败（如客户端断连）不该让请求失败。
        """
        limit = max_steps or get_settings().agent_max_steps
        steps: list[AgentStep] = []
        decisions: list[RouteDecision] = []
        invocations: list[ToolInvocation] = []
        prior: list[str] = []
        # node_seq 只是 trace 里的序号；round 才是受 max_steps 约束的决策轮次
        node_seq = 0
        rounds = 0
        # 运行内幂等：失败的不重试，成功的不重复执行。
        failed_calls: dict[str, str] = {}
        executed_calls: set[str] = set()
        # 写操作按工具名去重：LLM 每轮生成的 reason/request_id 文本都不同，
        # 按完整参数签名会失效，导致确认执行后又弹一次确认卡片。
        executed_write_tools: set[str] = set()

        def record(node: str, detail: dict[str, Any]) -> None:
            nonlocal node_seq
            node_seq += 1
            step = AgentStep(step=node_seq, node=node, detail=detail)
            steps.append(step)
            if on_step is not None:
                try:
                    on_step(step)
                except Exception:  # noqa: BLE001
                    # 推送失败（客户端断连等）不应中断 Agent 执行
                    pass

        # 用户已确认的写操作：跳过路由直接执行
        if confirmed_write is not None:
            inv = self._executor.execute(
                confirmed_write.tool_name, confirmed_write.arguments, ctx
            )
            invocations.append(inv)
            record("execute_confirmed_write", _inv_detail(inv))
            prior.append(_prior_line(inv))
            signature = _call_signature(
                confirmed_write.tool_name, confirmed_write.arguments
            )
            if inv.success:
                executed_calls.add(signature)
                executed_write_tools.add(confirmed_write.tool_name)
            else:
                failed_calls[signature] = inv.error_code or "UNKNOWN"

        while rounds < limit:
            rounds += 1
            decision = self._router.decide(
                question,
                chunks,
                self._registry.catalog_for_prompt(),
                context_messages=context_messages,
                prior_steps=prior or None,
                current_user_id=ctx.user_id,
                forbidden_calls=sorted(failed_calls),
            )
            decisions.append(decision)
            record(
                "route",
                {
                    "round": rounds,
                    "action": decision.action.value,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                    "tool_name": decision.tool_name,
                },
            )

            if decision.action is RouteAction.INSUFFICIENT:
                verdict = SufficiencyVerdict(
                    sufficient=False,
                    reasoning=decision.reasoning,
                    missing_information=["路由判定现有知识与工具均无法覆盖该问题"],
                    suggested_next_step=decision.followup_question
                    or "补充更多信息或提交工单转人工",
                )
                return AgentRunResult(
                    outcome=AgentOutcome.INSUFFICIENT_INFORMATION,
                    answer=self._answerer.insufficient_answer(
                        verdict.missing_information, verdict.suggested_next_step
                    ),
                    steps=steps,
                    decisions=decisions,
                    invocations=invocations,
                    sufficiency=verdict,
                    citations=chunks,
                )

            if decision.action is RouteAction.CALL_TOOL:
                signature = _call_signature(
                    decision.tool_name or "", decision.tool_arguments
                )
                # 本轮已成功执行过的调用不再执行第二次。必须在写确认判断之前拦截，
                # 否则确认执行后会再次弹出确认卡片，永远走不到回答。
                tool_name = decision.tool_name or ""
                already_done = signature in executed_calls or (
                    tool_name in executed_write_tools
                )
                if already_done:
                    record(
                        "skip_already_executed_call",
                        {"tool_name": tool_name, "signature": signature},
                    )
                    prior.append(
                        f"{signature} 本轮已经执行成功，结果见上，"
                        "不要重复执行，请直接根据已有结果回答用户。"
                    )
                    verdict = self._checker.check(
                        question,
                        format_knowledge(chunks),
                        _format_invocations(invocations),
                    )
                    record(
                        "verify_sufficiency",
                        {
                            "sufficient": verdict.sufficient,
                            "reasoning": verdict.reasoning,
                            "missing_information": verdict.missing_information,
                        },
                    )
                    answer = self._answerer.answer(
                        question,
                        chunks,
                        _format_invocations(invocations),
                        context_messages=context_messages,
                    )
                    record("generate_answer", {"length": len(answer)})
                    return AgentRunResult(
                        outcome=AgentOutcome.TOOL_ASSISTED_ANSWER,
                        answer=answer,
                        steps=steps,
                        decisions=decisions,
                        invocations=invocations,
                        sufficiency=verdict,
                        citations=chunks,
                    )

                pending = self._maybe_pending_write(decision)
                if pending is not None:
                    record(
                        "await_write_confirmation",
                        {
                            "tool_name": pending.tool_name,
                            "arguments": pending.arguments,
                        },
                    )
                    return AgentRunResult(
                        outcome=AgentOutcome.WRITE_CONFIRMATION_REQUIRED,
                        answer=(
                            f"该操作会修改系统数据（{pending.description}），"
                            "确认后我再执行。"
                        ),
                        steps=steps,
                        decisions=decisions,
                        invocations=invocations,
                        pending_write=pending,
                        citations=chunks,
                    )

                if signature in failed_calls:
                    # 同一个失败调用不再执行第二次，否则会白烧掉全部步数
                    record(
                        "skip_repeated_failed_call",
                        {
                            "tool_name": decision.tool_name,
                            "signature": signature,
                            "previous_error": failed_calls[signature],
                        },
                    )
                    prior.append(
                        f"调用 {signature} 已失败过({failed_calls[signature]})，"
                        "不要重复提交，改用其他工具或改为向用户追问。"
                    )
                    continue

                inv = self._executor.execute(
                    decision.tool_name or "", decision.tool_arguments, ctx
                )
                invocations.append(inv)
                record("execute_tool", _inv_detail(inv))
                prior.append(_prior_line(inv))
                if inv.success:
                    executed_calls.add(signature)
                    if inv.is_write:
                        executed_write_tools.add(inv.tool_name)
                else:
                    failed_calls[signature] = inv.error_code or "UNKNOWN"

            verdict = self._checker.check(
                question, format_knowledge(chunks), _format_invocations(invocations)
            )
            record(
                "verify_sufficiency",
                {
                    "sufficient": verdict.sufficient,
                    "reasoning": verdict.reasoning,
                    "missing_information": verdict.missing_information,
                },
            )

            if verdict.sufficient:
                answer = self._answerer.answer(
                    question,
                    chunks,
                    _format_invocations(invocations),
                    context_messages=context_messages,
                )
                record("generate_answer", {"length": len(answer)})
                return AgentRunResult(
                    outcome=AgentOutcome.TOOL_ASSISTED_ANSWER
                    if invocations
                    else AgentOutcome.DIRECT_ANSWER,
                    answer=answer,
                    steps=steps,
                    decisions=decisions,
                    invocations=invocations,
                    sufficiency=verdict,
                    citations=chunks,
                )

            prior.append(
                f"充分性校验未通过：{verdict.reasoning}"
                + (f"，建议 {verdict.suggested_next_step}" if verdict.suggested_next_step else "")
            )

        record("max_steps_exceeded", {"rounds_used": rounds, "limit": limit})
        # 已经拿到证据时不该丢弃：带上"未通过充分性校验"的说明作答，
        # 比直接回"无法回答"更有用，也不会掩盖校验未通过的事实。
        has_evidence = bool(chunks) or any(inv.success for inv in invocations)
        if has_evidence:
            answer = self._answerer.answer(
                question,
                chunks,
                _format_invocations(invocations),
                context_messages=context_messages,
                caveat=(
                    "以下回答基于已收集到的资料，但未通过完整的信息充分性校验，"
                    "请自行核对关键结论。"
                ),
            )
        else:
            answer = self._answerer.insufficient_answer(
                ["经过多轮尝试仍未收集到足够信息"],
                "建议提交工单由人工工程师接手处理",
            )
        return AgentRunResult(
            outcome=AgentOutcome.MAX_STEPS_EXCEEDED,
            answer=answer,
            steps=steps,
            decisions=decisions,
            invocations=invocations,
            citations=chunks,
        )

    def _maybe_pending_write(
        self, decision: RouteDecision
    ) -> PendingWriteAction | None:
        name = decision.tool_name or ""
        if not self._registry.has(name):
            return None
        tool = self._registry.get(name)
        if not tool.is_write:
            return None
        return PendingWriteAction(
            tool_name=name,
            arguments=dict(decision.tool_arguments),
            description=tool.description,
            reasoning=decision.reasoning,
        )


def _inv_detail(inv: ToolInvocation) -> dict[str, Any]:
    return {
        "tool_name": inv.tool_name,
        "is_write": inv.is_write,
        "success": inv.success,
        "cache_hit": inv.cache_hit,
        "idempotent_replay": inv.idempotent_replay,
        "error_code": inv.error_code,
        "elapsed_ms": inv.elapsed_ms,
    }


def _call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """写操作的 request_id 每轮可能不同，签名时剔除，避免绕过重复检测。"""
    stable = {k: v for k, v in arguments.items() if k != "request_id"}
    return f"{tool_name}({json.dumps(stable, sort_keys=True, ensure_ascii=False)})"


def _prior_line(inv: ToolInvocation) -> str:
    if inv.success:
        return f"调用 {inv.tool_name} 成功，结果: {json.dumps(inv.result, ensure_ascii=False)[:300]}"
    return f"调用 {inv.tool_name} 失败: {inv.error_code} {inv.error_message}"


def _format_invocations(invocations: list[ToolInvocation]) -> str:
    if not invocations:
        return ""
    lines = []
    for inv in invocations:
        if inv.success:
            lines.append(
                f"- {inv.tool_name}: {json.dumps(inv.result, ensure_ascii=False)}"
            )
        else:
            lines.append(f"- {inv.tool_name} 执行失败: {inv.error_code} {inv.error_message}")
    return "\n".join(lines)
