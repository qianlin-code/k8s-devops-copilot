import json
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent.answerer import Answerer, VerifiedAnswerResult, format_knowledge
from app.agent.router import RouteAction, RouteDecision, Router
from app.agent.sufficiency import SufficiencyChecker, SufficiencyVerdict
from app.agent.tools.base import Tool, ToolContext
from app.agent.tools.executor import ToolExecutor, ToolInvocation
from app.agent.tools.registry import ToolRegistry
from app.config import get_settings
from app.errors import ToolError
from app.rag.query_policy import is_knowledge_only_question
from app.rag.reranker import RerankedChunk


class AgentOutcome(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    TOOL_ASSISTED_ANSWER = "tool_assisted_answer"
    WRITE_CONFIRMATION_REQUIRED = "write_confirmation_required"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    # 状态机自身不会产出这个值，它由 ChatService.confirm_write 在用户拒绝时给出。
    # 仍收进枚举：`ChatResponse.outcome` 是宽松 str，前端 OUTCOME_LABELS 却要
    # 逐个硬编码映射，散落的字面量容易漏映射或拼错。
    WRITE_REJECTED = "write_rejected"


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
    verified_answer: VerifiedAnswerResult | None = None


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
        # 运行内幂等：失败的不重试，成功的不重复执行。签名只看每个工具声明的
        # idempotency_fields（默认全部参数），排除了 reason 这类自由文本，
        # 所以能直接按签名判重——不需要再叠加"按工具名整体去重"这层，那样会把
        # restart_deployment(A) 和 restart_deployment(B) 这两个不同目标混为一谈，
        # 静默跳过 B；该失败模式记录在 docs/评测与失败案例.md。
        failed_calls: dict[str, str] = {}
        executed_calls: set[str] = set()
        # 连续「路由给出的动作无法执行」的轮次数。7B 模型在填不出必填参数时会
        # 一遍遍重复同一个无效调用（实测 6 轮里 4 轮都是纯路由+跳过，359s 后以
        # max_steps_exceeded 收场）。再问一次路由不会有新信息，必须在代码层收敛。
        unproductive_rounds = 0
        retried_explicit_tool_route = False

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
                confirmed_write.tool_name, confirmed_write.arguments, self._registry
            )
            if inv.success:
                executed_calls.add(signature)
            else:
                failed_calls[signature] = inv.error_code or "UNKNOWN"

        def finish_with_answer(
            *, outcome: AgentOutcome, verdict: SufficiencyVerdict | None = None
        ) -> AgentRunResult:
            verified = self._answerer.answer(
                question,
                chunks,
                invocations,
                context_messages=context_messages,
            )
            final_outcome = (
                AgentOutcome.INSUFFICIENT_INFORMATION
                if verified.status == "fallback"
                else outcome
            )
            record(
                "generate_answer",
                {
                    "length": len(verified.text),
                    "status": verified.status,
                    "attempts": verified.attempts,
                    "fallback_reason": verified.fallback_reason,
                },
            )
            return AgentRunResult(
                outcome=final_outcome,
                answer=verified.text,
                steps=steps,
                decisions=decisions,
                invocations=invocations,
                sufficiency=verdict,
                citations=chunks,
                verified_answer=verified,
            )

        # 通过相关性阈值的知识片段已是可引用证据。对于没有明确实时资源目标或
        # 操作意图的问题，直接由回答器作答，避免 7B 把手册中的“检查/重启”等
        # 操作步骤误读成当前需要调用工具的指令。
        if not invocations and chunks and is_knowledge_only_question(question):
            decision = RouteDecision(
                action=RouteAction.ANSWER,
                reasoning="已检索到通过相关性阈值的知识证据，问题未请求实时状态或操作。",
                confidence=1.0,
            )
            decisions.append(decision)
            record(
                "route",
                {
                    "round": 0,
                    "action": decision.action.value,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                    "tool_name": None,
                    "policy": "knowledge_evidence_direct_answer",
                },
            )
            return finish_with_answer(outcome=AgentOutcome.DIRECT_ANSWER)

        def settle() -> AgentRunResult:
            """收敛本轮：用现有证据做一次充分性校验，据结果回答或转追问。

            用在「路由反复给出无法执行的动作」时。仍然过充分性校验而不是直接
            回答 —— 反幻觉这道闸不能因为收敛就绕开。
            """
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
                return finish_with_answer(
                    outcome=AgentOutcome.TOOL_ASSISTED_ANSWER
                    if invocations
                    else AgentOutcome.DIRECT_ANSWER,
                    verdict=verdict,
                )
            record("settle_insufficient", {"reason": "router_stuck"})
            return AgentRunResult(
                outcome=AgentOutcome.INSUFFICIENT_INFORMATION,
                answer=self._answerer.insufficient_answer(
                    verdict.missing_information,
                    verdict.suggested_next_step or "补充缺失的信息后再试",
                ),
                steps=steps,
                decisions=decisions,
                invocations=invocations,
                sufficiency=verdict,
                citations=chunks,
            )

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
                if (
                    not retried_explicit_tool_route
                    and not is_knowledge_only_question(question)
                    and rounds < limit
                ):
                    retried_explicit_tool_route = True
                    prior.append(
                        "用户的问题包含明确的实时查询或操作意图。若可用工具能够覆盖，"
                        "即使缺少参数也必须选择 call_tool 和对应工具；只填写用户明确"
                        "提供的参数，缺失字段由服务端统一追问。"
                    )
                    record(
                        "retry_explicit_tool_route",
                        {"reason": "explicit_tool_request_routed_as_insufficient"},
                    )
                    continue
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
                field_issues = self._tool_field_issues(
                    decision, question, context_messages
                )
                if field_issues:
                    tool = (
                        self._registry.get(decision.tool_name or "")
                        if self._registry.has(decision.tool_name or "")
                        else None
                    )
                    is_write = bool(tool and tool.is_write)
                    verdict = SufficiencyVerdict(
                        sufficient=False,
                        reasoning="工具参数缺失、格式非法，或包含用户未明确提供的值。",
                        missing_information=field_issues,
                        suggested_next_step=(
                            "请补充上述字段后，我会先展示待确认的操作。"
                            if is_write
                            else "请补充上述字段后，我再查询实时状态。"
                        ),
                    )
                    record(
                        "request_write_details" if is_write else "request_tool_details",
                        {"tool_name": decision.tool_name, "missing_fields": field_issues},
                    )
                    return AgentRunResult(
                        outcome=AgentOutcome.INSUFFICIENT_INFORMATION,
                        answer=self._answerer.insufficient_answer(
                            field_issues, verdict.suggested_next_step
                        ),
                        steps=steps,
                        decisions=decisions,
                        invocations=invocations,
                        sufficiency=verdict,
                        citations=chunks,
                    )
                signature = _call_signature(
                    decision.tool_name or "", decision.tool_arguments, self._registry
                )
                # 本轮已成功执行过的调用不再执行第二次。必须在写确认判断之前拦截，
                # 否则确认执行后会再次弹出确认卡片，永远走不到回答。
                tool_name = decision.tool_name or ""
                if signature in executed_calls:
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
                    return finish_with_answer(
                        outcome=AgentOutcome.TOOL_ASSISTED_ANSWER,
                        verdict=verdict,
                    )

                if signature in failed_calls:
                    # 同一个失败调用不再执行第二次，否则会白烧掉全部步数。
                    # 必须在写确认判断之前拦截——否则确认执行失败（如
                    # RESOURCE_NOT_FOUND）后，同一个写工具签名会先命中下面
                    # 的"待确认写操作"分支，把失败包装成新的确认卡片再弹一次，
                    # 永远走不到失败提示。
                    unproductive_rounds += 1
                    record(
                        "skip_repeated_failed_call",
                        {
                            "tool_name": decision.tool_name,
                            "signature": signature,
                            "previous_error": failed_calls[signature],
                            "unproductive_rounds": unproductive_rounds,
                        },
                    )
                    prior.append(
                        f"调用 {signature} 已失败过({failed_calls[signature]})，"
                        "不要重复提交，改用其他工具或改为向用户追问。"
                    )
                    # 给一次换工具或改追问的机会，再重复就收敛。
                    # 原来这里直接 continue，跳过充分性校验进下一轮路由——模型
                    # 没有任何新信息，只会重复同一个无效调用，把 max_steps 全烧在
                    # 「路由→跳过→路由」上（实测知识性问题 359s / 6 轮全耗尽，
                    # 以 max_steps_exceeded 收场）。
                    if unproductive_rounds >= 2:
                        return settle()
                    continue

                pending, missing_write_fields = self._prepare_pending_write(decision)
                if missing_write_fields:
                    verdict = SufficiencyVerdict(
                        sufficient=False,
                        reasoning="写操作缺少通过参数校验所需的信息。",
                        missing_information=missing_write_fields,
                        suggested_next_step="请补充上述字段后，我会先展示待确认的操作。",
                    )
                    record(
                        "request_write_details",
                        {"tool_name": decision.tool_name, "missing_fields": missing_write_fields},
                    )
                    return AgentRunResult(
                        outcome=AgentOutcome.INSUFFICIENT_INFORMATION,
                        answer=self._answerer.insufficient_answer(
                            missing_write_fields, verdict.suggested_next_step
                        ),
                        steps=steps,
                        decisions=decisions,
                        invocations=invocations,
                        sufficiency=verdict,
                        citations=chunks,
                    )
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

                inv = self._executor.execute(
                    decision.tool_name or "", decision.tool_arguments, ctx
                )
                invocations.append(inv)
                record("execute_tool", _inv_detail(inv))
                prior.append(_prior_line(inv))
                if inv.success:
                    # 真的执行了工具就是有产出，重置计数：否则一次成功调用之后
                    # 偶发一次重复失败也会被算进"卡住"，提前收敛掉正常的多步链路
                    unproductive_rounds = 0
                    executed_calls.add(signature)
                    if not inv.is_write:
                        # 成功的只读结果就是当前问题所需的实时事实；不再让同一个
                        # 本地模型二次审核并错误否决，导致无信息循环或步数耗尽。
                        return finish_with_answer(
                            outcome=AgentOutcome.TOOL_ASSISTED_ANSWER
                        )
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
                return finish_with_answer(
                    outcome=AgentOutcome.TOOL_ASSISTED_ANSWER
                    if invocations
                    else AgentOutcome.DIRECT_ANSWER,
                    verdict=verdict,
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
            return finish_with_answer(
                outcome=AgentOutcome.MAX_STEPS_EXCEEDED,
            )
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

    def _prepare_pending_write(
        self, decision: RouteDecision
    ) -> tuple[PendingWriteAction | None, list[str]]:
        name = decision.tool_name or ""
        if not self._registry.has(name):
            return None, []
        tool = self._registry.get(name)
        if not tool.is_write:
            return None, []

        # request_id 是服务端幂等键，不是要求用户或模型补齐的业务信息。
        # 只在其余参数已可能完整时补入，避免把缺字段的写请求误包装成确认操作。
        raw_args = dict(decision.tool_arguments)
        raw_args.setdefault("request_id", f"agent-{uuid.uuid4().hex}")
        try:
            parsed = tool.parse_args(raw_args)
        except ToolError as exc:
            violations = exc.details.get("violations", [])
            fields = [str(item.get("field", "参数")) for item in violations]
            return None, _format_tool_field_issues(tool, fields) or ["写操作所需参数"]
        normalized = parsed.model_dump(mode="json")
        decision.tool_arguments = normalized
        return PendingWriteAction(
            tool_name=name,
            arguments=normalized,
            description=tool.description,
            reasoning=decision.reasoning,
        ), []

    def _tool_field_issues(
        self,
        decision: RouteDecision,
        question: str,
        context_messages: list[dict[str, str]] | None,
    ) -> list[str]:
        name = decision.tool_name or ""
        if not self._registry.has(name):
            return []
        tool = self._registry.get(name)
        raw_args = dict(decision.tool_arguments)
        if tool.is_write:
            raw_args["request_id"] = "server-validation-placeholder"
        issues: list[str] = []
        try:
            tool.parse_args(raw_args)
        except ToolError as exc:
            violations = exc.details.get("violations", [])
            issues.extend(str(item.get("field", "参数")) for item in violations)

        grounded_fields = set(tool.user_grounded_fields)
        if "namespace" in tool.args_schema.model_fields:
            grounded_fields.add("namespace")
        user_text = _normalize_grounding_text(
            "\n".join(
            [
                *(message.get("content", "") for message in context_messages or [] if message.get("role") == "user"),
                question,
            ]
            )
        )
        for field_name in sorted(grounded_fields):
            value = decision.tool_arguments.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_value = _normalize_grounding_text(value)
            if not normalized_value or normalized_value not in user_text:
                issues.append(field_name)
        return _format_tool_field_issues(tool, issues)


def _normalize_grounding_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _format_tool_field_issues(
    tool: Tool[Any, Any], field_paths: list[str]
) -> list[str]:
    properties = tool.args_schema.model_json_schema().get("properties", {})
    formatted: list[str] = []
    for field_path in sorted(set(field_paths)):
        if not field_path or field_path == "request_id":
            continue
        field_schema = properties.get(field_path)
        if not isinstance(field_schema, dict):
            formatted.append(field_path)
            continue

        min_length = field_schema.get("minLength")
        max_length = field_schema.get("maxLength")
        has_min = isinstance(min_length, int) and not isinstance(min_length, bool)
        has_max = isinstance(max_length, int) and not isinstance(max_length, bool)
        if has_min and has_max:
            formatted.append(f"{field_path}（{min_length}–{max_length} 个字符）")
        elif has_min:
            formatted.append(f"{field_path}（至少 {min_length} 个字符）")
        elif has_max:
            formatted.append(f"{field_path}（最多 {max_length} 个字符）")
        else:
            formatted.append(field_path)
    return formatted


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


def _call_signature(
    tool_name: str, arguments: dict[str, Any], registry: ToolRegistry
) -> str:
    """构造判重签名。

    优先用工具声明的 idempotency_fields（排除 reason 等自由文本，只看真正
    决定操作身份的字段）；工具查不到（幻觉工具名）或未声明时，退化为
    「除 request_id 外的全部参数」——request_id 本身每轮都会变，必须排除，
    否则同一个操作会被误判成不同签名，绕过重复检测。
    """
    fields: tuple[str, ...] | None = None
    if registry.has(tool_name):
        args_schema = registry.get(tool_name).args_schema
        fields = getattr(args_schema, "idempotency_fields", None)
    if fields:
        stable = {k: arguments.get(k) for k in fields}
    else:
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
