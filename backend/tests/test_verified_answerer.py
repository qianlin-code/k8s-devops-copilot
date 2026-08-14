from typing import Any

import pytest

from app.agent.answerer import (
    Answerer,
    VerifiedAnswerResult,
    VerifiedEvidence,
    VerifiedAnswerPlan,
    _enumerate_tool_values,
    enumerate_evidence_atoms,
    format_evidence_atoms,
)
from app.agent.state_machine import AgentOutcome, AgentRunResult
from app.agent.tools.executor import ToolInvocation
from app.errors import LLMUnavailableError, NonRetryableError
from app.rag.reranker import RerankedChunk
from app.rag.vector_store import ScoredChunk
from app.services.chat_service import ChatService


class PlanLLM:
    def __init__(self, *plans: dict[str, Any] | Exception) -> None:
        self.plans = list(plans)
        self.calls: list[dict[str, Any]] = []
        self.messages: list[list[dict[str, str]]] = []

    def structured(self, messages, schema, **kwargs):  # noqa: ANN001
        self.messages.append(messages)
        self.calls.append(kwargs)
        item = self.plans.pop(0)
        if isinstance(item, Exception):
            raise item
        return schema.model_validate(item)


def plan(*atom_ids: str) -> dict[str, Any]:
    return {"selected_atom_ids": list(atom_ids)}


def chunk(
    text: str,
    heading: str = "根因",
    *,
    chunk_type: str | None = None,
    is_procedural: bool = False,
) -> RerankedChunk:
    return RerankedChunk(
        chunk=ScoredChunk(
            chunk_id=f"chunk-{heading}",
            document_id="k8s-doc",
            document_title="K8s 故障手册",
            text=text,
            heading_path=["K8s 故障手册", heading],
            chunk_index=0,
            score=1.0,
            is_procedural=is_procedural,
            chunk_type=chunk_type,
        ),
        rerank_score=0.9,
        rank_before=1,
        rank_after=1,
    )


def topic_chunk(
    text: str,
    topic: str,
    section: str,
    *,
    document_id: str = "k8s-doc",
    chunk_type: str,
) -> RerankedChunk:
    return RerankedChunk(
        chunk=ScoredChunk(
            chunk_id=f"chunk-{topic}-{section}",
            document_id=document_id,
            document_title="K8s 故障手册",
            text=text,
            heading_path=["K8s 故障手册", topic, section],
            chunk_index=0,
            score=1.0,
            is_procedural=chunk_type == "procedural",
            chunk_type=chunk_type,
        ),
        rerank_score=0.9,
        rank_before=1,
        rank_after=1,
    )


def invocation(
    *, success: bool = True, result: dict[str, Any] | None = None
) -> ToolInvocation:
    return ToolInvocation(
        tool_name="get_pod_status",
        is_write=False,
        arguments={"namespace": "prod", "argument_only": "do-not-cite"},
        success=success,
        result=result,
        error_code=None if success else "RESOURCE_NOT_FOUND",
        error_message=None if success else "not found",
        cache_hit=False,
        idempotent_replay=False,
        elapsed_ms=3,
    )


def successful(*items: ToolInvocation) -> list[tuple[int, ToolInvocation]]:
    return [
        (index, item)
        for index, item in enumerate(items, start=1)
        if item.success and item.result is not None
    ]


def test_plan_schema_only_accepts_atom_id_list_shape() -> None:
    assert VerifiedAnswerPlan.model_validate(plan("K1.A1")).selected_atom_ids == [
        "K1.A1"
    ]
    with pytest.raises(ValueError):
        VerifiedAnswerPlan.model_validate(
            {"selected_atom_ids": ["K1.A1"], "quote": "伪造"}
        )
    with pytest.raises(ValueError):
        VerifiedAnswerPlan.model_validate(
            {"selected_atom_ids": [f"K1.A{i}" for i in range(1, 18)]}
        )


def test_non_procedural_atoms_split_on_markdown_paragraphs_only() -> None:
    source = "第一段第一行。\n第一段折行。\n\n第二段。"
    atoms = enumerate_evidence_atoms([chunk(source)], [])

    assert [item.source_id for item in atoms] == ["K1.A1", "K1.A2"]
    assert atoms[0].exact_quote == "第一段第一行。\n第一段折行。"
    assert atoms[1].exact_quote == "第二段。"
    assert all(item.section == "conclusion" for item in atoms)


def test_procedural_atoms_keep_list_continuations_and_code_blocks() -> None:
    source = (
        "1. 执行检查命令\n"
        "   并查看 Events。\n"
        "   ```sh\n"
        "   kubectl describe pod demo\n"
        "   ```\n"
        "2. 核对配置。"
    )
    atoms = enumerate_evidence_atoms(
        [chunk(source, "处理步骤", chunk_type="procedural")], []
    )

    assert [item.source_id for item in atoms] == ["K1.A1", "K1.A2"]
    assert "kubectl describe pod demo" in atoms[0].rendered_text
    assert atoms[1].rendered_text == "2. 核对配置。"
    assert all(item.section == "evidence_step" for item in atoms)


def test_procedural_text_without_list_boundary_stays_one_atom() -> None:
    source = "先核对事件，再检查资源。\n不要改写命令。"
    atoms = enumerate_evidence_atoms(
        [chunk(source, "处理步骤", chunk_type="procedural")], []
    )
    assert len(atoms) == 1
    assert atoms[0].exact_quote == source


def test_long_mid_chunk_without_boundary_is_not_semantically_rewritten() -> None:
    source = "A" * 1500
    atoms = enumerate_evidence_atoms([chunk(source)], [])
    assert [item.exact_quote for item in atoms] == [source]


def test_atom_ids_are_stable_by_rerank_and_source_order() -> None:
    atoms = enumerate_evidence_atoms(
        [chunk("首段。\n\n次段。", "根因"), chunk("另一来源。", "现象")], []
    )
    assert [item.source_id for item in atoms] == ["K1.A1", "K1.A2", "K2.A1"]


def test_persisted_chunk_type_wins_over_misleading_heading() -> None:
    atoms = enumerate_evidence_atoms(
        [chunk("事实。", "处理步骤", chunk_type="root_cause", is_procedural=True)], []
    )
    assert atoms[0].section == "conclusion"


def test_legacy_chunk_uses_flag_then_shared_heading_inference() -> None:
    flagged = enumerate_evidence_atoms(
        [chunk("1. 检查。", "旧标题", is_procedural=True)], []
    )
    inferred = enumerate_evidence_atoms(
        [chunk("1. 检查。", "排查步骤")], []
    )
    assert flagged[0].section == "evidence_step"
    assert inferred[0].section == "evidence_step"


def test_knowledge_atom_renders_exact_source_with_stable_mapping() -> None:
    quote = "CrashLoopBackOff 的重启间隔会逐次拉长。"
    llm = PlanLLM(plan("K1.A1"))
    result = Answerer(llm).answer("多久恢复", [chunk(quote)], [])

    assert result.status == "verified"
    assert f"{quote} [K1.A1]" in result.text
    assert result.evidence[0].exact_quote == quote
    assert llm.calls == [{"temperature": 0.0, "max_repairs": 0}]


def test_only_steps_is_a_verified_answer_with_fixed_conclusion() -> None:
    source = chunk("1. 查看 Events。", "处理步骤", chunk_type="procedural")
    result = Answerer(PlanLLM(plan("K1.A1"))).answer("如何排查", [source], [])
    assert result.status == "verified"
    assert "当前证据未提供可单独验证的结论" in result.text
    assert "1. 查看 Events。 [K1.A1]" in result.text


def test_format_atoms_exposes_ids_and_text_without_model_authored_fields() -> None:
    atoms = enumerate_evidence_atoms([chunk("逐字原文")], [])
    rendered = format_evidence_atoms(atoms)
    assert "[K1.A1] 类型=事实结论" in rendered
    assert "逐字原文" in rendered
    assert "quote" not in rendered


def test_selector_prompt_rejects_symptom_only_and_unstated_preconditions() -> None:
    llm = PlanLLM(plan("K1.A1"))
    Answerer(llm).answer("口语化故障", [chunk("可验证根因")], [])
    system = llm.messages[0][0]["content"]
    assert "不能只复述现象" in system
    assert "不要假设用户未说明的配置变更" in system


def test_tool_atoms_enumerate_root_containers_and_scalar_leaves_stably() -> None:
    inv = invocation(result={"pod": {"phase": "Running", "ready": True}})
    atoms = enumerate_evidence_atoms([], successful(inv))
    assert [(item.source_id, item.json_pointer) for item in atoms] == [
        ("T1.A1", ""),
        ("T1.A2", "/pod"),
        ("T1.A3", "/pod/phase"),
        ("T1.A4", "/pod/ready"),
    ]
    assert atoms[1].serialized_value == '{"phase":"Running","ready":true}'


def test_tool_pointer_escaping_and_array_order_are_stable() -> None:
    values = _enumerate_tool_values({"a/b~c": ["x", "y"]})
    assert [pointer for pointer, _ in values] == [
        "",
        "/a~1b~0c",
        "/a~1b~0c/0",
        "/a~1b~0c/1",
    ]


def test_large_tool_container_is_omitted_but_small_leaves_remain() -> None:
    values = _enumerate_tool_values({"payload": "x" * 1100, "status": "ok"})
    assert "" not in {pointer for pointer, _ in values}
    assert "/payload" not in {pointer for pointer, _ in values}
    assert ("/status", "ok") in values


def test_tool_atoms_are_limited_per_invocation() -> None:
    values = _enumerate_tool_values({f"field-{index:03d}": index for index in range(100)})
    assert len(values) == 64


def test_failed_tool_and_arguments_are_never_atoms() -> None:
    failed = invocation(success=False, result=None)
    succeeded = invocation(result={"phase": "Running"})
    atoms = enumerate_evidence_atoms([], successful(failed, succeeded))
    assert all("do-not-cite" not in item.rendered_text for item in atoms)
    assert all(item.invocation_index == 2 for item in atoms)


def test_tool_atom_renders_value_and_original_invocation_index() -> None:
    failed = invocation(success=False, result=None)
    succeeded = invocation(result={"phase": "Running"})
    result = Answerer(PlanLLM(plan("T1.A2"))).answer(
        "Pod 状态", [], [failed, succeeded]
    )
    assert "Running [T1.A2]" in result.text
    assert result.evidence[0].json_pointer == "/phase"
    assert result.evidence[0].invocation_index == 2


@pytest.mark.parametrize(
    ("bad_plan", "reason"),
    [
        (plan(), "no_evidence_selected"),
        (plan("K1.A2"), "unknown_atom"),
        (plan("K1.A1", "K1.A1"), "duplicate_atom"),
        (plan("K1"), "unknown_atom"),
    ],
)
def test_invalid_selection_repairs_once_then_falls_back(
    bad_plan: dict[str, Any], reason: str
) -> None:
    result = Answerer(PlanLLM(bad_plan, bad_plan)).answer(
        "问题", [chunk("存在的原文")], []
    )
    assert result.status == "fallback"
    assert result.attempts == 2
    assert result.fallback_reason == reason


def test_semantic_validation_error_is_repaired_once() -> None:
    llm = PlanLLM(plan("K2.A1"), plan("K1.A1"))
    result = Answerer(llm).answer("Pending 是什么", [chunk("Pending 尚未调度。")], [])
    assert result.status == "verified"
    assert result.attempts == 2


def test_diagnostic_question_repairs_symptom_only_selection_once() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("K2.A1"))
    result = Answerer(llm).answer(
        "一直报没权限怎么排查",
        [
            chunk("当前身份没有权限。", "现象", chunk_type="symptom"),
            chunk("RBAC 规则没有允许该操作。", "根因", chunk_type="root_cause"),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert result.evidence[0].source_id == "K2.A1"
    repair_prompt = llm.messages[1][-1]["content"]
    assert "不能只复述现象" in repair_prompt
    assert "[用户问题]\n一直报没权限怎么排查" in repair_prompt
    assert "[K1.A1]" not in repair_prompt
    assert "当前身份没有权限。" not in repair_prompt
    assert "[K2.A1]" in repair_prompt
    assert "RBAC 规则没有允许该操作。" in repair_prompt


def test_diagnostic_selection_rejects_multiple_fault_topics_and_repairs_once() -> None:
    llm = PlanLLM(
        plan("K1.A1", "K3.A1"),
        plan("K3.A1", "K4.A1"),
    )
    result = Answerer(llm).answer(
        "配了转发规则但流量进不来，怎么排查",
        [
            topic_chunk("Service selector 不匹配。", "Service 无端点", "根因", chunk_type="root_cause"),
            topic_chunk("1. 核对 selector。", "Service 无端点", "处理步骤", chunk_type="procedural"),
            topic_chunk("Ingress 控制器未部署。", "Ingress 无法访问", "根因", chunk_type="root_cause"),
            topic_chunk("1. 核对 host/path。", "Ingress 无法访问", "处理步骤", chunk_type="procedural"),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["K3.A1", "K4.A1"]
    assert "只能选择一个故障主题" in llm.messages[1][-1]["content"]


def test_diagnostic_reselects_the_highest_ranked_supported_symptom_topic() -> None:
    llm = PlanLLM(
        plan("K2.A1", "K5.A1"),
        plan("K3.A1", "K4.A1"),
    )
    result = Answerer(llm).answer(
        "为什么一个工作负载能访问同一目标而另一个不能",
        [
            topic_chunk(
                "部分工作负载被策略隔离。",
                "按工作负载生效的访问策略",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "共享目标没有可用后端。",
                "共享目标无后端",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "选择规则只覆盖了受影响的工作负载。",
                "按工作负载生效的访问策略",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "1. 核对选择规则是否覆盖受影响的工作负载。",
                "按工作负载生效的访问策略",
                "处理步骤",
                chunk_type="procedural",
            ),
            topic_chunk(
                "1. 检查共享目标的后端。",
                "共享目标无后端",
                "处理步骤",
                chunk_type="procedural",
            ),
            topic_chunk(
                "1. 检查无关入口规则。",
                "外部入口异常",
                "处理步骤",
                chunk_type="procedural",
            ),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["K3.A1", "K4.A1"]
    repair_prompt = llm.messages[1][-1]["content"]
    assert "最高排序的现象已有同主题解释" in repair_prompt
    assert "[K3.A1]" in repair_prompt
    assert "[K4.A1]" in repair_prompt
    assert "[K2.A1]" not in repair_prompt
    assert "[K5.A1]" not in repair_prompt
    assert "[K6.A1]" not in repair_prompt


def test_diagnostic_accepts_the_supported_symptom_topic_on_first_attempt() -> None:
    result = Answerer(PlanLLM(plan("K2.A1", "K3.A1"))).answer(
        "为什么一个调用方访问同一目标成功而另一个调用方访问失败",
        [
            topic_chunk(
                "只有部分调用方被访问规则拒绝。",
                "按调用方生效的访问策略",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "选择条件只命中了受影响的调用方。",
                "按调用方生效的访问策略",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "1. 核对选择条件与受影响调用方。",
                "按调用方生效的访问策略",
                "处理步骤",
                chunk_type="procedural",
            ),
            topic_chunk(
                "共享目标没有可用后端。",
                "共享目标无后端",
                "根因",
                chunk_type="root_cause",
            ),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 1
    assert [item.source_id for item in result.evidence] == ["K2.A1", "K3.A1"]


def test_causal_question_with_difference_word_still_requires_explanation() -> None:
    llm = PlanLLM(plan("K3.A1", "K4.A1"), plan("K2.A1"))
    result = Answerer(llm).answer(
        "为什么一个调用方能访问同一目标，但另一个不能，结果不同",
        [
            topic_chunk(
                "只有部分调用方被规则拒绝。",
                "按调用方生效的访问策略",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "规则只选择了受影响的调用方。",
                "按调用方生效的访问策略",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "共享目标没有可用后端。",
                "共享目标无后端",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "1. 检查共享目标后端。",
                "共享目标无后端",
                "处理步骤",
                chunk_type="procedural",
            ),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["K2.A1"]


def test_general_diagnostic_does_not_force_the_highest_ranked_symptom_topic() -> None:
    result = Answerer(PlanLLM(plan("K4.A1", "K5.A1"))).answer(
        "早上正常，现在突然连不上内部目标了",
        [
            topic_chunk(
                "部分工作负载被访问策略隔离。",
                "按工作负载生效的访问策略",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "名称解析失败。",
                "内部名称解析失败",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "访问策略只选择了受影响的工作负载。",
                "按工作负载生效的访问策略",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "解析服务没有可用端点。",
                "内部名称解析失败",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "1. 检查解析服务及其端点。",
                "内部名称解析失败",
                "处理步骤",
                chunk_type="procedural",
            ),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 1
    assert [item.source_id for item in result.evidence] == ["K4.A1", "K5.A1"]


def test_invalid_symptom_topic_reselection_falls_back_without_third_attempt() -> None:
    llm = PlanLLM(
        plan("K2.A1", "K5.A1"),
        plan("K2.A1", "K5.A1"),
        plan("K3.A1", "K4.A1"),
    )
    result = Answerer(llm).answer(
        "为什么一个调用方能访问同一目标而另一个失败",
        [
            topic_chunk(
                "部分调用方被策略隔离。",
                "按调用方生效的访问策略",
                "现象",
                chunk_type="symptom",
            ),
            topic_chunk(
                "共享目标没有可用后端。",
                "共享目标无后端",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "规则只选择了受影响的调用方。",
                "按调用方生效的访问策略",
                "根因",
                chunk_type="root_cause",
            ),
            topic_chunk(
                "1. 核对规则选择范围。",
                "按调用方生效的访问策略",
                "处理步骤",
                chunk_type="procedural",
            ),
            topic_chunk(
                "1. 检查共享目标后端。",
                "共享目标无后端",
                "处理步骤",
                chunk_type="procedural",
            ),
        ],
        [],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "unknown_atom"
    assert len(llm.calls) == 2


def test_root_cause_requires_a_procedure_from_the_same_topic() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("K1.A1", "K2.A1"))
    result = Answerer(llm).answer(
        "环境变量取值为空，怎么排查",
        [
            topic_chunk("configMapKeyRef 的 key 不存在。", "ConfigMap 引用失败", "根因", chunk_type="root_cause"),
            topic_chunk("1. 核对引用 key 与 ConfigMap 实际 key。", "ConfigMap 引用失败", "处理步骤", chunk_type="procedural"),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["K1.A1", "K2.A1"]
    repair_prompt = llm.messages[1][-1]["content"]
    assert "同一主题的处理步骤" in repair_prompt
    assert "[K1.A1]" in repair_prompt
    assert "[K2.A1]" in repair_prompt


def test_second_cross_topic_or_missing_procedure_selection_falls_back_without_third_call() -> None:
    llm = PlanLLM(
        plan("K1.A1", "K3.A1"),
        plan("K1.A1", "K3.A1"),
        plan("K3.A1", "K4.A1"),
    )
    result = Answerer(llm).answer(
        "为什么连不上，怎么处理",
        [
            topic_chunk("Service 根因。", "Service", "根因", chunk_type="root_cause"),
            topic_chunk("1. Service 步骤。", "Service", "处理步骤", chunk_type="procedural"),
            topic_chunk("NetworkPolicy 根因。", "NetworkPolicy", "根因", chunk_type="root_cause"),
            topic_chunk("1. NetworkPolicy 步骤。", "NetworkPolicy", "处理步骤", chunk_type="procedural"),
        ],
        [],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "multiple_topics_selected"
    assert len(llm.calls) == 2


def test_symptom_repair_preserves_original_ids_for_root_causes_and_steps() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("K3.A1", "K2.A1"))
    result = Answerer(llm).answer(
        "突然连不上内部服务，怎么排查",
        [
            chunk("内部服务连接失败。", "现象", chunk_type="symptom"),
            chunk(
                "1. 检查 DNS Service 端点。",
                "处理步骤",
                chunk_type="procedural",
            ),
            chunk("CoreDNS 未正常运行会导致解析失败。", "根因", chunk_type="root_cause"),
        ],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["K3.A1", "K2.A1"]
    repair_prompt = llm.messages[1][-1]["content"]
    assert "[K1.A1]" not in repair_prompt
    assert "[K2.A1]" in repair_prompt
    assert "[K3.A1]" in repair_prompt


def test_symptom_repair_rejects_an_atom_excluded_from_second_attempt() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("K1.A1"))
    result = Answerer(llm).answer(
        "一直失败怎么排查",
        [
            chunk("请求失败。", "现象", chunk_type="symptom"),
            chunk("权限规则没有允许该请求。", "根因", chunk_type="root_cause"),
        ],
        [],
    )

    assert result.status == "fallback"
    assert result.attempts == 2
    assert result.fallback_reason == "unknown_atom"
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "second_plan",
    [
        plan(),
        plan("K2.A1", "K2.A1"),
        {"selected_atom_ids": ["not-an-atom"]},
    ],
)
def test_restricted_repair_failure_never_triggers_a_third_attempt(
    second_plan: dict[str, Any],
) -> None:
    llm = PlanLLM(plan("K1.A1"), second_plan, plan("K2.A1"))
    result = Answerer(llm).answer(
        "持续报错怎么处理",
        [
            chunk("持续报错。", "现象", chunk_type="symptom"),
            chunk("配置错误会导致该现象。", "根因", chunk_type="root_cause"),
        ],
        [],
    )

    assert result.status == "fallback"
    assert result.attempts == 2
    assert len(llm.calls) == 2


def test_diagnostic_symptom_is_allowed_when_no_explanatory_atom_exists() -> None:
    llm = PlanLLM(plan("K1.A1"))
    result = Answerer(llm).answer(
        "这个报错是怎么回事",
        [chunk("当前请求返回 Forbidden。", "现象", chunk_type="symptom")],
        [],
    )

    assert result.status == "verified"
    assert result.attempts == 1
    assert len(llm.calls) == 1


def test_successful_tool_result_is_available_in_restricted_repair() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("T1.A2"))
    tool = invocation(result={"phase": "Running"})
    result = Answerer(llm).answer(
        "当前异常怎么排查",
        [chunk("Pod 当前异常。", "现象", chunk_type="symptom")],
        [tool],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert result.evidence[0].source_id == "T1.A2"
    repair_prompt = llm.messages[1][-1]["content"]
    assert "[K1.A1]" not in repair_prompt
    assert "[T1.A2]" in repair_prompt


def test_successful_tool_requires_tool_evidence_and_retries_with_tool_atoms_only() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("T1.A2"))
    tool = invocation(result={"phase": "Running", "reason": "CrashLoopBackOff"})

    result = Answerer(llm).answer(
        "Pod 当前是什么状态",
        [chunk("CrashLoopBackOff 表示容器反复重启。")],
        [tool],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["T1.A2"]
    repair_prompt = llm.messages[1][-1]["content"]
    assert "必须引用成功工具返回的实时结果" in repair_prompt
    assert "[K1.A1]" not in repair_prompt
    assert "[T1.A1]" in repair_prompt
    assert "[T1.A2]" in repair_prompt


def test_successful_tool_summary_is_required_and_retries_with_summary_atom_only() -> None:
    llm = PlanLLM(plan("T1.A1"), plan("T1.A2"))
    tool = invocation(
        result={
            "answer_summary": "prod/api 当前状态为 Running。",
            "phase": "Running",
        }
    )

    result = Answerer(llm).answer("Pod 当前是什么状态", [], [tool])

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["T1.A2"]
    assert "prod/api 当前状态为 Running。 [T1.A2]" in result.text
    repair_prompt = llm.messages[1][-1]["content"]
    assert "服务端生成的工具摘要" in repair_prompt
    assert "[T1.A1]" not in repair_prompt
    assert "[T1.A2]" in repair_prompt
    assert "[T1.A3]" not in repair_prompt


def test_tool_summary_retry_takes_priority_when_initial_selection_is_knowledge_only() -> None:
    llm = PlanLLM(plan("K1.A1"), plan("T1.A2"))
    tool = invocation(
        result={
            "answer_summary": "prod/api 当前状态为 Running。",
            "phase": "Running",
        }
    )

    result = Answerer(llm).answer(
        "Pod 是否正常",
        [chunk("Running 表示容器正在运行。")],
        [tool],
    )

    assert result.status == "verified"
    assert result.attempts == 2
    assert [item.source_id for item in result.evidence] == ["T1.A2"]
    repair_prompt = llm.messages[1][-1]["content"]
    assert "服务端生成的工具摘要" in repair_prompt
    assert "[K1.A1]" not in repair_prompt
    assert "[T1.A1]" not in repair_prompt
    assert "[T1.A2]" in repair_prompt
    assert "[T1.A3]" not in repair_prompt


def test_invalid_tool_summary_repair_falls_back_without_third_attempt() -> None:
    llm = PlanLLM(plan("T1.A1"), plan("T1.A3"), plan("T1.A2"))
    tool = invocation(
        result={
            "answer_summary": "prod/api 当前状态为 Running。",
            "phase": "Running",
        }
    )

    result = Answerer(llm).answer("Pod 当前是什么状态", [], [tool])

    assert result.status == "fallback"
    assert result.attempts == 2
    assert result.fallback_reason == "unknown_atom"
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "second_plan",
    [
        plan("K1.A1"),
        plan(),
        plan("T1.A2", "T1.A2"),
        {"selected_atom_ids": ["not-an-atom"]},
    ],
)
def test_tool_only_repair_failure_never_triggers_a_third_attempt(
    second_plan: dict[str, Any],
) -> None:
    llm = PlanLLM(plan("K1.A1"), second_plan, plan("T1.A2"))
    result = Answerer(llm).answer(
        "Pod 当前是什么状态",
        [chunk("Pod 状态需要结合实时结果判断。")],
        [invocation(result={"phase": "Running"})],
    )

    assert result.status == "fallback"
    assert result.attempts == 2
    assert len(llm.calls) == 2


def test_initial_mixed_or_tool_only_selection_satisfies_tool_requirement() -> None:
    tool = invocation(result={"phase": "Running"})

    mixed = Answerer(PlanLLM(plan("K1.A1", "T1.A2"))).answer(
        "说明当前状态",
        [chunk("Running 表示容器正在运行。")],
        [tool],
    )
    tool_only = Answerer(PlanLLM(plan("T1.A2"))).answer(
        "Pod 当前是什么状态", [], [tool]
    )

    assert mixed.status == "verified"
    assert mixed.attempts == 1
    assert {item.evidence_kind for item in mixed.evidence} == {"knowledge", "tool"}
    assert tool_only.status == "verified"
    assert tool_only.attempts == 1
    assert tool_only.evidence[0].evidence_kind == "tool"


def test_failed_tool_does_not_require_tool_evidence() -> None:
    result = Answerer(PlanLLM(plan("K1.A1"))).answer(
        "为什么查询失败",
        [chunk("目标资源不存在。")],
        [invocation(success=False, result=None)],
    )

    assert result.status == "verified"
    assert result.attempts == 1
    assert result.evidence[0].source_id == "K1.A1"


def test_comparison_question_may_select_symptom_evidence() -> None:
    result = Answerer(PlanLLM(plan("K1.A1"))).answer(
        "这两个状态有什么区别",
        [
            chunk("Pending 表示尚未调度。", "现象", chunk_type="symptom"),
            chunk("调度条件不满足。", "根因", chunk_type="root_cause"),
        ],
        [],
    )
    assert result.status == "verified"
    assert result.attempts == 1


def test_mixed_sources_render_both_atom_markers() -> None:
    inv = invocation(result={"phase": "Running"})
    result = Answerer(PlanLLM(plan("T1.A2", "K1.A1"))).answer(
        "说明当前状态", [chunk("Running 表示容器正在运行。")], [inv]
    )
    assert "Running [T1.A2]" in result.text
    assert "Running 表示容器正在运行。 [K1.A1]" in result.text


@pytest.mark.parametrize(
    ("case_id", "unsupported"),
    [
        ("q02", "重新部署"),
        ("q04", "手动删除 finalizer"),
        ("q05", "kubectl get secret"),
        ("q07", "kubectl rollout restart"),
        ("q16", "kubectl top pod"),
        ("q17", "kubectl describe quota"),
        ("q18", "扩大所有命名空间配额"),
    ],
)
def test_manual_review_regressions_cannot_emit_unsupported_clause(
    case_id: str, unsupported: str
) -> None:
    result = Answerer(PlanLLM(plan("K1.A1"))).answer(
        case_id, [chunk(f"{case_id} 的证据只支持检查已有状态。")], []
    )
    assert result.status == "verified"
    assert unsupported not in result.text


@pytest.mark.parametrize(
    "text",
    [
        "CrashLoopBackOff 按退避策略反复重启，间隔逐次拉长，没有确定恢复时长。",
        "Forbidden 由 RBAC 规则或绑定未授予目标操作导致。",
        "Pending 尚未调度；Waiting 是已调度 Pod 的容器等待状态。",
        "RoleBinding 将 Role 或 ClusterRole 授权给命名空间内的主体。",
        "DNS 故障可能来自 CoreDNS 或 Service/Endpoint 配置。",
        "修改 RBAC 后使用 kubectl auth can-i 验证授权。",
    ],
)
def test_supported_regressions_render_source_atom_exactly(text: str) -> None:
    result = Answerer(PlanLLM(plan("K1.A1"))).answer("问题", [chunk(text)], [])
    assert result.status == "verified"
    assert text in result.text


def test_model_unavailable_returns_controlled_fallback() -> None:
    result = Answerer(PlanLLM(LLMUnavailableError("offline"))).answer(
        "问题", [chunk("证据")], []
    )
    assert result.status == "fallback"
    assert result.attempts == 1
    assert result.fallback_reason == "model_unavailable"


@pytest.mark.parametrize("reason", ["output_truncated", "empty_output"])
def test_protocol_failure_does_not_trigger_second_answer_attempt(reason: str) -> None:
    class ProtocolFailureLLM:
        def __init__(self) -> None:
            self.calls = 0

        def structured(self, messages, schema, **kwargs):  # noqa: ANN001
            self.calls += 1
            raise NonRetryableError("controlled protocol failure", details={"reason": reason})

    llm = ProtocolFailureLLM()
    result = Answerer(llm).answer("为什么失败", [chunk("根因", "原因说明")], [])

    assert llm.calls == 1
    assert result.status == "fallback"
    assert result.attempts == 1
    assert result.fallback_reason == reason


def test_provider_rejection_does_not_trigger_second_answer_attempt() -> None:
    class RejectedLLM:
        def __init__(self) -> None:
            self.calls = 0

        def structured(self, messages, schema, **kwargs):  # noqa: ANN001
            self.calls += 1
            raise NonRetryableError("LLM credentials rejected")

    llm = RejectedLLM()
    result = Answerer(llm).answer("为什么失败", [chunk("根因", "原因说明")], [])

    assert llm.calls == 1
    assert result.status == "fallback"
    assert result.attempts == 1
    assert result.fallback_reason == "model_unavailable"


def test_output_redaction_invalidates_the_entire_verified_mapping() -> None:
    leaked = "sk-1234567890abcdef"
    verified = VerifiedAnswerResult(
        text=f"结论\n- {leaked} [K1.A1]\n\n证据步骤\n- 当前证据未提供可验证步骤。",
        status="verified",
        attempts=1,
        evidence=[
            VerifiedEvidence(
                item_index=1,
                section="conclusion",
                evidence_kind="knowledge",
                source_id="K1.A1",
                rendered_text=leaked,
                chunk_id="chunk-secret",
                citation_label="敏感片段",
                exact_quote=leaked,
            )
        ],
    )
    result = AgentRunResult(
        outcome=AgentOutcome.DIRECT_ANSWER,
        answer=verified.text,
        steps=[],
        verified_answer=verified,
    )

    sanitized = ChatService._sanitize_agent_result(result)

    assert result.outcome == AgentOutcome.INSUFFICIENT_INFORMATION
    assert result.verified_answer is not None
    assert result.verified_answer.status == "fallback"
    assert result.verified_answer.fallback_reason == "output_redacted"
    assert result.verified_answer.evidence == []
    assert result.answer == sanitized.text
    assert leaked not in sanitized.text
    assert sanitized.redactions == ["credential_or_path"]
