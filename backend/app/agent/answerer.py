import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError

from app.agent.tools.executor import ToolInvocation
from app.errors import AppError
from app.llm.client import LLMClient
from app.rag.chunking.markdown_chunker import infer_chunk_type
from app.rag.reranker import RerankedChunk
from app.schemas.base import StrictBaseModel

_ATOM_ID_PATTERN = r"^[KT][1-9][0-9]*\.A[1-9][0-9]*$"
_LIST_ITEM = re.compile(r"^[ \t]*(?:[0-9]+[.)]|[-*+])[ \t]+")
_MAX_SELECTED_ATOMS = 16
_MAX_TOOL_ATOMS_PER_INVOCATION = 64
_MAX_TOOL_ATOM_BYTES = 1024

AtomId = Annotated[str, Field(pattern=_ATOM_ID_PATTERN, max_length=32)]


class VerifiedAnswerPlan(StrictBaseModel):
    selected_atom_ids: list[AtomId] = Field(
        default_factory=list,
        max_length=_MAX_SELECTED_ATOMS,
    )


@dataclass(slots=True)
class EvidenceAtom:
    source_id: str
    section: Literal["conclusion", "evidence_step"]
    evidence_kind: Literal["knowledge", "tool"]
    rendered_text: str
    semantic_type: Literal["symptom", "root_cause", "procedure", "fact", "tool_result"]
    topic_id: str | None = None
    chunk_id: str | None = None
    citation_label: str | None = None
    exact_quote: str | None = None
    tool_name: str | None = None
    invocation_index: int | None = None
    json_pointer: str | None = None
    serialized_value: str | None = None


@dataclass(slots=True)
class VerifiedEvidence:
    item_index: int
    section: Literal["conclusion", "evidence_step"]
    evidence_kind: Literal["knowledge", "tool"]
    source_id: str
    rendered_text: str
    chunk_id: str | None = None
    citation_label: str | None = None
    exact_quote: str | None = None
    tool_name: str | None = None
    invocation_index: int | None = None
    json_pointer: str | None = None
    serialized_value: str | None = None


@dataclass(slots=True)
class VerifiedAnswerResult:
    text: str
    status: Literal["verified", "fallback"]
    attempts: int
    evidence: list[VerifiedEvidence] = field(default_factory=list)
    fallback_reason: str | None = None


class _EvidenceValidationError(ValueError):
    def __init__(
        self, code: str, message: str, *, retry_topic_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_topic_id = retry_topic_id


_SYSTEM = """你是 Kubernetes 运维证据选择器。服务端已经把可引用内容拆成了证据原子。
你不能撰写回答、复制原文、生成命令或 JSON Pointer；只能选择与问题直接相关的原子 ID。

规则：
- 只从输入中实际存在的 K#.A# 或 T#.A# 中选择。
- 按回答问题时应展示的顺序选择；不要重复 ID。
- 事实、现象、根因和工具字段可作为结论；操作步骤只能作为证据步骤。
- 同一故障主题同时提供现象与根因/步骤时，选择能解释或排查问题的根因/步骤，不能只复述现象。
- 排障问题一次只能选择一个故障主题；确定主题后同时选择该主题的根因和可用处理步骤。
- 当问题明确比较多个主体访问同一目标且一个成功、另一个失败时，若最高排序的现象原子存在同主题根因或步骤，必须沿用该现象主题。
- 其他排障问题由问题中的明确线索选择最相关的单一主题；不要假设用户未说明的配置变更。
- 不要根据常识补全来源未提供的命令、时长、原因、状态或建议。
- 若证据只能回答问题的一部分，选择能直接支持该部分的原子，不要因其他部分无证据而返回空。
- 只有全部原子都与问题无关时才返回空数组。

只返回一个 JSON 对象，不要返回 Markdown 或解释：
{"selected_atom_ids":["K1.A1","K2.A1"]}
"""

_INSUFFICIENT_TEMPLATE = (
    "抱歉，根据现有的知识库内容和系统数据，我无法准确回答这个问题。\n\n"
    "{detail}\n\n"
    "建议：{suggestion}"
)


class Answerer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def answer(
        self,
        question: str,
        chunks: list[RerankedChunk],
        invocations: list[ToolInvocation],
        *,
        context_messages: list[dict[str, str]] | None = None,
    ) -> VerifiedAnswerResult:
        successful = [
            (index, invocation)
            for index, invocation in enumerate(invocations, start=1)
            if invocation.success and invocation.result is not None
        ]
        atoms = enumerate_evidence_atoms(chunks, successful)
        base_messages = [{"role": "system", "content": _SYSTEM}]
        if context_messages:
            base_messages.extend(context_messages)
        messages = [*base_messages]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"[可选择证据原子]\n{format_evidence_atoms(atoms)}\n\n"
                    f"[用户问题]\n{question}"
                ),
            }
        )

        last_reason = "structured_output_invalid"
        retry_topic_id: str | None = None
        active_atoms = atoms
        for attempt in range(1, 3):
            try:
                plan = self._llm.structured(
                    messages, VerifiedAnswerPlan, temperature=0.0, max_repairs=0
                )
                evidence = _verify_plan(
                    plan,
                    active_atoms,
                    require_explanation=_requires_explanatory_evidence(question),
                    require_symptom_anchor=_requires_differential_anchor(question),
                    require_tool_evidence=bool(successful),
                )
                return VerifiedAnswerResult(
                    text=_render_answer(evidence),
                    status="verified",
                    attempts=attempt,
                    evidence=evidence,
                )
            except AppError as exc:
                retry_topic_id = None
                if exc.retryable:
                    return self._fallback("model_unavailable", attempt)
                if "validation_error" not in exc.details:
                    reason = str(exc.details.get("reason") or "model_unavailable")
                    return self._fallback(reason, attempt)
                validation_message = "structured output did not match the required schema"
                last_reason = "structured_output_invalid"
            except ValidationError as exc:
                retry_topic_id = None
                validation_message = "structured output did not match the required schema"
                last_reason = _fallback_reason_for_schema_error(exc)
            except _EvidenceValidationError as exc:
                validation_message = str(exc)
                last_reason = exc.code
                retry_topic_id = exc.retry_topic_id
            except ValueError as exc:
                retry_topic_id = None
                validation_message = str(exc)
                last_reason = "evidence_validation_failed"
            if attempt == 1:
                retry_atoms = _retry_atoms_for_reason(
                    atoms, last_reason, retry_topic_id=retry_topic_id
                )
                active_atoms = retry_atoms
                messages = [*base_messages]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "服务端拒绝了上一次选择："
                            f"{validation_message[:300]}。"
                            f"{_repair_instruction(last_reason)}"
                            "只返回 JSON：{\"selected_atom_ids\":[]}。\n\n"
                            f"[可选择证据原子]\n{format_evidence_atoms(retry_atoms)}\n\n"
                            f"[用户问题]\n{question}"
                        ),
                    }
                )

        return self._fallback(last_reason, 2)

    def _fallback(self, reason: str, attempts: int) -> VerifiedAnswerResult:
        return VerifiedAnswerResult(
            text=self.insufficient_answer(
                ["回答证据未能通过逐项校验"],
                "请补充可核验的信息后重试，或提交工单转人工处理。",
            ),
            status="fallback",
            attempts=attempts,
            fallback_reason=reason,
        )

    def insufficient_answer(
        self, missing: list[str], suggestion: str | None
    ) -> str:
        detail = (
            "缺少以下关键信息：\n" + "\n".join(f"- {item}" for item in missing)
            if missing
            else "现有资料与您的问题相关性不足。"
        )
        return _INSUFFICIENT_TEMPLATE.format(
            detail=detail,
            suggestion=suggestion or "补充更多细节后重新提问，或提交工单转人工处理。",
        )


def enumerate_evidence_atoms(
    chunks: list[RerankedChunk],
    successful_invocations: list[tuple[int, ToolInvocation]],
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    topic_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    for source_index, reranked in enumerate(chunks, start=1):
        chunk_type = _chunk_type(reranked)
        procedural = chunk_type == "procedural"
        topic_key = _knowledge_topic_key(reranked)
        topic_id = None
        if topic_key is not None:
            topic_id = topic_ids.setdefault(topic_key, f"P{len(topic_ids) + 1}")
        spans = (
            _split_procedural_atoms(reranked.chunk.text)
            if procedural
            else _split_paragraph_atoms(reranked.chunk.text)
        )
        for atom_index, span in enumerate(spans, start=1):
            atoms.append(
                EvidenceAtom(
                    source_id=f"K{source_index}.A{atom_index}",
                    section="evidence_step" if procedural else "conclusion",
                    evidence_kind="knowledge",
                    rendered_text=span,
                    semantic_type=(
                        "procedure"
                        if procedural
                        else chunk_type
                        if chunk_type in {"symptom", "root_cause"}
                        else "fact"
                    ),
                    topic_id=topic_id,
                    chunk_id=reranked.chunk.chunk_id,
                    citation_label=reranked.chunk.citation_label(),
                    exact_quote=span,
                )
            )

    for source_index, (invocation_index, invocation) in enumerate(
        successful_invocations, start=1
    ):
        assert invocation.result is not None
        for atom_index, (pointer, serialized) in enumerate(
            _enumerate_tool_values(invocation.result), start=1
        ):
            atoms.append(
                EvidenceAtom(
                    source_id=f"T{source_index}.A{atom_index}",
                    section="conclusion",
                    evidence_kind="tool",
                    rendered_text=serialized,
                    semantic_type="tool_result",
                    tool_name=invocation.tool_name,
                    invocation_index=invocation_index,
                    json_pointer=pointer,
                    serialized_value=serialized,
                )
            )
    return atoms


def _split_paragraph_atoms(text: str) -> list[str]:
    return _split_source_spans(text, list_items=False)


def _split_procedural_atoms(text: str) -> list[str]:
    spans = _split_source_spans(text, list_items=True)
    return spans or ([text.strip()] if text.strip() else [])


def _split_source_spans(text: str, *, list_items: bool) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text] if text else []

    starts = [0]
    offset = 0
    in_code_block = False
    saw_list_item = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
        if index and not in_code_block:
            if list_items and _LIST_ITEM.match(line):
                starts.append(offset)
                saw_list_item = True
            elif not list_items and not line.strip() and lines[index - 1].strip():
                starts.append(offset + len(line))
        if list_items and index == 0 and _LIST_ITEM.match(line):
            saw_list_item = True
        offset += len(line)

    if list_items and not saw_list_item:
        return [text.strip()] if text.strip() else []

    starts = sorted(set(position for position in starts if position < len(text)))
    spans: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        span = text[start:end].strip()
        if span:
            spans.append(span)
    return spans


def _enumerate_tool_values(document: Any) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def visit(value: Any, pointer: str) -> None:
        serialized = _serialize_tool_value(value)
        if len(serialized.encode("utf-8")) <= _MAX_TOOL_ATOM_BYTES:
            candidates.append((pointer, serialized))
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{pointer}/{_escape_json_pointer(str(key))}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")

    visit(document, "")
    return candidates[:_MAX_TOOL_ATOMS_PER_INVOCATION]


def _escape_json_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _serialize_tool_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _verify_plan(
    plan: VerifiedAnswerPlan,
    atoms: list[EvidenceAtom],
    *,
    require_explanation: bool = False,
    require_symptom_anchor: bool = False,
    require_tool_evidence: bool = False,
) -> list[VerifiedEvidence]:
    if not plan.selected_atom_ids:
        raise _EvidenceValidationError(
            "no_evidence_selected", "the plan selected no evidence atoms"
        )
    if len(set(plan.selected_atom_ids)) != len(plan.selected_atom_ids):
        raise _EvidenceValidationError(
            "duplicate_atom", "the plan selected duplicate evidence atoms"
        )
    atom_map = {atom.source_id: atom for atom in atoms}
    selected: list[EvidenceAtom] = []
    for source_id in plan.selected_atom_ids:
        if not re.fullmatch(_ATOM_ID_PATTERN, source_id):
            raise _EvidenceValidationError(
                "unknown_atom", f"invalid evidence atom id {source_id}"
            )
        atom = atom_map.get(source_id)
        if atom is None:
            raise _EvidenceValidationError(
                "unknown_atom", f"unknown evidence atom {source_id}"
            )
        selected.append(atom)

    summary_atoms = [
        item
        for item in atoms
        if item.evidence_kind == "tool" and item.json_pointer == "/answer_summary"
    ]
    if summary_atoms and not any(
        item.evidence_kind == "tool" and item.json_pointer == "/answer_summary"
        for item in selected
    ):
        raise _EvidenceValidationError(
            "missing_tool_summary",
            "a deterministic tool answer summary must be selected when available",
        )

    if require_tool_evidence and not any(
        item.evidence_kind == "tool" for item in selected
    ):
        raise _EvidenceValidationError(
            "missing_tool_evidence",
            "a successful tool invocation requires at least one tool evidence atom",
        )

    if require_explanation:
        selected_topic_ids = {
            item.topic_id
            for item in selected
            if item.evidence_kind == "knowledge" and item.topic_id is not None
        }
        if len(selected_topic_ids) > 1:
            raise _EvidenceValidationError(
                "multiple_topics_selected",
                "a diagnostic answer must select exactly one knowledge fault topic",
            )

        anchor_topic_id = (
            _diagnostic_anchor_topic(atoms) if require_symptom_anchor else None
        )
        if (
            anchor_topic_id is not None
            and selected_topic_ids
            and anchor_topic_id not in selected_topic_ids
            and not any(item.semantic_type == "tool_result" for item in selected)
        ):
            raise _EvidenceValidationError(
                "symptom_topic_mismatch",
                "the selected fault topic does not match the highest-ranked supported symptom",
                retry_topic_id=anchor_topic_id,
            )

    if (
        require_explanation
        and selected
        and all(item.semantic_type == "symptom" for item in selected)
        and any(
            item.semantic_type in {"root_cause", "procedure", "tool_result"}
            for item in atoms
        )
    ):
        raise _EvidenceValidationError(
            "symptom_only_selection",
            "diagnostic question selected symptoms without explanatory evidence",
        )

    if require_explanation:
        selected_root_topics = {
            item.topic_id
            for item in selected
            if item.semantic_type == "root_cause" and item.topic_id is not None
        }
        for topic_id in selected_root_topics:
            has_available_procedure = any(
                item.topic_id == topic_id and item.semantic_type == "procedure"
                for item in atoms
            )
            selected_procedure = any(
                item.topic_id == topic_id and item.semantic_type == "procedure"
                for item in selected
            )
            if has_available_procedure and not selected_procedure:
                raise _EvidenceValidationError(
                    "missing_procedure_evidence",
                    "a selected root cause requires a procedure from the same topic",
                    retry_topic_id=topic_id,
                )

    ordered = [
        *[item for item in selected if item.section == "conclusion"],
        *[item for item in selected if item.section == "evidence_step"],
    ]
    return [
        VerifiedEvidence(
            item_index=index,
            section=atom.section,
            evidence_kind=atom.evidence_kind,
            source_id=atom.source_id,
            rendered_text=atom.rendered_text,
            chunk_id=atom.chunk_id,
            citation_label=atom.citation_label,
            exact_quote=atom.exact_quote,
            tool_name=atom.tool_name,
            invocation_index=atom.invocation_index,
            json_pointer=atom.json_pointer,
            serialized_value=atom.serialized_value,
        )
        for index, atom in enumerate(ordered, start=1)
    ]


def _fallback_reason_for_schema_error(exc: ValidationError) -> str:
    atom_id_error_types = {"string_pattern_mismatch", "string_too_long"}
    for error in exc.errors():
        location = error.get("loc", ())
        if (
            location
            and location[0] == "selected_atom_ids"
            and error.get("type") in atom_id_error_types
        ):
            return "unknown_atom"
    return "structured_output_invalid"


def _retry_atoms_for_reason(
    atoms: list[EvidenceAtom],
    reason: str,
    *,
    retry_topic_id: str | None = None,
) -> list[EvidenceAtom]:
    if reason in {"symptom_only_selection", "multiple_topics_selected"}:
        return [
            atom
            for atom in atoms
            if atom.semantic_type in {"root_cause", "procedure", "tool_result"}
        ]
    if (
        reason in {"missing_procedure_evidence", "symptom_topic_mismatch"}
        and retry_topic_id is not None
    ):
        return [
            atom
            for atom in atoms
            if atom.topic_id == retry_topic_id
            and atom.semantic_type in {"root_cause", "procedure"}
        ]
    if reason == "missing_tool_evidence":
        return [atom for atom in atoms if atom.evidence_kind == "tool"]
    if reason == "missing_tool_summary":
        return [
            atom
            for atom in atoms
            if atom.evidence_kind == "tool"
            and atom.json_pointer == "/answer_summary"
        ]
    return atoms


def _repair_instruction(reason: str) -> str:
    if reason == "no_evidence_selected":
        return "只要任一原子能回答问题的一部分，就选择该原子。"
    if reason == "duplicate_atom":
        return "每个原子 ID 最多选择一次。"
    if reason == "unknown_atom":
        return "只能选择输入中实际列出的 K#.A# 或 T#.A#。"
    if reason == "symptom_only_selection":
        return "这是排障问题，不能只复述现象；至少选择一个根因、步骤或工具结果原子。"
    if reason == "multiple_topics_selected":
        return "这是排障问题，只能选择一个故障主题，并选择该主题的根因和处理步骤。"
    if reason == "missing_procedure_evidence":
        return "已选择根因时，必须同时选择输入中同一主题的处理步骤。"
    if reason == "symptom_topic_mismatch":
        return "最高排序的现象已有同主题解释；只能从该现象主题中选择根因和处理步骤。"
    if reason == "missing_tool_evidence":
        return "本轮已有成功工具调用，回答必须引用成功工具返回的实时结果原子。"
    if reason == "missing_tool_summary":
        return "本轮已有服务端生成的工具摘要，回答必须引用对应摘要原子。"
    return "严格按 JSON Schema 返回，不要增加字段或解释。"


def _chunk_type(chunk: RerankedChunk) -> str | None:
    stored_type = chunk.chunk.chunk_type
    if stored_type is not None:
        return stored_type
    if chunk.chunk.is_procedural:
        return "procedural"
    inferred, _ = infer_chunk_type(chunk.chunk.heading_path)
    return inferred


def _is_procedural_chunk(chunk: RerankedChunk) -> bool:
    return _chunk_type(chunk) == "procedural"


def _knowledge_topic_key(
    chunk: RerankedChunk,
) -> tuple[str, tuple[str, ...]] | None:
    heading_path = chunk.chunk.heading_path
    if not chunk.chunk.document_id or len(heading_path) < 3:
        return None
    return (
        chunk.chunk.document_id,
        tuple(part.strip().casefold() for part in heading_path[:-1]),
    )


def _diagnostic_anchor_topic(atoms: list[EvidenceAtom]) -> str | None:
    """Use the highest-ranked symptom only when its topic has explanatory evidence."""
    explanatory_topics = {
        atom.topic_id
        for atom in atoms
        if atom.topic_id is not None
        and atom.semantic_type in {"root_cause", "procedure"}
    }
    for atom in atoms:
        if atom.semantic_type == "symptom" and atom.topic_id in explanatory_topics:
            return atom.topic_id
    return None


def _requires_explanatory_evidence(question: str) -> bool:
    normalized = question.casefold()
    if any(marker in normalized for marker in ("为什么", "为何", "原因")):
        return True
    if any(marker in normalized for marker in ("区别", "不同", "什么是", "表现", "现象")):
        return False
    return any(
        marker in normalized
        for marker in (
            "怎么",
            "如何",
            "为什么",
            "原因",
            "排查",
            "报错",
            "失败",
            "异常",
            "不能",
            "无法",
            "连不上",
            "没权限",
        )
    )


def _requires_differential_anchor(question: str) -> bool:
    normalized = question.casefold()
    has_shared_target = any(marker in normalized for marker in ("同一", "相同"))
    has_contrast = any(
        marker in normalized
        for marker in ("但", "而", "另一个", "有的", "其他", "别人", "同事")
    )
    has_success = any(
        marker in normalized for marker in ("能访问", "能连", "成功", "正常", "可以")
    )
    has_failure = any(
        marker in normalized
        for marker in ("不能", "无法", "失败", "不通", "连不上", "报错")
    )
    return has_shared_target and has_contrast and has_success and has_failure


def _render_answer(evidence: list[VerifiedEvidence]) -> str:
    conclusions = [item for item in evidence if item.section == "conclusion"]
    steps = [item for item in evidence if item.section == "evidence_step"]

    lines = ["结论"]
    if conclusions:
        lines.extend(
            f"- {item.rendered_text} [{item.source_id}]" for item in conclusions
        )
    else:
        lines.append("- 当前证据未提供可单独验证的结论。")
    lines.extend(["", "证据步骤"])
    if steps:
        lines.extend(f"- {item.rendered_text} [{item.source_id}]" for item in steps)
    else:
        lines.append("- 当前证据未提供可验证步骤。")
    return "\n".join(lines)


def format_evidence_atoms(atoms: list[EvidenceAtom]) -> str:
    if not atoms:
        return "(无可选择证据原子)"
    rendered: list[str] = []
    for atom in atoms:
        section = "事实结论" if atom.section == "conclusion" else "证据步骤"
        if atom.evidence_kind == "knowledge":
            origin = atom.citation_label or "知识库"
        else:
            origin = f"{atom.tool_name} result{atom.json_pointer or '/'}"
        rendered.append(
            f"[{atom.source_id}] 类型={section} 角色={atom.semantic_type} "
            f"主题={atom.topic_id or 'none'} 来源={origin}\n"
            f"{atom.rendered_text}"
        )
    return "\n\n".join(rendered)


def format_knowledge(chunks: list[RerankedChunk]) -> str:
    if not chunks:
        return "(无相关片段)"
    return "\n\n".join(
        f"[K{index}] 来源: {item.chunk.citation_label()}\n{item.chunk.text}"
        for index, item in enumerate(chunks, start=1)
    )


def format_tool_sources(invocations: list[tuple[int, ToolInvocation]]) -> str:
    if not invocations:
        return "(无成功工具结果)"
    return "\n".join(
        f"[T{index}] 工具: {invocation.tool_name}\n"
        f"result={json.dumps(invocation.result, ensure_ascii=False, sort_keys=True)}"
        for index, (_, invocation) in enumerate(invocations, start=1)
    )
