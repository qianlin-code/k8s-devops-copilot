from dataclasses import dataclass, field

from app.config import get_settings
from app.errors import AppError
from app.llm.client import LLMClient
from app.storage.models import Message, MessageRole

_SUMMARY_SYSTEM = (
    "你是对话摘要器。把下面的运维排查对话压缩成要点摘要，"
    "保留：涉及的命名空间/Pod/Deployment、已确认的故障现象、已执行过的操作及结果、尚未解决的问题。"
    "丢弃寒暄与重复内容。不要编造未出现的信息。控制在 300 字以内。"
)


@dataclass(slots=True)
class ContextBundle:
    messages: list[dict[str, str]]
    summary: str | None
    total_turns: int
    windowed_turns: int
    summarized: bool
    summary_source_turns: int = 0
    degrade_reason: str | None = None
    history_snippet: str = ""
    stages: list[str] = field(default_factory=list)


class ConversationContextManager:
    """滑动窗口保留最近 N 轮；超窗时把更早的历史交给 LLM 压成摘要。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def build(
        self,
        history: list[Message],
        *,
        existing_summary: str | None = None,
        window_turns: int | None = None,
        enable_summary: bool | None = None,
    ) -> ContextBundle:
        settings = get_settings()
        window = window_turns or settings.context_window_turns
        allow_summary = (
            settings.enable_context_summary if enable_summary is None else enable_summary
        )
        stages: list[str] = []

        dialog = [
            m
            for m in history
            if m.role in (MessageRole.USER.value, MessageRole.ASSISTANT.value)
        ]
        total_turns = _count_turns(dialog)
        keep = window * 2
        recent = dialog[-keep:] if keep < len(dialog) else dialog
        overflow = dialog[: len(dialog) - len(recent)]

        summary = existing_summary
        summarized = False
        degrade_reason: str | None = None

        if overflow:
            stages.append(f"window_overflow:{len(overflow)}_messages")
            if not allow_summary:
                degrade_reason = "summary_disabled"
            elif self._llm is None:
                degrade_reason = "no_llm_client"
            else:
                generated = self._summarize(overflow, existing_summary)
                if generated is None:
                    degrade_reason = "summary_failed_truncated"
                else:
                    summary, summarized = generated, True
                    stages.append("summary_generated")
        else:
            stages.append("window_fits")

        messages: list[dict[str, str]] = []
        if summary:
            messages.append(
                {"role": "system", "content": f"[历史对话摘要]\n{summary}"}
            )
        messages.extend({"role": m.role, "content": m.content} for m in recent)

        return ContextBundle(
            messages=messages,
            summary=summary,
            total_turns=total_turns,
            windowed_turns=_count_turns(recent),
            summarized=summarized,
            summary_source_turns=_count_turns(overflow),
            degrade_reason=degrade_reason,
            history_snippet=_snippet(recent),
            stages=stages,
        )

    def _summarize(
        self, overflow: list[Message], previous_summary: str | None
    ) -> str | None:
        assert self._llm is not None
        transcript = "\n".join(f"{m.role}: {m.content}" for m in overflow)
        if previous_summary:
            transcript = f"[已有摘要]\n{previous_summary}\n\n[新增对话]\n{transcript}"
        try:
            text = self._llm.chat(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": transcript},
                ],
                temperature=0.1,
                max_tokens=500,
            )
        except AppError:
            return None
        cleaned = text.strip()
        return cleaned or None


def _count_turns(messages: list[Message] | list[dict[str, str]]) -> int:
    return sum(
        1
        for m in messages
        if (m.role if isinstance(m, Message) else m["role"]) == MessageRole.USER.value
    )


def _snippet(messages: list[Message], limit: int = 400) -> str:
    parts = [f"{m.role}: {m.content}" for m in messages[-4:]]
    text = "\n".join(parts)
    return text[-limit:] if len(text) > limit else text
