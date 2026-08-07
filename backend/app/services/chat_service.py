import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context_manager import ContextBundle, ConversationContextManager
from app.agent.state_machine import (
    AgentOutcome,
    AgentRunResult,
    AgentStateMachine,
    AgentStep,
    PendingWriteAction,
)
from app.agent.tools.base import ToolContext
from app.config import get_settings
from app.errors import ErrorCode, NonRetryableError, NotFoundError
from app.rag.retriever import RetrievalResult, Retriever
from app.schemas.base import to_utc_iso
from app.schemas.chat import ChatResponse, PendingWriteActionSchema
from app.schemas.progress import ProgressEvent
from app.security.input_guard import guard_user_input
from app.security.output_guard import sanitize_output
from app.storage.models import Conversation, Message, MessageRole
from app.tracing.trace_builder import build_execution_trace


# 状态机节点名 -> 面向用户的阶段描述
_STEP_LABELS = {
    "route": "正在判断该直接回答还是调用工具",
    "execute_tool": "正在调用工具查询系统状态",
    "execute_confirmed_write": "正在执行已确认的写操作",
    "await_write_confirmation": "等待确认写操作",
    "verify_sufficiency": "正在校验信息是否充分",
    "generate_answer": "正在生成回答",
    "skip_already_executed_call": "跳过已执行的调用",
    "skip_failed_call": "跳过已失败的调用",
    "max_steps_exceeded": "已达最大步数，正在收尾",
}


@dataclass(slots=True)
class _Turn:
    conversation: Conversation
    context: ContextBundle
    retrieval: RetrievalResult
    agent: AgentRunResult
    input_flags: list[str]
    output_redactions: list[str]
    elapsed_ms: int


class ChatService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        agent: AgentStateMachine,
        context_manager: ConversationContextManager,
    ) -> None:
        self._retriever = retriever
        self._agent = agent
        self._context = context_manager

    def ask(
        self,
        session: Session,
        *,
        question: str,
        user_id: str,
        conversation_id: str | None,
        trace_id: str,
        include_trace: bool,
    ) -> ChatResponse:
        started = time.perf_counter()
        guarded = guard_user_input(question)

        conversation, context = self._open_turn(
            session, conversation_id, user_id, guarded.text, trace_id
        )

        retrieval = self._retriever.retrieve(
            guarded.text, history_snippet=context.history_snippet or None
        )
        result = self._agent.run(
            guarded.text,
            retrieval.chunks,
            ToolContext(
                session=session,
                trace_id=trace_id,
                user_id=user_id,
                conversation_id=conversation.id,
            ),
            context_messages=context.messages or None,
        )

        sanitized = sanitize_output(result.answer)
        turn = _Turn(
            conversation=conversation,
            context=context,
            retrieval=retrieval,
            agent=result,
            input_flags=guarded.flags,
            output_redactions=sanitized.redactions,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return self._persist_and_render(
            session, turn, sanitized.text, trace_id, include_trace
        )

    def ask_streaming(
        self,
        session: Session,
        *,
        question: str,
        user_id: str,
        conversation_id: str | None,
        trace_id: str,
        include_trace: bool,
        emit: Callable[[ProgressEvent], None],
    ) -> ChatResponse:
        """与 ask() 同一条链路，额外在每个阶段回调 emit 推送进展。

        本地 7B 模型下一轮对话要串联 4 次 LLM 调用（改写/路由/校验/生成），
        总耗时 20-40s。不推进展的话前端只能干等，会被误判为卡死。
        """
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        def send(phase: str, label: str, **detail: object) -> None:
            emit(
                ProgressEvent(
                    phase=phase,  # type: ignore[arg-type]
                    label=label,
                    elapsed_ms=elapsed(),
                    detail=detail,
                )
            )

        guarded = guard_user_input(question)
        send("guarded", "输入安全检查通过", flags=guarded.flags)

        # 与 ask() 共用开场短事务：建会话 + 落用户消息后立即提交，
        # 之后的长耗时段不持有写锁
        conversation, context = self._open_turn(
            session, conversation_id, user_id, guarded.text, trace_id
        )
        send("accepted", "会话已建立", conversation_id=conversation.id)
        send(
            "context_built",
            "多轮上下文已组装",
            total_turns=context.total_turns,
            windowed_turns=context.windowed_turns,
            summarized=context.summarized,
        )

        send("retrieving", "正在检索知识库")
        retrieval = self._retriever.retrieve(
            guarded.text, history_snippet=context.history_snippet or None
        )
        send(
            "retrieved",
            f"检索到 {len(retrieval.chunks)} 条相关片段",
            chunk_count=len(retrieval.chunks),
            rerank_applied=retrieval.rerank_applied,
        )

        def on_step(step: AgentStep) -> None:
            send(
                "agent_step",
                _STEP_LABELS.get(step.node, step.node),
                node=step.node,
                step=step.step,
            )

        result = self._agent.run(
            guarded.text,
            retrieval.chunks,
            ToolContext(
                session=session,
                trace_id=trace_id,
                user_id=user_id,
                conversation_id=conversation.id,
            ),
            context_messages=context.messages or None,
            on_step=on_step,
        )

        sanitized = sanitize_output(result.answer)
        turn = _Turn(
            conversation=conversation,
            context=context,
            retrieval=retrieval,
            agent=result,
            input_flags=guarded.flags,
            output_redactions=sanitized.redactions,
            elapsed_ms=elapsed(),
        )
        return self._persist_and_render(
            session, turn, sanitized.text, trace_id, include_trace
        )

    def confirm_write(
        self,
        session: Session,
        *,
        conversation_id: str,
        user_id: str,
        confirmation_token: str,
        approved: bool,
        trace_id: str,
        include_trace: bool,
    ) -> ChatResponse:
        started = time.perf_counter()
        # 必须校验归属：少了这层，任何人拿到 conversation_id + token
        # 就能执行别人会话里的写操作（重置缓存、创建工单）。
        conversation = self._load_owned_conversation(session, conversation_id, user_id)

        pending, question = self._load_pending_write(session, conversation_id, confirmation_token)

        if not approved:
            answer = f"已取消该操作（{pending.description}），未对系统做任何修改。"
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT.value,
                    content=answer,
                    trace_id=trace_id,
                    trace_payload={"outcome": "write_rejected", "tool": pending.tool_name},
                )
            )
            session.flush()
            message = self._latest_assistant(session, conversation.id)
            return ChatResponse(
                conversation_id=conversation.id,
                message_id=message.id,
                outcome="write_rejected",
                answer=answer,
                pending_write=None,
                trace=None,
                created_at=to_utc_iso(message.created_at),
            )

        history = self._load_history(session, conversation.id)
        context = self._context.build(history, existing_summary=conversation.summary)
        retrieval = self._retriever.retrieve(
            question, history_snippet=context.history_snippet or None
        )
        result = self._agent.run(
            question,
            retrieval.chunks,
            ToolContext(
                session=session,
                trace_id=trace_id,
                user_id=user_id,
                conversation_id=conversation.id,
            ),
            context_messages=context.messages or None,
            confirmed_write=pending,
        )
        sanitized = sanitize_output(result.answer)
        turn = _Turn(
            conversation=conversation,
            context=context,
            retrieval=retrieval,
            agent=result,
            input_flags=[],
            output_redactions=sanitized.redactions,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return self._persist_and_render(
            session, turn, sanitized.text, trace_id, include_trace
        )

    def _persist_and_render(
        self,
        session: Session,
        turn: _Turn,
        answer: str,
        trace_id: str,
        include_trace: bool,
    ) -> ChatResponse:
        pending_schema: PendingWriteActionSchema | None = None
        if turn.agent.pending_write is not None:
            pending_schema = PendingWriteActionSchema(
                tool_name=turn.agent.pending_write.tool_name,
                description=turn.agent.pending_write.description,
                arguments=turn.agent.pending_write.arguments,
                reasoning=turn.agent.pending_write.reasoning,
                confirmation_token=uuid.uuid4().hex,
            )

        trace = build_execution_trace(
            trace_id=trace_id,
            total_elapsed_ms=turn.elapsed_ms,
            context=turn.context,
            retrieval=turn.retrieval,
            agent=turn.agent,
            input_flags=turn.input_flags,
            output_redactions=turn.output_redactions,
            agent_max_steps=get_settings().agent_max_steps,
        )
        payload = trace.model_dump(mode="json")
        if pending_schema is not None:
            payload["pending_write"] = pending_schema.model_dump(mode="json")

        session.add(
            Message(
                conversation_id=turn.conversation.id,
                role=MessageRole.ASSISTANT.value,
                content=answer,
                trace_id=trace_id,
                trace_payload=payload,
            )
        )
        session.flush()
        message = self._latest_assistant(session, turn.conversation.id)

        return ChatResponse(
            conversation_id=turn.conversation.id,
            message_id=message.id,
            outcome=turn.agent.outcome.value,
            answer=answer,
            pending_write=pending_schema,
            trace=trace if include_trace else None,
            created_at=to_utc_iso(message.created_at),
        )

    def _open_turn(
        self,
        session: Session,
        conversation_id: str | None,
        user_id: str,
        question: str,
        trace_id: str,
    ) -> tuple[Conversation, ContextBundle]:
        """开场写入：建会话 + 落用户消息，作为一个短事务立即提交。

        必须提交后才能进入检索 + Agent 循环（20-40s）。否则写事务在整个
        长耗时段一直持有 SQLite 写锁，并发请求即使有 busy_timeout 也等不到 ——
        实测中前一个请求还在跑，下一个请求在 INSERT conversations 处
        等满 10s 然后抛 database is locked。
        """
        conversation = self._resolve_conversation(
            session, conversation_id, user_id, question
        )
        history = self._load_history(session, conversation.id)
        context = self._context.build(history, existing_summary=conversation.summary)
        if context.summarized and context.summary:
            conversation.summary = context.summary
        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER.value,
                content=question,
                trace_id=trace_id,
            )
        )
        session.commit()
        return conversation, context

    def _resolve_conversation(
        self,
        session: Session,
        conversation_id: str | None,
        user_id: str,
        first_question: str,
    ) -> Conversation:
        if conversation_id is None:
            conversation = Conversation(
                id=str(uuid.uuid4()), user_id=user_id, title=first_question[:80]
            )
            session.add(conversation)
            session.flush()
            return conversation
        return self._load_owned_conversation(session, conversation_id, user_id)

    def _load_owned_conversation(
        self, session: Session, conversation_id: str, user_id: str
    ) -> Conversation:
        """按 id 取会话并校验归属。所有以 conversation_id 为入口的操作都要过这里。"""
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(
                f"Conversation '{conversation_id}' not found",
                details={"conversation_id": conversation_id},
            )
        if conversation.user_id != user_id:
            raise NonRetryableError(
                "Conversation belongs to a different user",
                code=ErrorCode.TOOL_PERMISSION_DENIED,
                http_status=403,
            )
        return conversation

    def _load_history(self, session: Session, conversation_id: str) -> list[Message]:
        return list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            )
        )

    def _load_pending_write(
        self, session: Session, conversation_id: str, token: str
    ) -> tuple[PendingWriteAction, str]:
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(10)
            )
        )
        for msg in messages:
            payload = msg.trace_payload or {}
            pending = payload.get("pending_write")
            if not pending or pending.get("confirmation_token") != token:
                continue
            question = next(
                (
                    m.content
                    for m in messages
                    if m.role == MessageRole.USER.value and m.id < msg.id
                ),
                "",
            )
            return (
                PendingWriteAction(
                    tool_name=pending["tool_name"],
                    arguments=pending["arguments"],
                    description=pending["description"],
                    reasoning=pending["reasoning"],
                ),
                question,
            )
        raise NonRetryableError(
            "Confirmation token is invalid or has expired",
            code=ErrorCode.VALIDATION_FAILED,
            details={"conversation_id": conversation_id},
        )

    def _latest_assistant(self, session: Session, conversation_id: str) -> Message:
        message = session.scalar(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.ASSISTANT.value,
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
        assert message is not None
        return message
