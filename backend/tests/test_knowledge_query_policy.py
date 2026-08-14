from dataclasses import dataclass

import pytest

from app.rag.query_policy import (
    KNOWLEDGE_QUERY_MIN_SCORE,
    is_knowledge_only_question,
    min_rerank_score_for_query,
)


@pytest.mark.parametrize(
    "question",
    [
        "CrashLoopBackOff 要等多久恢复",
        "改完权限绑定要重启什么东西才生效吗",
    ],
)
def test_knowledge_questions_use_relaxed_threshold(question: str) -> None:
    assert is_knowledge_only_question(question)
    assert min_rerank_score_for_query(question, production_score=0.12) == pytest.approx(
        KNOWLEDGE_QUERY_MIN_SCORE
    )
    assert KNOWLEDGE_QUERY_MIN_SCORE == pytest.approx(0.03)


@pytest.mark.parametrize(
    "question",
    [
        "查一下 ops-demo 下 api-gateway-7f9c 这个 Pod 现在的状态",
        "帮我看看 ops-demo 下 worker-queue 这个 Deployment 副本够不够",
        "请在 ops-demo 创建告警工单，标题是调度器异常排查",
        "帮我重启 ops-demo 下的 worker-queue Deployment",
    ],
)
def test_live_and_write_questions_keep_production_threshold(question: str) -> None:
    assert not is_knowledge_only_question(question)
    assert min_rerank_score_for_query(question, production_score=0.12) == pytest.approx(
        0.12
    )


@dataclass
class _Retrieval:
    chunks: list


def test_chat_service_passes_policy_threshold_to_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.chat_service import ChatService

    calls: list[dict] = []

    class Retriever:
        def retrieve(self, question: str, **kwargs):
            calls.append({"question": question, **kwargs})
            return _Retrieval(chunks=[])

    class Context:
        history_snippet = ""
        messages = []

    class Conversation:
        id = "conversation-1"

    class Agent:
        def run(self, *args, **kwargs):
            raise AssertionError("retrieval policy should be tested before agent execution")

    service = ChatService(retriever=Retriever(), agent=Agent(), context_manager=Context())
    monkeypatch.setattr(
        service, "_open_turn", lambda *args, **kwargs: (Conversation(), Context())
    )

    with pytest.raises(AssertionError):
        service.ask(
            object(),
            question="CrashLoopBackOff 要等多久恢复",
            user_id="ops-1",
            conversation_id=None,
            trace_id="trace-1",
            include_trace=False,
        )

    assert calls[0]["min_score"] == pytest.approx(0.03)
