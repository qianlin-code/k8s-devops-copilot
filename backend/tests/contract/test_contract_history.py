from fastapi.testclient import TestClient

from app.schemas.history import (
    ConversationDetailResponse,
    ConversationListResponse,
    ToolAuditListResponse,
)
from tests.conftest import API_HEADERS, auth_headers
from tests.contract.test_contract_basics import assert_error_contract
from tests.fakes import ScriptedLLMClient


def _turn(client: TestClient, llm: ScriptedLLMClient, question: str, cid: str | None = None) -> dict:
    llm.queue(
        {"action": "answer", "reasoning": "知识片段足够", "confidence": 0.9},
        {"sufficient": True, "reasoning": "片段命中"},
        f"针对「{question}」的回答内容。",
    )
    payload = {"question": question}
    if cid:
        payload["conversation_id"] = cid
    resp = client.post("/api/v1/chat", headers=API_HEADERS, json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_conversation_list_matches_schema(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    _turn(client, llm, "Pod Pending 怎么处理")
    resp = client.get("/api/v1/conversations", headers=API_HEADERS)
    assert resp.status_code == 200
    body = ConversationListResponse.model_validate(resp.json())
    assert body.total == 1
    assert body.conversations[0].message_count == 2


def test_conversation_list_filters_by_user(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    """不同用户身份现在由 JWT token 区分，而非请求参数。"""
    _turn(client, llm, "Pod Pending 怎么处理")
    other_user_headers = auth_headers("ops-9999")
    resp = client.get("/api/v1/conversations", headers=other_user_headers)
    assert ConversationListResponse.model_validate(resp.json()).total == 0


def test_conversation_detail_carries_trace(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    first = _turn(client, llm, "Pod Pending 怎么处理")
    resp = client.get(
        f"/api/v1/conversations/{first['conversation_id']}",
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    body = ConversationDetailResponse.model_validate(resp.json())
    assert [m.role for m in body.messages] == ["user", "assistant"]
    assistant = body.messages[1]
    assert assistant.trace is not None
    assert assistant.trace["trace_id"] == assistant.trace_id
    assert body.messages[0].trace is None


def test_conversation_detail_can_omit_trace(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    first = _turn(client, llm, "Pod Pending 怎么处理")
    resp = client.get(
        f"/api/v1/conversations/{first['conversation_id']}",
        headers=API_HEADERS,
        params={"include_trace": "false"},
    )
    body = ConversationDetailResponse.model_validate(resp.json())
    assert all(m.trace is None for m in body.messages)


def test_multi_turn_context_summarization(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    """超出滑动窗口后应触发摘要降级，并在 trace 中体现。"""
    first = _turn(client, llm, "第 1 个问题")
    cid = first["conversation_id"]
    for i in range(2, 9):
        _turn(client, llm, f"第 {i} 个问题", cid)

    llm.queue_summary("历史摘要：用户连续询问 Pod 调度与重启相关问题。")
    llm.queue(
        {"action": "answer", "reasoning": "结合摘要可答", "confidence": 0.9},
        {"sufficient": True, "reasoning": "足够"},
        "综合此前对话给出的回答。",
    )
    resp = client.post(
        "/api/v1/chat",
        headers=API_HEADERS,
        json={"question": "第 9 个问题", "conversation_id": cid},
    )
    assert resp.status_code == 200, resp.text
    context = resp.json()["trace"]["context"]
    assert context["total_turns"] >= 8
    assert context["summarized"] is True
    assert context["summary_source_turns"] > 0
    assert context["windowed_turns"] <= context["total_turns"]
    assert context["summary"]


def test_conversation_detail_unknown_id(client: TestClient) -> None:
    resp = client.get("/api/v1/conversations/missing-id", headers=API_HEADERS)
    assert resp.status_code == 404
    assert assert_error_contract(resp.json()).code == "RESOURCE_NOT_FOUND"


def test_tool_audit_records_cache_hit(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    for _ in range(2):
        llm.queue(
            {
                "action": "call_tool",
                "reasoning": "查 Pod 状态",
                "confidence": 0.9,
                "tool_name": "get_pod_status",
                "tool_arguments": {"namespace": "ops-demo", "name": "api-gateway-7f9c"},
            },
            {"sufficient": True, "reasoning": "已取得状态"},
            "该 Pod 当前处于 Pending 状态。",
        )
        client.post(
            "/api/v1/chat",
            headers=API_HEADERS,
            json={"question": "查一下 ops-demo 下 api-gateway-7f9c 状态"},
        )

    resp = client.get(
        "/api/v1/tool-audits",
        headers=API_HEADERS,
        params={"tool_name": "get_pod_status"},
    )
    assert resp.status_code == 200
    body = ToolAuditListResponse.model_validate(resp.json())
    assert body.total == 2
    assert all(item.is_write is False for item in body.items)
    assert all(item.success for item in body.items)


def test_tool_audit_filters_by_conversation(
    client: TestClient, llm: ScriptedLLMClient, seeded_kb: str
) -> None:
    llm.queue(
        {
            "action": "call_tool",
            "reasoning": "查 Pod",
            "confidence": 0.9,
            "tool_name": "get_pod_status",
            "tool_arguments": {"namespace": "ops-demo", "name": "api-gateway-7f9c"},
        },
        {"sufficient": True, "reasoning": "够了"},
        "回答。",
    )
    body = client.post(
        "/api/v1/chat",
        headers=API_HEADERS,
        json={"question": "查 ops-demo 下的 Pod"},
    ).json()

    matched = client.get(
        "/api/v1/tool-audits",
        headers=API_HEADERS,
        params={"conversation_id": body["conversation_id"]},
    ).json()
    assert ToolAuditListResponse.model_validate(matched).total == 1

    other = client.get(
        "/api/v1/tool-audits",
        headers=API_HEADERS,
        params={"conversation_id": "some-other-id"},
    ).json()
    assert ToolAuditListResponse.model_validate(other).total == 0
