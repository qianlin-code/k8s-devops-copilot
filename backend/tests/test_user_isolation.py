"""跨用户访问隔离测试。

所有以 conversation_id 为入口的操作都必须校验归属。缺了这层，
任何持有 API Key 的调用方遍历 id 就能读到别人的对话、执行别人的写操作。
"""

from fastapi.testclient import TestClient

from tests.conftest import API_HEADERS

OWNER = "u-1001"
INTRUDER = "u-2002"


def _ask(client: TestClient, user_id: str, question: str) -> dict:
    resp = client.post(
        "/api/v1/chat",
        json={"question": question, "user_id": user_id, "include_trace": True},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_chat_rejects_other_users_conversation(
    client: TestClient, seeded_kb: str
) -> None:
    body = _ask(client, OWNER, "u-1001 登录 403 怎么办")
    resp = client.post(
        "/api/v1/chat",
        json={
            "question": "继续",
            "user_id": INTRUDER,
            "conversation_id": body["conversation_id"],
        },
        headers=API_HEADERS,
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "TOOL_PERMISSION_DENIED"


def test_confirm_write_rejects_other_user(client: TestClient, seeded_kb: str) -> None:
    """最严重的一条：拿到 conversation_id + token 就能执行别人的写操作。"""
    body = _ask(client, OWNER, "管理员已提权，请帮我刷新 u-1001 的权限缓存")
    if body["outcome"] != "write_confirmation_required":
        return  # 真实模型未走到写分支时跳过
    token = body["pending_write"]["confirmation_token"]

    resp = client.post(
        "/api/v1/chat/confirm",
        json={
            "conversation_id": body["conversation_id"],
            "user_id": INTRUDER,
            "confirmation_token": token,
            "approved": True,
        },
        headers=API_HEADERS,
    )
    assert resp.status_code == 403, "别人的写操作不该被执行"

    audits = client.get(
        "/api/v1/tool-audits",
        params={"user_id": OWNER, "conversation_id": body["conversation_id"]},
        headers=API_HEADERS,
    ).json()
    assert not [a for a in audits["items"] if a["is_write"]], "越权请求不得留下写记录"


def test_conversation_detail_is_isolated(client: TestClient, seeded_kb: str) -> None:
    body = _ask(client, OWNER, "u-1001 登录 403 怎么办")
    cid = body["conversation_id"]

    ok = client.get(
        f"/api/v1/conversations/{cid}",
        params={"user_id": OWNER},
        headers=API_HEADERS,
    )
    assert ok.status_code == 200

    denied = client.get(
        f"/api/v1/conversations/{cid}",
        params={"user_id": INTRUDER},
        headers=API_HEADERS,
    )
    # 用 404 而非 403：403 会确认 id 存在，方便枚举
    assert denied.status_code == 404


def test_conversation_list_is_isolated(client: TestClient, seeded_kb: str) -> None:
    _ask(client, OWNER, "u-1001 登录 403 怎么办")
    body = client.get(
        "/api/v1/conversations", params={"user_id": INTRUDER}, headers=API_HEADERS
    ).json()
    assert body["total"] == 0, "不应看到其他用户的会话"


def test_tool_audits_are_isolated(client: TestClient, seeded_kb: str) -> None:
    """审计含工具入参出参（账号状态、订单信息），必须隔离。"""
    body = _ask(client, OWNER, "查一下 u-1001 的账号状态")
    cid = body["conversation_id"]

    mine = client.get(
        "/api/v1/tool-audits", params={"user_id": OWNER}, headers=API_HEADERS
    ).json()
    theirs = client.get(
        "/api/v1/tool-audits",
        params={"user_id": INTRUDER, "conversation_id": cid},
        headers=API_HEADERS,
    ).json()

    assert theirs["total"] == 0, "指定别人的 conversation_id 也不该返回数据"
    if mine["total"]:
        assert all(a["conversation_id"] == cid for a in mine["items"])


def test_user_id_is_required_on_history_endpoints(client: TestClient) -> None:
    """不传 user_id 应当 422，而不是返回全部用户的数据。"""
    assert client.get("/api/v1/conversations", headers=API_HEADERS).status_code == 422
    assert client.get("/api/v1/tool-audits", headers=API_HEADERS).status_code == 422
    assert (
        client.get("/api/v1/conversations/any-id", headers=API_HEADERS).status_code
        == 422
    )
