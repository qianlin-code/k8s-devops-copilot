"""跨用户访问隔离测试。

所有以 conversation_id 为入口的操作都必须校验归属。缺了这层，
任何持有 API Key 的调用方遍历 id 就能读到别人的对话、执行别人的写操作。

JWT 切换后，user_id 由 token 决定而非请求体参数，用不同 token 模拟不同用户。
"""

from fastapi.testclient import TestClient

from tests.conftest import auth_headers

OWNER = "ops-1"
INTRUDER = "ops-2"


def _ask(client: TestClient, user_id: str, question: str) -> dict:
    resp = client.post(
        "/api/v1/chat",
        json={"question": question, "include_trace": True},
        headers=auth_headers(user_id),  # 用不同 user_id 的 token 模拟不同用户
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_chat_rejects_other_users_conversation(
    client: TestClient, seeded_kb: str
) -> None:
    body = _ask(client, OWNER, "ops-demo 命名空间下 Pod Pending 怎么办")
    resp = client.post(
        "/api/v1/chat",
        json={
            "question": "继续",
            "conversation_id": body["conversation_id"],
        },
        headers=auth_headers(INTRUDER),  # 入侵者用自己的 token
    )
    # 用 404 而非 403：403 会确认「这个 id 存在但不属于你」，方便枚举。
    # 与 /conversations/{id} 保持同一语义同一状态码，前端无需兼容两种。
    assert resp.status_code == 404
    assert resp.json()["code"] == "RESOURCE_NOT_FOUND"


def test_confirm_write_rejects_other_user(client: TestClient, seeded_kb: str) -> None:
    """最严重的一条：拿到 conversation_id + token 就能执行别人的写操作。"""
    body = _ask(client, OWNER, "请帮我重启 ops-demo 命名空间下的 billing-sync Deployment")
    if body["outcome"] != "write_confirmation_required":
        return  # 真实模型未走到写分支时跳过
    token = body["pending_write"]["confirmation_token"]

    resp = client.post(
        "/api/v1/chat/confirm",
        json={
            "conversation_id": body["conversation_id"],
            "confirmation_token": token,
            "approved": True,
        },
        headers=auth_headers(INTRUDER),  # 入侵者用自己的 token
    )
    assert resp.status_code == 404, "别人的写操作不该被执行"

    audits = client.get(
        "/api/v1/tool-audits",
        params={"conversation_id": body["conversation_id"]},
        headers=auth_headers(OWNER),  # Owner 查自己的审计
    ).json()
    assert not [a for a in audits["items"] if a["is_write"]], "越权请求不得留下写记录"


def test_conversation_detail_is_isolated(client: TestClient, seeded_kb: str) -> None:
    body = _ask(client, OWNER, "ops-demo 命名空间下 Pod Pending 怎么办")
    cid = body["conversation_id"]

    ok = client.get(
        f"/api/v1/conversations/{cid}",
        headers=auth_headers(OWNER),
    )
    assert ok.status_code == 200

    denied = client.get(
        f"/api/v1/conversations/{cid}",
        headers=auth_headers(INTRUDER),
    )
    # 用 404 而非 403：403 会确认 id 存在，方便枚举
    assert denied.status_code == 404


def test_mark_sedimentation_rejects_other_users_conversation(
    client: TestClient, seeded_kb: str
) -> None:
    """标记沉淀也以 conversation_id 为入口，必须校验归属。

    待审队列条目会完整带出原对话的 question/answer，且 `/knowledge/sedimentations`
    是管理台视角（不按用户过滤）。若标记不校验归属，任何持有 API Key 的调用方
    都能把别人的会话推进队列，再从队列里读到别人的对话内容。
    """
    body = _ask(client, OWNER, "ops-demo 命名空间下 Pod Pending 怎么办")
    cid = body["conversation_id"]

    denied = client.post(
        "/api/v1/knowledge/sedimentations",
        json={"conversation_id": cid},  # marked_by 从 JWT token 获取
        headers=auth_headers(INTRUDER),  # 入侵者用自己的 token
    )
    assert denied.status_code == 404, "不该能标记别人的会话"
    assert denied.json()["code"] == "RESOURCE_NOT_FOUND"

    # 确认没有任何条目被写进待审队列
    queue = client.get(
        "/api/v1/knowledge/sedimentations",
        params={"status": "pending"},
        headers=auth_headers("admin-1", role="admin"),  # admin 才能列待审队列
    ).json()
    assert queue["total"] == 0, "越权标记不得留下待审条目"

    # 本人标记仍然正常
    ok = client.post(
        "/api/v1/knowledge/sedimentations",
        json={"conversation_id": cid},  # marked_by 从 JWT token 获取
        headers=auth_headers(OWNER),
    )
    assert ok.status_code == 200


def test_conversation_list_is_isolated(client: TestClient, seeded_kb: str) -> None:
    _ask(client, OWNER, "ops-demo 命名空间下 Pod Pending 怎么办")
    body = client.get(
        "/api/v1/conversations", headers=auth_headers(INTRUDER)
    ).json()
    assert body["total"] == 0, "不应看到其他用户的会话"


def test_tool_audits_are_isolated(client: TestClient, seeded_kb: str) -> None:
    """审计含工具入参出参（Pod 状态、Deployment 信息），必须隔离。"""
    body = _ask(client, OWNER, "查一下 ops-demo 下 api-gateway-7f9c 这个 Pod 的状态")
    cid = body["conversation_id"]

    mine = client.get(
        "/api/v1/tool-audits", headers=auth_headers(OWNER)
    ).json()
    theirs = client.get(
        "/api/v1/tool-audits",
        params={"conversation_id": cid},
        headers=auth_headers(INTRUDER),
    ).json()

    assert theirs["total"] == 0, "指定别人的 conversation_id 也不该返回数据"
    if mine["total"]:
        assert all(a["conversation_id"] == cid for a in mine["items"])


def test_user_id_is_required_on_history_endpoints(client: TestClient) -> None:
    """JWT 切换后：user_id 从 token 解析，不再需要传参数，测试验证鉴权生效。"""
    assert client.get("/api/v1/conversations").status_code == 401
    assert client.get("/api/v1/tool-audits").status_code == 401
    assert client.get("/api/v1/conversations/any-id").status_code == 401
