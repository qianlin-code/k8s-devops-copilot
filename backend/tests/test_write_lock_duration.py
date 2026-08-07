"""写锁持有时长测试。

原实现在 flush() 开启写事务后才跑检索 + Agent 循环（20-40s），
全程攥着 SQLite 写锁 —— 并发请求即使有 busy_timeout 也等不到。
现在改为进入长耗时段前先 commit。
"""

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.storage.db import get_engine, session_scope
from tests.conftest import API_HEADERS


def _wal_has_uncommitted_writer() -> bool:
    """另开一个连接尝试立即拿写锁，拿不到说明有别的写事务在持有。"""
    with get_engine().connect() as probe:
        probe.exec_driver_sql("PRAGMA busy_timeout=0")
        try:
            probe.exec_driver_sql("BEGIN IMMEDIATE")
            probe.exec_driver_sql("ROLLBACK")
            return False
        except Exception:
            return True


def test_user_message_is_committed_before_long_work(
    client: TestClient, seeded_kb: str, monkeypatch
) -> None:
    """检索开始时用户消息应已落库、写锁应已释放。"""
    observed: dict[str, object] = {}

    from app.rag import retriever as retriever_module

    original = retriever_module.Retriever.retrieve

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        # 检索发生在长耗时段的入口，此刻不该还持有写锁
        observed["locked_during_retrieval"] = _wal_has_uncommitted_writer()
        with session_scope() as s:
            observed["messages_visible"] = s.execute(
                text("SELECT COUNT(*) FROM messages")
            ).scalar()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(retriever_module.Retriever, "retrieve", spy)

    resp = client.post(
        "/api/v1/chat",
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200

    assert observed["locked_during_retrieval"] is False, (
        "进入检索时仍持有写锁——长耗时段不应攥着 SQLite 写锁"
    )
    assert observed["messages_visible"] == 1, "用户消息应已提交，其他连接可见"


def test_streaming_also_commits_before_long_work(
    client: TestClient, seeded_kb: str, monkeypatch
) -> None:
    """流式接口走同一条链路，同样不能长时间持锁。"""
    observed: dict[str, bool] = {}

    from app.rag import retriever as retriever_module

    original = retriever_module.Retriever.retrieve

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        observed["locked"] = _wal_has_uncommitted_writer()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(retriever_module.Retriever, "retrieve", spy)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
        headers=API_HEADERS,
    ) as resp:
        resp.read()

    assert observed["locked"] is False


def test_new_conversation_insert_survives_concurrent_writer(
    client: TestClient, seeded_kb: str
) -> None:
    """另有写事务在持锁时，新建会话仍应成功（靠 busy_timeout 排队等待）。

    实测崩溃点就在 `INSERT INTO conversations` —— 前一个长请求持锁，
    这个 INSERT 等满 10s 后抛 database is locked。
    """
    import threading
    import time

    from sqlalchemy import text as sql_text

    hold = 1.5
    ready = threading.Event()

    def holder() -> None:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            conn.execute(
                sql_text(
                    "INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
                    "VALUES ('lock-holder','u-9999','hold',datetime('now'),datetime('now'))"
                )
            )
            ready.set()
            time.sleep(hold)
            conn.exec_driver_sql("ROLLBACK")

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    started = time.perf_counter()
    resp = client.post(
        "/api/v1/chat",
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
        headers=API_HEADERS,
    )
    waited = time.perf_counter() - started
    thread.join(timeout=5)

    assert resp.status_code == 200, f"并发写下新建会话失败：{resp.text[:200]}"
    assert waited >= hold * 0.5, "应当等待锁释放而非立即失败"


def test_streaming_survives_concurrent_writer(
    client: TestClient, seeded_kb: str
) -> None:
    """流式接口在并发写下同样不能崩。

    浏览器场景里 health 轮询与对话请求并发，正是这条路径先暴露问题的。
    """
    import threading
    import time

    from sqlalchemy import text as sql_text

    ready = threading.Event()

    def holder() -> None:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            conn.execute(
                sql_text(
                    "INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
                    "VALUES ('lock-holder-2','u-9999','hold',datetime('now'),datetime('now'))"
                )
            )
            ready.set()
            time.sleep(1.5)
            conn.exec_driver_sql("ROLLBACK")

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
        headers=API_HEADERS,
    ) as resp:
        raw = resp.read().decode("utf-8")
    thread.join(timeout=5)

    events = []
    for block in raw.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("event: "):
                events.append(line[7:])
    assert "error" not in events, f"并发写下流式接口报错：{raw[:400]}"
    assert events and events[-1] == "done"
    assert "database is locked" not in raw


def test_audit_does_not_hold_lock_through_remaining_steps(
    client: TestClient, seeded_kb: str, monkeypatch, llm
) -> None:
    """工具审计写入后必须提交，不能只 flush。

    审计发生在 Agent 循环中段，后面还有多次 LLM 调用。只 flush 会开启写事务
    却不释放锁，等于又把写锁攥到整轮结束 —— 并发对话下第二个请求的
    审计 INSERT 就会撞锁。这是并发压测里唯一残留的 database is locked。
    """
    from app.agent.tools import executor as executor_module

    observed: list[bool] = []
    original = executor_module.ToolExecutor._audit

    def spy(self, inv, ctx, *, request_id):  # noqa: ANN001, ANN002
        original(self, inv, ctx, request_id=request_id)
        # 审计落库后应已释放写锁，后续 LLM 调用期间别人能写
        observed.append(_wal_has_uncommitted_writer())

    monkeypatch.setattr(executor_module.ToolExecutor, "_audit", spy)

    # 显式编排走工具分支，否则替身可能直接作答、测不到审计
    llm.queue(
        {
            "action": "call_tool",
            "reasoning": "需要查账号状态",
            "confidence": 0.9,
            "tool_name": "get_account_status",
            "tool_arguments": {"user_id": "u-1001"},
        },
        {"sufficient": True, "reasoning": "已取得状态"},
        "账号 u-1001 的权限等级为 restricted。",
    )

    resp = client.post(
        "/api/v1/chat",
        json={"question": "查一下 u-1001 的账号状态", "user_id": "u-1001"},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    assert observed, "本轮应至少有一次工具调用"
    assert not any(observed), "审计写入后仍持有写锁——应 commit 而非 flush"


def test_conversation_survives_after_commit(client: TestClient, seeded_kb: str) -> None:
    """中途 commit 后仍能正常写入助手消息并返回完整响应。"""
    body = client.post(
        "/api/v1/chat",
        json={"question": "u-1001 登录 403 怎么办", "user_id": "u-1001"},
        headers=API_HEADERS,
    ).json()

    detail = client.get(
        f"/api/v1/conversations/{body['conversation_id']}",
        params={"user_id": "u-1001"},
        headers=API_HEADERS,
    ).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"], f"消息应完整落库，实际 {roles}"
