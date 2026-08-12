from fastapi.testclient import TestClient

from app.schemas.common import ErrorResponse, HealthResponse, ReadinessResponse
from tests.conftest import API_HEADERS

ERROR_KEYS = {"code", "message", "trace_id", "retryable", "details"}


def assert_error_contract(payload: dict) -> ErrorResponse:
    assert set(payload) == ERROR_KEYS, f"error shape drifted: {sorted(payload)}"
    assert "Traceback" not in payload["message"]
    assert payload["trace_id"], "trace_id must always be present"
    return ErrorResponse.model_validate(payload)


def test_health_matches_schema(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    HealthResponse.model_validate(resp.json())


def test_readiness_matches_schema(client: TestClient) -> None:
    resp = client.get("/api/v1/readiness", headers=API_HEADERS)
    assert resp.status_code == 200
    body = ReadinessResponse.model_validate(resp.json())
    assert {c.name for c in body.checks} >= {"config", "database", "vector_store"}


def test_missing_jwt_returns_unified_error(client: TestClient) -> None:
    """JWT 切换后：缺少 Authorization header 应返回 401。"""
    resp = client.get("/api/v1/conversations")
    assert resp.status_code == 401
    error = assert_error_contract(resp.json())
    assert error.code == "UNAUTHORIZED"
    assert error.retryable is False


def test_invalid_jwt_returns_unified_error(client: TestClient) -> None:
    """JWT 切换后：无效 token 应返回 401。"""
    resp = client.get("/api/v1/conversations", headers={"Authorization": "Bearer invalid-token"})
    assert resp.status_code == 401
    assert assert_error_contract(resp.json()).code == "UNAUTHORIZED"


def test_unknown_route_returns_unified_error(client: TestClient) -> None:
    resp = client.get("/api/v1/not-a-route", headers=API_HEADERS)
    assert resp.status_code == 404
    assert assert_error_contract(resp.json()).code == "RESOURCE_NOT_FOUND"


def test_extra_field_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/chat",
        headers=API_HEADERS,
        json={"question": "hi", "rogue_field": 1},
    )
    assert resp.status_code == 422
    error = assert_error_contract(resp.json())
    assert error.code == "VALIDATION_FAILED"
    assert any(
        v["type"] == "extra_forbidden" for v in error.details["violations"]
    ), error.details


def test_missing_required_field_is_rejected(client: TestClient) -> None:
    """user_id 已从 JWT token 获取，不再是必填字段。测试其他必填字段缺失。"""
    resp = client.post("/api/v1/chat", headers=API_HEADERS, json={})
    assert resp.status_code == 422
    assert assert_error_contract(resp.json()).code == "VALIDATION_FAILED"


def test_trace_id_header_echoed(client: TestClient) -> None:
    resp = client.get("/api/v1/health", headers={"X-Trace-Id": "fixed-trace-123"})
    assert resp.headers["X-Trace-Id"] == "fixed-trace-123"


def test_openapi_exposes_expected_paths(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/api/v1/chat",
        "/api/v1/chat/confirm",
        "/api/v1/conversations",
        "/api/v1/conversations/{conversation_id}",
        "/api/v1/knowledge/documents",
        "/api/v1/knowledge/sedimentations",
        "/api/v1/tool-audits",
        "/api/v1/health",
        "/api/v1/readiness",
    } <= paths
