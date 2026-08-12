"""生产环境配置护栏测试。

宁可起不来，也不要带着弱密钥或本地 CORS 上线 —— 这类问题一旦漏到线上，
外部就能直接调用全部接口。
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import DEFAULT_DEV_JWT_SECRET, Settings
from tests.conftest import API_HEADERS

_STRONG_JWT_SECRET = "prod-jwt-secret-with-sufficient-length-01"
_PROD_ORIGINS = ["https://copilot.example.com"]


def _settings(**overrides) -> Settings:
    base = {
        "environment": "prod",
        "jwt_secret_key": _STRONG_JWT_SECRET,
        "cors_allow_origins": _PROD_ORIGINS,
        "embedding_provider": "ollama",
        "llm_provider": "ollama",
    }
    return Settings(**{**base, **overrides})


def test_prod_accepts_strong_config() -> None:
    settings = _settings()
    assert settings.jwt_secret_key == _STRONG_JWT_SECRET


def test_prod_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="development default"):
        _settings(jwt_secret_key=DEFAULT_DEV_JWT_SECRET)


def test_prod_rejects_short_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(jwt_secret_key="short-secret")


def test_prod_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match=r"must not contain"):
        _settings(cors_allow_origins=["*"])


def test_prod_rejects_localhost_cors() -> None:
    with pytest.raises(ValidationError, match="local origins"):
        _settings(cors_allow_origins=["http://localhost:5173"])


def test_dev_allows_defaults() -> None:
    """开发环境不应被这些约束卡住。"""
    settings = Settings(
        environment="dev",
        embedding_provider="ollama",
        llm_provider="ollama",
    )
    assert "localhost" in settings.cors_allow_origins[0]


def test_health_hides_topology_in_prod(monkeypatch) -> None:
    """未鉴权端点不得泄露模型 provider 与集合名。"""
    from app.config import get_settings

    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", _STRONG_JWT_SECRET)
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://copilot.example.com"]')
    get_settings.cache_clear()
    try:
        from app.main import create_app

        with TestClient(create_app()) as client:
            body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["environment"] == "prod"
        assert body["llm_provider"] is None
        assert body["embedding_provider"] is None
        assert body["collection_name"] is None
    finally:
        get_settings.cache_clear()


def test_health_exposes_topology_in_dev(client: TestClient) -> None:
    """开发环境保留这些字段，前端侧边栏要展示。"""
    body = client.get("/api/v1/health").json()
    assert body["llm_provider"]
    assert body["collection_name"]


def test_cors_preflight_allows_authorization_header(client: TestClient) -> None:
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_readiness_still_requires_jwt(client: TestClient) -> None:
    assert client.get("/api/v1/readiness").status_code == 401
    assert client.get("/api/v1/readiness", headers=API_HEADERS).status_code == 200


def test_prod_startup_skips_mock_data_seeding(tmp_path, monkeypatch) -> None:
    """生产环境启动不应把演示用的 Pod/Deployment 灌进真实数据库。

    _check_database() 无条件跑 seed_mock_data() 曾是个真实漏洞：生产库若是
    全新的，会插入演示 Pod/Deployment。这里用一个全新的空库
    （_isolate autouse fixture 已经在 dev 环境下 seed 过默认库，不能复用）
    直接跑一次 _check_database()，断言 mock_accounts 表在 prod 下仍是空的。
    """
    from sqlalchemy import select

    from app.config import get_settings
    from app.rag import bm25_index, vector_store
    from app.startup_checks import _check_database
    from app.storage.db import reset_engine, session_scope
    from app.storage.models import MockPod

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'prod.db').as_posix()}")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", _STRONG_JWT_SECRET)
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://copilot.example.com"]')
    get_settings.cache_clear()
    reset_engine()
    vector_store.reset_vector_store()
    bm25_index.reset_bm25_index()
    try:
        detail = _check_database()
        assert "skipped(prod)" in detail
        with session_scope() as session:
            assert session.scalar(select(MockPod).limit(1)) is None
    finally:
        get_settings.cache_clear()
        reset_engine()
        vector_store.reset_vector_store()
        bm25_index.reset_bm25_index()
