"""生产环境配置护栏测试。

宁可起不来，也不要带着弱密钥或本地 CORS 上线 —— 这类问题一旦漏到线上，
外部就能直接调用全部接口。
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import DEFAULT_DEV_API_KEY, Settings
from tests.conftest import API_HEADERS

_STRONG_KEY = "prod-key-with-sufficient-length-01"
_PROD_ORIGINS = ["https://copilot.example.com"]


def _settings(**overrides) -> Settings:
    base = {
        "environment": "prod",
        "api_key": _STRONG_KEY,
        "cors_allow_origins": _PROD_ORIGINS,
        "embedding_provider": "ollama",
        "llm_provider": "ollama",
    }
    return Settings(**{**base, **overrides})


def test_prod_accepts_strong_config() -> None:
    settings = _settings()
    assert settings.api_key == _STRONG_KEY


def test_prod_rejects_default_api_key() -> None:
    with pytest.raises(ValidationError, match="development default"):
        _settings(api_key=DEFAULT_DEV_API_KEY)


def test_prod_rejects_short_api_key() -> None:
    with pytest.raises(ValidationError, match="at least 24 characters"):
        _settings(api_key="short-key")


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
        api_key=DEFAULT_DEV_API_KEY,
        embedding_provider="ollama",
        llm_provider="ollama",
    )
    assert "localhost" in settings.cors_allow_origins[0]


def test_health_hides_topology_in_prod(monkeypatch) -> None:
    """未鉴权端点不得泄露模型 provider 与集合名。"""
    from app.config import get_settings

    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("API_KEY", _STRONG_KEY)
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


def test_readiness_still_requires_api_key(client: TestClient) -> None:
    assert client.get("/api/v1/readiness").status_code == 401
    assert client.get("/api/v1/readiness", headers=API_HEADERS).status_code == 200
