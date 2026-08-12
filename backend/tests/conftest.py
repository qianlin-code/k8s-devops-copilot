import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# 环境必须在导入 app.config 之前设置好
_TMP = Path(tempfile.mkdtemp(prefix="copilot-tests-"))
os.environ.update(
    {
        "ENVIRONMENT": "dev",
        "STARTUP_PROBE_EXTERNAL": "false",
        # 测试用替身 reranker，不需要也不应该加载真实模型
        "WARMUP_RERANKER": "false",
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "QDRANT_PATH": str(_TMP / "qdrant"),
        "EMBEDDING_PROVIDER": "ollama",
        "LLM_PROVIDER": "ollama",
        "OLLAMA_EMBEDDING_MODEL": "fake-embedding",
        "OLLAMA_EMBEDDING_DIM": "64",
        "ENABLE_QUERY_REWRITE": "false",
        "AGENT_MAX_STEPS": "4",
        "TOOL_CACHE_TTL_SECONDS": "0",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

import app.dependencies as deps  # noqa: E402
from app.auth.jwt import create_access_token  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import factory  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag import bm25_index, reranker, vector_store  # noqa: E402
from app.storage.db import init_db, reset_engine, session_scope  # noqa: E402
from app.storage.seed import seed_mock_data, seed_test_users  # noqa: E402
from tests.fakes import FakeEmbeddingClient, KeywordReranker, ScriptedLLMClient  # noqa: E402

# 历史类接口要求 user_id（按用户隔离，这里指发起请求的运维人员身份），测试统一用这个身份
TEST_USER_ID = "ops-1"
TEST_SEED_PASSWORD = "test-only-password-not-a-demo-credential"


def auth_headers(user_id: str = TEST_USER_ID, *, role: str = "user") -> dict[str, str]:
    """生成测试用 JWT 鉴权头。

    JWT 是自包含的，验证只查签名和过期时间，不查库——测试不需要真的注册
    User 记录，可以直接用任意 user_id 签发 token（保持与迁移前测试用的
    "ops-1"/"ops-2" 等字符串身份兼容）。
    """
    token = create_access_token(
        user_id=user_id,
        username=user_id,
        role=role,
        organization_id="test-org",
    )
    return {"Authorization": f"Bearer {token}"}


API_HEADERS = auth_headers(TEST_USER_ID)
ADMIN_HEADERS = auth_headers("admin-1", role="admin")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """每个测试用独立的库与向量集合，并把外部依赖换成替身。"""
    workdir = Path(tempfile.mkdtemp(prefix="case-"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(workdir / 'case.db').as_posix()}")
    monkeypatch.setenv("QDRANT_PATH", str(workdir / "qdrant"))
    get_settings.cache_clear()
    reset_engine()
    vector_store.reset_vector_store()
    bm25_index.reset_bm25_index()
    factory.reset_clients()
    deps.reset_dependencies()

    llm = ScriptedLLMClient()
    monkeypatch.setattr(factory, "get_llm_client", lambda: llm)
    monkeypatch.setattr(factory, "get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr(deps, "get_llm_client", lambda: llm)
    monkeypatch.setattr(deps, "get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr(deps, "get_reranker", lambda: KeywordReranker())
    reranker.set_reranker(KeywordReranker())

    init_db()
    with session_scope() as session:
        seed_mock_data(session)
        seed_test_users(session, password=TEST_SEED_PASSWORD)  # JWT 切换后需要真实用户

    yield

    get_settings.cache_clear()
    reset_engine()
    vector_store.reset_vector_store()
    bm25_index.reset_bm25_index()
    factory.reset_clients()
    deps.reset_dependencies()
    reranker.reset_reranker()
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> ScriptedLLMClient:
    return deps.get_llm_client()  # type: ignore[return-value]


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def seeded_kb(client: TestClient) -> str:
    """灌一篇知识库文档，返回 document_id。"""
    resp = client.post(
        "/api/v1/knowledge/documents",
        headers=ADMIN_HEADERS,  # 知识库写操作需要 admin 权限
        json={
            "title": "登录故障排查手册",
            "content": (
                "# 登录故障排查手册\n\n"
                "## 403 Forbidden 权限不足\n"
                "账号 permission_level 为 restricted 时登录会返回 403，"
                "需要管理员将权限提升到 standard，随后刷新权限缓存。\n\n"
                "## 401 凭证过期\n"
                "Token 过期需要重新登录获取新凭证。\n"
            ),
            "chunk_strategy": "markdown",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["document"]["document_id"]


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    shutil.rmtree(_TMP, ignore_errors=True)
