"""JWT token 签发和验证的单元测试。"""

import time

import jwt
import pytest

from app.auth.jwt import AuthContext, create_access_token, verify_access_token
from app.config import get_settings
from app.errors import UnauthorizedError


def test_create_and_verify_token():
    token = create_access_token(
        user_id="user-123",
        username="test-user",
        role="user",
        organization_id="org-456",
    )
    assert isinstance(token, str)
    assert len(token) > 20

    context = verify_access_token(token)
    assert isinstance(context, AuthContext)
    assert context.user_id == "user-123"
    assert context.username == "test-user"
    assert context.role == "user"
    assert context.organization_id == "org-456"


def test_verify_expired_token():
    """过期 token 应该抛 UnauthorizedError。"""
    settings = get_settings()
    # 手动构造一个已过期的 token（exp 设为 1 秒前）
    payload = {
        "sub": "user-123",
        "username": "test-user",
        "role": "admin",
        "organization_id": "org-456",
        "iat": int(time.time()) - 10,
        "exp": int(time.time()) - 1,  # 已过期
    }
    expired_token = jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )

    with pytest.raises(UnauthorizedError, match="Token expired"):
        verify_access_token(expired_token)


def test_verify_invalid_signature():
    """签名非法的 token 应该抛 UnauthorizedError。"""
    token = create_access_token(
        user_id="user-123",
        username="test-user",
        role="user",
        organization_id="org-456",
    )
    # 篡改 token 最后几个字符，破坏签名
    tampered = token[:-10] + "XXXXXXXXXX"

    with pytest.raises(UnauthorizedError, match="Invalid token"):
        verify_access_token(tampered)


def test_verify_malformed_token():
    """格式错误的 token 应该抛 UnauthorizedError。"""
    with pytest.raises(UnauthorizedError, match="Invalid token"):
        verify_access_token("not-a-valid-jwt-token")


def test_token_contains_all_claims():
    """生成的 token 应该包含所有必需的声明。"""
    settings = get_settings()
    token = create_access_token(
        user_id="user-123",
        username="admin",
        role="admin",
        organization_id="org-456",
    )

    # 不验证签名，直接解码查看 payload
    payload = jwt.decode(
        token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
    )

    assert payload["sub"] == "user-123"
    assert payload["username"] == "admin"
    assert payload["role"] == "admin"
    assert payload["organization_id"] == "org-456"
    assert "iat" in payload
    assert "exp" in payload
    assert payload["exp"] > payload["iat"]
