"""JWT token 签发和验证。

access token 有效期 8 小时（可配置），暂不实现 refresh token（二期考虑）。
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from app.config import get_settings
from app.errors import UnauthorizedError


@dataclass(slots=True, frozen=True)
class AuthContext:
    """从 JWT token 解析出的用户认证上下文。"""

    user_id: str
    username: str
    role: str
    organization_id: str


def create_access_token(
    user_id: str, username: str, role: str, organization_id: str
) -> str:
    """签发 JWT access token，有效期由配置项 JWT_ACCESS_TOKEN_EXPIRE_HOURS 决定。"""
    settings = get_settings()
    now = datetime.now(UTC)
    expire = now + timedelta(hours=settings.jwt_access_token_expire_hours)

    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "organization_id": organization_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_access_token(token: str) -> AuthContext:
    """验证 JWT token 并返回用户上下文。

    token 过期、签名非法、格式错误时抛 UnauthorizedError。
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return AuthContext(
            user_id=str(payload["sub"]),
            username=str(payload["username"]),
            role=str(payload["role"]),
            organization_id=str(payload["organization_id"]),
        )
    except jwt.ExpiredSignatureError:
        raise UnauthorizedError("Token expired") from None
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError(f"Invalid token: {exc}") from None
