"""JWT 鉴权依赖注入。"""

from fastapi import Depends, Header

from app.auth.jwt import AuthContext, verify_access_token
from app.errors import ForbiddenError, UnauthorizedError
from app.storage.models import UserRole

AUTH_HEADER = "Authorization"


async def require_jwt(
    authorization: str | None = Header(default=None, alias=AUTH_HEADER)
) -> AuthContext:
    """从 Authorization: Bearer <token> 解析 JWT 并返回用户上下文。

    token 缺失、格式错误、过期、签名非法时抛 UnauthorizedError。
    """
    if not authorization:
        raise UnauthorizedError(f"Missing {AUTH_HEADER} header")

    # Authorization: Bearer <token>
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise UnauthorizedError(f"Invalid {AUTH_HEADER} format, expected 'Bearer <token>'")

    token = parts[1]
    return verify_access_token(token)


async def require_admin(auth: AuthContext = Depends(require_jwt)) -> AuthContext:
    """要求 admin 角色。普通用户调用时返回 403。"""
    if auth.role != UserRole.ADMIN.value:
        raise ForbiddenError("Requires admin role")
    return auth
