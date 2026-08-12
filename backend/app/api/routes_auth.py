"""用户认证路由：登录、注册。"""

import uuid

import bcrypt
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.jwt import create_access_token
from app.errors import ErrorCode, NonRetryableError, UnauthorizedError
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
)
from app.schemas.common import ERROR_RESPONSES
from app.storage.db import get_db
from app.storage.models import Organization, User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"], responses=ERROR_RESPONSES)


@router.post("/login", response_model=LoginResponse, summary="用户登录")
def login(payload: LoginRequest, session: Session = Depends(get_db)) -> LoginResponse:
    """验证用户名密码，签发 JWT token。"""
    user = session.scalar(select(User).where(User.username == payload.username))
    if user is None:
        raise UnauthorizedError("Invalid username or password")

    # bcrypt.checkpw 要求 bytes
    if not bcrypt.checkpw(payload.password.encode(), user.password_hash.encode()):
        raise UnauthorizedError("Invalid username or password")

    if not user.is_active:
        raise UnauthorizedError("User account is deactivated")

    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
        organization_id=user.organization_id,
    )

    return LoginResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
        organization_id=user.organization_id,
    )


@router.post("/register", response_model=RegisterResponse, summary="用户注册")
def register(
    payload: RegisterRequest, session: Session = Depends(get_db)
) -> RegisterResponse:
    """创建新用户账号。

    本阶段暂时公开注册，后续可改为 admin 专属（需要现有 admin 创建新账号）。
    """
    existing = session.scalar(select(User).where(User.username == payload.username))
    if existing is not None:
        raise NonRetryableError(
            "Username already exists", code=ErrorCode.VALIDATION_FAILED
        )

    # 检查组织是否存在，不存在则创建
    org = session.scalar(
        select(Organization).where(Organization.name == payload.organization_name)
    )
    if org is None:
        org = Organization(id=str(uuid.uuid4()), name=payload.organization_name)
        session.add(org)
        session.flush()

    # bcrypt work factor 12（2^12 = 4096 iterations）
    password_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt(rounds=12))

    user = User(
        id=str(uuid.uuid4()),
        username=payload.username,
        password_hash=password_hash.decode(),  # 存 str 而非 bytes
        role=UserRole.USER.value,  # 新注册用户默认是普通用户
        organization_id=org.id,
    )
    session.add(user)
    session.commit()

    return RegisterResponse(
        user_id=user.id,
        username=user.username,
        role=user.role,
        organization_id=user.organization_id,
    )
