"""用户认证相关 schema。"""

from pydantic import Field

from app.schemas.base import StrictBaseModel


class LoginRequest(StrictBaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(StrictBaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str
    organization_id: str


class RegisterRequest(StrictBaseModel):
    username: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=8, max_length=128)
    organization_name: str = Field(min_length=1, max_length=255)


class RegisterResponse(StrictBaseModel):
    user_id: str
    username: str
    role: str
    organization_id: str
