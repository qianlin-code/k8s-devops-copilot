import secrets

from fastapi import Depends, Header
from typing import Optional

from app.config import Settings, get_settings
from app.errors import UnauthorizedError

API_KEY_HEADER = "X-API-Key"


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> str:
    if not x_api_key:
        raise UnauthorizedError(f"Missing {API_KEY_HEADER} header")
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise UnauthorizedError("Invalid API key")
    return x_api_key
