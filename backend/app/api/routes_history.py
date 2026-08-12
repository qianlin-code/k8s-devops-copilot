from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth.dependencies import require_jwt
from app.auth.jwt import AuthContext
from app.schemas.common import ERROR_RESPONSES
from app.schemas.history import (
    ConversationDetailResponse,
    ConversationListResponse,
    ToolAuditListResponse,
)
from app.services.history_service import HistoryService
from app.storage.db import get_db

router = APIRouter(
    tags=["history"],
    dependencies=[Depends(require_jwt)],
    responses=ERROR_RESPONSES,
)
_service = HistoryService()


@router.get(
    "/conversations", response_model=ConversationListResponse, summary="会话列表"
)
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    auth: AuthContext = Depends(require_jwt),
) -> ConversationListResponse:
    return _service.list_conversations(
        session, user_id=auth.user_id, limit=limit, offset=offset
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="会话详情，含每轮执行链路",
)
async def get_conversation(
    conversation_id: str,
    include_trace: bool = Query(default=True),
    session: Session = Depends(get_db),
    auth: AuthContext = Depends(require_jwt),
) -> ConversationDetailResponse:
    """user_id 从 JWT token 获取，校验会话归属。不校验的话遍历
    conversation_id 就能读到任意用户的对话与执行链路。
    """
    return _service.get_conversation(
        session, conversation_id, include_trace=include_trace, user_id=auth.user_id
    )


@router.get(
    "/tool-audits",
    response_model=ToolAuditListResponse,
    summary="工具调用审计日志",
)
async def list_tool_audits(
    conversation_id: str | None = Query(default=None),
    tool_name: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    auth: AuthContext = Depends(require_jwt),
) -> ToolAuditListResponse:
    return _service.list_tool_audits(
        session,
        conversation_id=conversation_id,
        tool_name=tool_name,
        limit=limit,
        offset=offset,
        user_id=auth.user_id,
    )
