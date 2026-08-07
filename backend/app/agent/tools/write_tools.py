import uuid
from typing import ClassVar, Optional

from pydantic import Field

from app.agent.tools.base import Tool, ToolContext, ToolResult, WriteToolArgs
from app.errors import ErrorCode, ToolError, ToolPermissionDeniedError
from app.schemas.base import to_utc_iso
from app.storage.models import MockAccount, Ticket, TicketStatus


class ResetPermissionCacheArgs(WriteToolArgs):
    user_id: str = Field(description="要刷新权限缓存的账号 ID")
    reason: str = Field(description="执行原因，写入审计")


class ResetPermissionCacheResult(ToolResult):
    user_id: str
    previous_cache_version: int
    new_cache_version: int
    message: str


class ResetPermissionCacheTool(Tool[ResetPermissionCacheArgs, ResetPermissionCacheResult]):
    name: ClassVar[str] = "reset_permission_cache"
    description: ClassVar[str] = (
        "刷新账号的权限缓存，使管理员刚做的提权立即生效。"
        "属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = ResetPermissionCacheArgs

    def run(
        self, args: ResetPermissionCacheArgs, ctx: ToolContext
    ) -> ResetPermissionCacheResult:
        account = ctx.session.get(MockAccount, args.user_id)
        if account is None:
            raise ToolError(
                f"Account '{args.user_id}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"user_id": args.user_id},
            )
        if account.status == "suspended":
            raise ToolPermissionDeniedError(
                "Cannot reset cache for a suspended account; resolve billing first",
                details={"user_id": args.user_id, "status": account.status},
            )
        previous = account.cache_version
        account.cache_version = previous + 1
        ctx.session.flush()
        return ResetPermissionCacheResult(
            user_id=account.user_id,
            previous_cache_version=previous,
            new_cache_version=account.cache_version,
            message="权限缓存已刷新，请让用户重新登录验证。",
        )


class CreateTicketArgs(WriteToolArgs):
    user_id: str = Field(description="报障账号 ID")
    title: str = Field(min_length=4, max_length=120, description="工单标题")
    description: str = Field(min_length=10, description="故障描述与已排查结论")
    priority: str = Field(default="medium", description="low/medium/high")


class CreateTicketResult(ToolResult):
    ticket_id: str
    status: str
    title: str
    priority: str
    created_at: str


class CreateTicketTool(Tool[CreateTicketArgs, CreateTicketResult]):
    name: ClassVar[str] = "create_ticket"
    description: ClassVar[str] = (
        "创建人工工单，转交给后台工程师。"
        "当知识库无解、或需要管理员权限才能处理时使用。属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = CreateTicketArgs

    def run(self, args: CreateTicketArgs, ctx: ToolContext) -> CreateTicketResult:
        if args.priority not in ("low", "medium", "high"):
            raise ToolError(
                "priority must be one of low/medium/high",
                code=ErrorCode.TOOL_ARGS_INVALID,
                details={"priority": args.priority},
            )
        ticket = Ticket(
            id=f"TK-{uuid.uuid4().hex[:10].upper()}",
            user_id=args.user_id,
            title=args.title,
            description=args.description,
            status=TicketStatus.OPEN.value,
            priority=args.priority,
            conversation_id=ctx.conversation_id,
        )
        ctx.session.add(ticket)
        ctx.session.flush()
        return CreateTicketResult(
            ticket_id=ticket.id,
            status=ticket.status,
            title=ticket.title,
            priority=ticket.priority,
            created_at=to_utc_iso(ticket.created_at),
        )


class UpdateTicketStatusArgs(WriteToolArgs):
    ticket_id: str = Field(description="工单 ID")
    status: str = Field(description="open/in_progress/resolved/closed")
    note: Optional[str] = Field(default=None, description="状态变更备注")


class UpdateTicketStatusResult(ToolResult):
    ticket_id: str
    previous_status: str
    new_status: str


class UpdateTicketStatusTool(Tool[UpdateTicketStatusArgs, UpdateTicketStatusResult]):
    name: ClassVar[str] = "update_ticket_status"
    description: ClassVar[str] = (
        "更新已有工单的状态。属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = UpdateTicketStatusArgs

    def run(
        self, args: UpdateTicketStatusArgs, ctx: ToolContext
    ) -> UpdateTicketStatusResult:
        valid = {s.value for s in TicketStatus}
        if args.status not in valid:
            raise ToolError(
                f"status must be one of {sorted(valid)}",
                code=ErrorCode.TOOL_ARGS_INVALID,
                details={"status": args.status},
            )
        ticket = ctx.session.get(Ticket, args.ticket_id)
        if ticket is None:
            raise ToolError(
                f"Ticket '{args.ticket_id}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"ticket_id": args.ticket_id},
            )
        previous = ticket.status
        ticket.status = args.status
        ctx.session.flush()
        return UpdateTicketStatusResult(
            ticket_id=ticket.id, previous_status=previous, new_status=ticket.status
        )
