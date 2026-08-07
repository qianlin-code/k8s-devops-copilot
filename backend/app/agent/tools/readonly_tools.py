from typing import ClassVar, Optional

from pydantic import Field
from sqlalchemy import select

from app.agent.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from app.errors import ErrorCode, ToolError
from app.schemas.base import to_utc_iso
from app.storage.models import MockAccount, MockOrder, Ticket


class GetAccountStatusArgs(ToolArgs):
    user_id: str = Field(description="要查询的账号 ID，例如 u-1001")


class GetAccountStatusResult(ToolResult):
    user_id: str
    email: str
    status: str
    permission_level: str
    locked_reason: Optional[str] = None
    last_login_at: Optional[str] = None
    cache_version: int


class GetAccountStatusTool(Tool[GetAccountStatusArgs, GetAccountStatusResult]):
    name: ClassVar[str] = "get_account_status"
    description: ClassVar[str] = (
        "查询账号的启用状态、权限等级、锁定原因。诊断登录失败(403/401)、权限不足类问题时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = GetAccountStatusArgs

    def run(
        self, args: GetAccountStatusArgs, ctx: ToolContext
    ) -> GetAccountStatusResult:
        account = ctx.session.get(MockAccount, args.user_id)
        if account is None:
            raise ToolError(
                f"Account '{args.user_id}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"user_id": args.user_id},
            )
        return GetAccountStatusResult(
            user_id=account.user_id,
            email=account.email,
            status=account.status,
            permission_level=account.permission_level,
            locked_reason=account.locked_reason,
            last_login_at=to_utc_iso(account.last_login_at)
            if account.last_login_at
            else None,
            cache_version=account.cache_version,
        )


class ListOrdersArgs(ToolArgs):
    user_id: str = Field(description="账号 ID")
    status: Optional[str] = Field(
        default=None, description="可选状态过滤：paid/pending_payment/overdue"
    )


class OrderItem(ToolResult):
    order_id: str
    product: str
    status: str
    amount: float
    created_at: str


class ListOrdersResult(ToolResult):
    user_id: str
    total: int
    orders: list[OrderItem]


class ListOrdersTool(Tool[ListOrdersArgs, ListOrdersResult]):
    name: ClassVar[str] = "list_orders"
    description: ClassVar[str] = (
        "查询账号下的订单及付款状态。处理欠费停服、订阅到期、扩容账单类问题时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = ListOrdersArgs

    def run(self, args: ListOrdersArgs, ctx: ToolContext) -> ListOrdersResult:
        stmt = select(MockOrder).where(MockOrder.user_id == args.user_id)
        if args.status:
            stmt = stmt.where(MockOrder.status == args.status)
        rows = ctx.session.scalars(stmt.order_by(MockOrder.created_at.desc())).all()
        return ListOrdersResult(
            user_id=args.user_id,
            total=len(rows),
            orders=[
                OrderItem(
                    order_id=r.id,
                    product=r.product,
                    status=r.status,
                    amount=r.amount,
                    created_at=to_utc_iso(r.created_at),
                )
                for r in rows
            ],
        )


class ListTicketsArgs(ToolArgs):
    user_id: str = Field(description="账号 ID")
    status: Optional[str] = Field(default=None, description="可选状态过滤")


class TicketItem(ToolResult):
    ticket_id: str
    title: str
    status: str
    priority: str
    created_at: str


class ListTicketsResult(ToolResult):
    user_id: str
    total: int
    tickets: list[TicketItem]


class ListTicketsTool(Tool[ListTicketsArgs, ListTicketsResult]):
    name: ClassVar[str] = "list_tickets"
    description: ClassVar[str] = (
        "查询账号已提交的工单。用户问进度、或判断是否已有重复工单时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = ListTicketsArgs

    def run(self, args: ListTicketsArgs, ctx: ToolContext) -> ListTicketsResult:
        stmt = select(Ticket).where(Ticket.user_id == args.user_id)
        if args.status:
            stmt = stmt.where(Ticket.status == args.status)
        rows = ctx.session.scalars(stmt.order_by(Ticket.created_at.desc())).all()
        return ListTicketsResult(
            user_id=args.user_id,
            total=len(rows),
            tickets=[
                TicketItem(
                    ticket_id=r.id,
                    title=r.title,
                    status=r.status,
                    priority=r.priority,
                    created_at=to_utc_iso(r.created_at),
                )
                for r in rows
            ],
        )
