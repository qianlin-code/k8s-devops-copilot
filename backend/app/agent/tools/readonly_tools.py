from typing import ClassVar, Optional

from pydantic import Field
from sqlalchemy import select

from app.agent.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from app.errors import ErrorCode, ToolError
from app.schemas.base import to_utc_iso
from app.storage.models import Incident, MockDeployment, MockPod


class GetPodStatusArgs(ToolArgs):
    namespace: str = Field(description="Pod 所在命名空间，例如 ops-demo")
    name: str = Field(description="Pod 名称，例如 api-gateway-7f9c")


class GetPodStatusResult(ToolResult):
    namespace: str
    name: str
    phase: str
    reason: Optional[str] = None
    node_name: Optional[str] = None
    restart_count: int
    last_transition_at: Optional[str] = None


class GetPodStatusTool(Tool[GetPodStatusArgs, GetPodStatusResult]):
    name: ClassVar[str] = "get_pod_status"
    description: ClassVar[str] = (
        "查询 Pod 的运行阶段、调度节点、重启次数与异常原因。"
        "诊断 Pending/CrashLoopBackOff/ImagePullBackOff 类问题时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = GetPodStatusArgs

    def run(self, args: GetPodStatusArgs, ctx: ToolContext) -> GetPodStatusResult:
        pod = ctx.session.get(MockPod, (args.namespace, args.name))
        if pod is None:
            raise ToolError(
                f"Pod '{args.namespace}/{args.name}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"namespace": args.namespace, "name": args.name},
            )
        return GetPodStatusResult(
            namespace=pod.namespace,
            name=pod.name,
            phase=pod.phase,
            reason=pod.reason,
            node_name=pod.node_name,
            restart_count=pod.restart_count,
            last_transition_at=to_utc_iso(pod.last_transition_at)
            if pod.last_transition_at
            else None,
        )


class ListDeploymentsArgs(ToolArgs):
    namespace: str = Field(description="命名空间，例如 ops-demo")
    name: Optional[str] = Field(
        default=None, description="可选：只查这一个 Deployment 名称"
    )


class DeploymentItem(ToolResult):
    namespace: str
    name: str
    image: str
    replicas: int
    available_replicas: int
    updated_at: str


class ListDeploymentsResult(ToolResult):
    namespace: str
    total: int
    deployments: list[DeploymentItem]


class ListDeploymentsTool(Tool[ListDeploymentsArgs, ListDeploymentsResult]):
    name: ClassVar[str] = "list_deployments"
    description: ClassVar[str] = (
        "查询命名空间下 Deployment 的期望副本数、可用副本数与镜像版本。"
        "处理扩缩容异常、副本不足、镜像版本核对类问题时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = ListDeploymentsArgs

    def run(
        self, args: ListDeploymentsArgs, ctx: ToolContext
    ) -> ListDeploymentsResult:
        stmt = select(MockDeployment).where(MockDeployment.namespace == args.namespace)
        if args.name:
            stmt = stmt.where(MockDeployment.name == args.name)
        rows = ctx.session.scalars(stmt.order_by(MockDeployment.name)).all()
        return ListDeploymentsResult(
            namespace=args.namespace,
            total=len(rows),
            deployments=[
                DeploymentItem(
                    namespace=r.namespace,
                    name=r.name,
                    image=r.image,
                    replicas=r.replicas,
                    available_replicas=r.available_replicas,
                    updated_at=to_utc_iso(r.updated_at),
                )
                for r in rows
            ],
        )


class ListAlertsArgs(ToolArgs):
    namespace: str = Field(description="命名空间")
    status: Optional[str] = Field(default=None, description="可选状态过滤")


class AlertItem(ToolResult):
    incident_id: str
    title: str
    status: str
    priority: str
    created_at: str


class ListAlertsResult(ToolResult):
    namespace: str
    total: int
    alerts: list[AlertItem]


class ListAlertsTool(Tool[ListAlertsArgs, ListAlertsResult]):
    name: ClassVar[str] = "list_alerts"
    description: ClassVar[str] = (
        "查询命名空间下已创建的告警事件工单。用户问处理进度、"
        "或判断是否已有重复告警时使用。"
    )
    is_write: ClassVar[bool] = False
    cacheable: ClassVar[bool] = True
    args_schema: ClassVar[type] = ListAlertsArgs

    def run(self, args: ListAlertsArgs, ctx: ToolContext) -> ListAlertsResult:
        stmt = select(Incident).where(Incident.namespace == args.namespace)
        if args.status:
            stmt = stmt.where(Incident.status == args.status)
        rows = ctx.session.scalars(stmt.order_by(Incident.created_at.desc())).all()
        return ListAlertsResult(
            namespace=args.namespace,
            total=len(rows),
            alerts=[
                AlertItem(
                    incident_id=r.id,
                    title=r.title,
                    status=r.status,
                    priority=r.priority,
                    created_at=to_utc_iso(r.created_at),
                )
                for r in rows
            ],
        )
