from typing import ClassVar, Optional

from pydantic import Field
from sqlalchemy import select

from app.agent.tools.base import NonEmptyToolString, Tool, ToolArgs, ToolContext, ToolResult
from app.errors import ErrorCode, ToolError
from app.schemas.base import to_utc_iso
from app.storage.models import Incident, MockDeployment, MockPod


class GetPodStatusArgs(ToolArgs):
    namespace: NonEmptyToolString = Field(description="Pod 所在命名空间，例如 ops-demo")
    name: str = Field(description="Pod 名称，例如 api-gateway-7f9c")


class GetPodStatusResult(ToolResult):
    answer_summary: str
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
            answer_summary=(
                f"{pod.namespace}/{pod.name} 当前状态为 {pod.phase}，"
                f"reason 为 {pod.reason or '未提供'}，已重启 {pod.restart_count} 次。"
            ),
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
    namespace: NonEmptyToolString = Field(description="命名空间，例如 ops-demo")
    name: Optional[str] = Field(
        default=None, description="可选：只查这一个 Deployment 名称"
    )


class DeploymentItem(ToolResult):
    namespace: str
    name: str
    image: str
    replicas: int
    available_replicas: int
    restart_allowed: bool
    restart_block_reason: Optional[str] = None
    updated_at: str


class ListDeploymentsResult(ToolResult):
    answer_summary: str
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
        if not rows:
            target = f" Deployment {args.name}" if args.name else " Deployment"
            answer_summary = f"{args.namespace} 下未找到{target}。"
        elif args.name and len(rows) == 1:
            row = rows[0]
            replica_state = (
                "副本数正常"
                if row.available_replicas >= row.replicas
                else "副本数不足"
            )
            answer_summary = (
                f"{row.namespace}/{row.name} 期望副本数为 {row.replicas}，"
                f"当前可用副本数为 {row.available_replicas}，{replica_state}。"
            )
            if row.available_replicas == 0:
                answer_summary += (
                    "可用副本数为 0，禁止重启；请先检查镜像、资源或配置问题。"
                )
        else:
            answer_summary = (
                f"{args.namespace} 下当前共有 {len(rows)} 个 Deployment。"
            )
        return ListDeploymentsResult(
            answer_summary=answer_summary,
            namespace=args.namespace,
            total=len(rows),
            deployments=[
                DeploymentItem(
                    namespace=r.namespace,
                    name=r.name,
                    image=r.image,
                    replicas=r.replicas,
                    available_replicas=r.available_replicas,
                    restart_allowed=r.available_replicas > 0,
                    restart_block_reason=(
                        None
                        if r.available_replicas > 0
                        else "可用副本数为 0，禁止重启；请先检查镜像、资源或配置问题。"
                    ),
                    updated_at=to_utc_iso(r.updated_at),
                )
                for r in rows
            ],
        )


class ListAlertsArgs(ToolArgs):
    namespace: NonEmptyToolString = Field(description="命名空间")
    status: Optional[str] = Field(default=None, description="可选状态过滤")


class AlertItem(ToolResult):
    incident_id: str
    title: str
    status: str
    priority: str
    created_at: str


class ListAlertsResult(ToolResult):
    answer_summary: str
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
        if rows:
            status_detail = f"（状态为 {args.status}）" if args.status else ""
            answer_summary = (
                f"{args.namespace} 下当前有 {len(rows)} 个已创建的告警工单{status_detail}。"
            )
        else:
            status_detail = f"状态为 {args.status} 的" if args.status else ""
            answer_summary = (
                f"{args.namespace} 下当前没有{status_detail}已创建的告警工单。"
            )
        return ListAlertsResult(
            answer_summary=answer_summary,
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
