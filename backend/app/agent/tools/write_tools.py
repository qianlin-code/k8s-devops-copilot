import uuid
from typing import ClassVar, Optional

from pydantic import Field

from app.agent.tools.base import Tool, ToolContext, ToolResult, WriteToolArgs
from app.errors import ErrorCode, ToolError, ToolPermissionDeniedError
from app.schemas.base import to_utc_iso
from app.storage.models import Incident, IncidentStatus, MockDeployment, MockPod


class RestartDeploymentArgs(WriteToolArgs):
    namespace: str = Field(description="命名空间")
    name: str = Field(description="要重启的 Deployment 名称")
    reason: str = Field(description="执行原因，写入审计")
    # 操作身份只看目标（namespace, name），reason 是解释性自由文本，
    # 不该参与去重判断——否则同一目标换个措辞就被误判成"新操作"
    idempotency_fields: ClassVar[tuple[str, ...]] = ("namespace", "name")


class RestartDeploymentResult(ToolResult):
    namespace: str
    name: str
    previous_available_replicas: int
    message: str


class RestartDeploymentTool(Tool[RestartDeploymentArgs, RestartDeploymentResult]):
    name: ClassVar[str] = "restart_deployment"
    description: ClassVar[str] = (
        "触发 Deployment 滚动重启，让新配置或修复后的镜像生效。"
        "属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = RestartDeploymentArgs

    def run(
        self, args: RestartDeploymentArgs, ctx: ToolContext
    ) -> RestartDeploymentResult:
        deployment = ctx.session.get(MockDeployment, (args.namespace, args.name))
        if deployment is None:
            raise ToolError(
                f"Deployment '{args.namespace}/{args.name}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"namespace": args.namespace, "name": args.name},
            )
        # 副本全部不可用时先要排查根因（镜像/资源/配置），重启治不了这类问题，
        # 反而会掩盖根因、白白消耗一次滚动更新窗口
        if deployment.available_replicas == 0:
            raise ToolPermissionDeniedError(
                "Cannot restart a deployment with zero available replicas; "
                "diagnose the root cause (image/resources/config) first",
                details={
                    "namespace": args.namespace,
                    "name": args.name,
                    "available_replicas": 0,
                },
            )
        previous = deployment.available_replicas
        ctx.session.flush()
        return RestartDeploymentResult(
            namespace=deployment.namespace,
            name=deployment.name,
            previous_available_replicas=previous,
            message="已触发滚动重启，请稍后确认 Pod 是否恢复到 Running 状态。",
        )


class CreateAlertArgs(WriteToolArgs):
    namespace: str = Field(description="告警所属命名空间")
    title: str = Field(min_length=4, max_length=120, description="工单标题")
    description: str = Field(min_length=10, description="故障描述与已排查结论")
    priority: str = Field(default="medium", description="low/medium/high")
    # 创建工单没有"目标资源"概念，标题本身就是身份——同一标题视为同一次创建请求
    idempotency_fields: ClassVar[tuple[str, ...]] = ("namespace", "title")


class CreateAlertResult(ToolResult):
    incident_id: str
    status: str
    title: str
    priority: str
    created_at: str


class CreateAlertTool(Tool[CreateAlertArgs, CreateAlertResult]):
    name: ClassVar[str] = "create_incident"
    description: ClassVar[str] = (
        "创建告警事件工单，转交给后台工程师。"
        "当知识库无解、或需要集群管理员权限才能处理时使用。属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = CreateAlertArgs

    def run(self, args: CreateAlertArgs, ctx: ToolContext) -> CreateAlertResult:
        if args.priority not in ("low", "medium", "high"):
            raise ToolError(
                "priority must be one of low/medium/high",
                code=ErrorCode.TOOL_ARGS_INVALID,
                details={"priority": args.priority},
            )
        incident = Incident(
            id=f"INC-{uuid.uuid4().hex[:10].upper()}",
            namespace=args.namespace,
            title=args.title,
            description=args.description,
            status=IncidentStatus.OPEN.value,
            priority=args.priority,
            conversation_id=ctx.conversation_id,
        )
        ctx.session.add(incident)
        ctx.session.flush()
        return CreateAlertResult(
            incident_id=incident.id,
            status=incident.status,
            title=incident.title,
            priority=incident.priority,
            created_at=to_utc_iso(incident.created_at),
        )


class UpdateAlertStatusArgs(WriteToolArgs):
    incident_id: str = Field(description="告警工单 ID")
    status: str = Field(description="open/in_progress/resolved/closed")
    note: Optional[str] = Field(default=None, description="状态变更备注")
    # 目标是 (incident_id, status)：同一工单改成不同状态应视为不同操作
    idempotency_fields: ClassVar[tuple[str, ...]] = ("incident_id", "status")


class UpdateAlertStatusResult(ToolResult):
    incident_id: str
    previous_status: str
    new_status: str


class UpdateAlertStatusTool(Tool[UpdateAlertStatusArgs, UpdateAlertStatusResult]):
    name: ClassVar[str] = "update_incident_status"
    description: ClassVar[str] = (
        "更新已有告警工单的状态。属于写操作，执行前需要用户确认。"
    )
    is_write: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False
    args_schema: ClassVar[type] = UpdateAlertStatusArgs

    def run(
        self, args: UpdateAlertStatusArgs, ctx: ToolContext
    ) -> UpdateAlertStatusResult:
        valid = {s.value for s in IncidentStatus}
        if args.status not in valid:
            raise ToolError(
                f"status must be one of {sorted(valid)}",
                code=ErrorCode.TOOL_ARGS_INVALID,
                details={"status": args.status},
            )
        incident = ctx.session.get(Incident, args.incident_id)
        if incident is None:
            raise ToolError(
                f"Incident '{args.incident_id}' not found",
                code=ErrorCode.RESOURCE_NOT_FOUND,
                details={"incident_id": args.incident_id},
            )
        previous = incident.status
        incident.status = args.status
        ctx.session.flush()
        return UpdateAlertStatusResult(
            incident_id=incident.id, previous_status=previous, new_status=incident.status
        )
