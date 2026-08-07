from typing import Literal

from pydantic import Field

from app.schemas.base import StrictBaseModel
from app.schemas.chat import ChatResponse
from app.schemas.common import ErrorResponse

# SSE 事件类型。前端按 phase 渲染进度，按 event 决定如何处理数据。
ProgressPhase = Literal[
    "accepted",  # 请求已接受，会话已建立
    "guarded",  # 输入安全检查通过
    "context_built",  # 多轮上下文组装完成
    "retrieving",  # 检索开始
    "retrieved",  # 检索完成（含各阶段耗时）
    "agent_step",  # Agent 状态机走过一个节点
    "generating",  # 开始生成最终回答
]


class ProgressEvent(StrictBaseModel):
    """阶段进展事件。

    只承载「进行到哪一步」，不重复传输 trace 细节 ——
    完整 trace 在终态的 ChatResponse 里一次性给出。
    """

    phase: ProgressPhase
    label: str = Field(description="面向用户的中文阶段描述")
    elapsed_ms: int = Field(ge=0, description="自请求开始的累计耗时")
    detail: dict[str, object] = Field(
        default_factory=dict, description="该阶段的补充信息，键随 phase 变化"
    )


class StreamErrorEvent(ErrorResponse):
    """SSE 的 error 事件载荷。

    比普通 ErrorResponse 多一个 http_status：SSE 帧发出时 HTTP 状态码
    已经是 200 了，客户端无从得知这个错误"本该"是几百。不带的话前端
    只能靠 code 猜，或者一律当 500 —— 注入拦截(422)会被误报成服务器故障。
    """

    http_status: int = Field(description="非流式接口下该错误对应的 HTTP 状态码")


class ChatStreamEnvelope(StrictBaseModel):
    """SSE 事件的文档化载荷。

    实际传输是 `event: <name>` + `data: <json>` 的 SSE 帧，不是这个对象本身。
    定义它只为让三类载荷进入 OpenAPI，从而被前端类型生成脚本收录 ——
    否则前端只能手写 SSE 类型，违背「schema 是唯一事实来源」。

    事件与载荷的对应：
      - `event: progress` -> ProgressEvent
      - `event: done`     -> ChatResponse（与非流式 /chat 完全一致）
      - `event: error`    -> ErrorResponse（与其他接口同一错误格式）
    """

    progress: ProgressEvent | None = Field(
        default=None, description="event: progress 的载荷"
    )
    done: ChatResponse | None = Field(default=None, description="event: done 的载荷")
    error: StreamErrorEvent | None = Field(
        default=None, description="event: error 的载荷"
    )
