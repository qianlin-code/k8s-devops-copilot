import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.agent.tools.base import Tool, ToolContext
from app.agent.tools.cache import ToolResultCache
from app.agent.tools.registry import ToolRegistry
from app.errors import AppError, ErrorCode, ToolError
from app.storage.models import ToolCallAudit


@dataclass(slots=True)
class ToolInvocation:
    tool_name: str
    is_write: bool
    arguments: dict[str, Any]
    success: bool
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    cache_hit: bool
    idempotent_replay: bool
    elapsed_ms: int


class ToolExecutor:
    """统一收口：只读走缓存，写操作靠 request_id 幂等，全部落审计。"""

    def __init__(self, registry: ToolRegistry, cache: ToolResultCache) -> None:
        self._registry = registry
        self._cache = cache

    def execute(
        self, tool_name: str, raw_args: dict[str, Any], ctx: ToolContext
    ) -> ToolInvocation:
        started = time.perf_counter()
        try:
            tool = self._registry.get(tool_name)
        except ToolError as exc:
            inv = _failed(tool_name, False, raw_args, exc, _ms(started))
            self._audit(inv, ctx, request_id=None)
            return inv

        try:
            args = tool.parse_args(raw_args)
        except ToolError as exc:
            inv = _failed(tool_name, tool.is_write, raw_args, exc, _ms(started))
            self._audit(inv, ctx, request_id=raw_args.get("request_id"))
            return inv

        normalized = args.model_dump(mode="json")
        request_id = normalized.get("request_id") if tool.is_write else None

        if tool.is_write:
            replay = self._find_replay(ctx, tool_name, request_id)
            if replay is not None:
                return ToolInvocation(
                    tool_name=tool_name,
                    is_write=True,
                    arguments=normalized,
                    success=replay.success,
                    result=replay.result,
                    error_code=replay.error_code,
                    error_message=replay.error_message,
                    cache_hit=False,
                    idempotent_replay=True,
                    elapsed_ms=_ms(started),
                )

        cache_key = None
        if tool.cacheable and not tool.is_write:
            cache_key = ToolResultCache.build_key(tool_name, normalized)
            cached = self._cache.get(cache_key)
            if cached is not None:
                inv = ToolInvocation(
                    tool_name=tool_name,
                    is_write=False,
                    arguments=normalized,
                    success=True,
                    result=cached,
                    error_code=None,
                    error_message=None,
                    cache_hit=True,
                    idempotent_replay=False,
                    elapsed_ms=_ms(started),
                )
                self._audit(inv, ctx, request_id=None)
                return inv

        try:
            result = tool.run(args, ctx)
            payload = result.model_dump(mode="json")
        except AppError as exc:
            inv = _failed(tool_name, tool.is_write, normalized, exc, _ms(started))
            self._audit(inv, ctx, request_id=request_id)
            return inv
        except Exception as exc:
            wrapped = ToolError(
                f"Tool '{tool_name}' raised an unexpected error",
                code=ErrorCode.INTERNAL_ERROR,
                details={"exception_type": type(exc).__name__},
            )
            inv = _failed(tool_name, tool.is_write, normalized, wrapped, _ms(started))
            self._audit(inv, ctx, request_id=request_id)
            return inv

        if cache_key is not None:
            self._cache.set(cache_key, payload)

        inv = ToolInvocation(
            tool_name=tool_name,
            is_write=tool.is_write,
            arguments=normalized,
            success=True,
            result=payload,
            error_code=None,
            error_message=None,
            cache_hit=False,
            idempotent_replay=False,
            elapsed_ms=_ms(started),
        )
        self._audit(inv, ctx, request_id=request_id)
        return inv

    def _find_replay(
        self, ctx: ToolContext, tool_name: str, request_id: str | None
    ) -> ToolCallAudit | None:
        """查找可重放的写操作审计行。

        必须按 conversation_id 限定范围：`request_id` 由 LLM 生成，实测会出现
        "123456" 这类极易碰撞的值。不限定范围时，B 会话用了 A 会话已用过的
        request_id 会直接重放 A 的结果，把别人的写操作结果当成自己的返回。
        表上的唯一约束同样是 (conversation_id, request_id) 复合键。
        """
        if not request_id:
            return None
        row = ctx.session.scalar(
            select(ToolCallAudit).where(
                ToolCallAudit.request_id == request_id,
                ToolCallAudit.conversation_id == ctx.conversation_id,
            )
        )
        if row is None:
            return None
        if row.tool_name != tool_name:
            raise ToolError(
                "request_id was already used by a different tool",
                code=ErrorCode.BUSINESS_RULE_VIOLATION,
                details={"request_id": request_id, "original_tool": row.tool_name},
            )
        return row

    def _audit(
        self, inv: ToolInvocation, ctx: ToolContext, *, request_id: str | None
    ) -> None:
        """写审计记录并立即提交。

        必须 commit 而不是 flush：审计发生在 Agent 循环中段，后面还有多次
        LLM 调用。flush 会开启写事务却不释放锁，等于又把写锁攥到整轮结束 ——
        并发对话下第二个请求的审计 INSERT 就会撞锁。

        审计记录语义上独立（它记的是"已经发生的事"，不该因后续步骤失败而回滚），
        所以提前提交没有一致性问题，反而更符合审计日志的语义。
        写工具的幂等也依赖它立即可见。
        """
        ctx.session.add(
            ToolCallAudit(
                request_id=request_id,
                trace_id=ctx.trace_id,
                conversation_id=ctx.conversation_id,
                tool_name=inv.tool_name,
                is_write=inv.is_write,
                arguments=inv.arguments,
                result=inv.result,
                success=inv.success,
                error_code=inv.error_code,
                error_message=inv.error_message,
                cache_hit=inv.cache_hit,
                idempotent_replay=inv.idempotent_replay,
                elapsed_ms=inv.elapsed_ms,
            )
        )
        ctx.session.commit()


def _failed(
    tool_name: str,
    is_write: bool,
    args: dict[str, Any],
    exc: AppError,
    elapsed_ms: int,
) -> ToolInvocation:
    return ToolInvocation(
        tool_name=tool_name,
        is_write=is_write,
        arguments=args,
        success=False,
        result=None,
        error_code=exc.code.value,
        error_message=exc.message,
        cache_hit=False,
        idempotent_replay=False,
        elapsed_ms=elapsed_ms,
    )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
