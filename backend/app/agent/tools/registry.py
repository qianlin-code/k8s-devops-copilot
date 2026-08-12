from typing import Any

from app.agent.tools.base import Tool, assert_write_contract
from app.agent.tools.readonly_tools import (
    GetPodStatusTool,
    ListAlertsTool,
    ListDeploymentsTool,
)
from app.agent.tools.write_tools import (
    CreateAlertTool,
    RestartDeploymentTool,
    UpdateAlertStatusTool,
)
from app.errors import ErrorCode, ToolError


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            assert_write_contract(tool)
            if tool.name in self._tools:
                raise ToolError(
                    f"Duplicate tool name '{tool.name}'",
                    code=ErrorCode.BUSINESS_RULE_VIOLATION,
                )
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                f"Unknown tool '{name}'",
                code=ErrorCode.TOOL_NOT_FOUND,
                details={"available": sorted(self._tools)},
            )
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def catalog_for_prompt(self) -> str:
        lines = []
        for tool in self._tools.values():
            kind = "写操作(需确认)" if tool.is_write else "只读"
            fields = ", ".join(tool.args_schema.model_fields.keys())
            lines.append(f"- {tool.name} [{kind}]: {tool.description} 入参: {fields}")
        return "\n".join(lines)


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry(
            [
                GetPodStatusTool(),
                ListDeploymentsTool(),
                ListAlertsTool(),
                RestartDeploymentTool(),
                CreateAlertTool(),
                UpdateAlertStatusTool(),
            ]
        )
    return _registry


def reset_tool_registry() -> None:
    global _registry
    _registry = None
