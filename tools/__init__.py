"""Canonical tool definitions and execution registry.

Domain tools live in this package.  Agent orchestration calls the registry
directly; MCP is only a transport adapter over the same registry.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


ToolRisk = Literal["read_only", "control", "transmit"]


class ToolInput(BaseModel):
    """Base request model shared by every tool.

    Unknown fields are rejected so a model cannot silently smuggle parameters
    past the declared tool contract.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolSpec:
    """Protocol-independent definition of one callable capability."""

    name: str
    description: str
    schema_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    category: str = "general"
    risk: ToolRisk = "read_only"
    retry_safe: bool = True

    @property
    def parameters(self) -> dict[str, Any]:
        schema = self.schema_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Validate and execute all local tools through one boundary."""

    def __init__(
        self,
        tool_specs: list[ToolSpec],
        *,
        hardware_busy: Callable[[], bool] | None = None,
    ) -> None:
        self._tool_specs = list(tool_specs)
        self._tool_map = {tool.name: tool for tool in self._tool_specs}
        self._hardware_busy = hardware_busy
        if len(self._tool_map) != len(self._tool_specs):
            raise ValueError("工具名称不能重复")

    def all(self) -> list[ToolSpec]:
        return list(self._tool_specs)

    # Compatibility helpers for current callers and evaluation scripts.
    def get_all(self) -> list[ToolSpec]:
        return self.all()

    def get_by_category(self, category: str) -> list[ToolSpec]:
        return [tool for tool in self._tool_specs if tool.category == category]

    def get(self, name: str) -> ToolSpec | None:
        return self._tool_map.get(name)

    def get_tool(self, name: str) -> ToolSpec | None:
        return self.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tool_map

    def is_hardware_busy(self) -> bool:
        if self._hardware_busy is None:
            return False
        try:
            return bool(self._hardware_busy())
        except Exception:
            return False

    def as_openai_tools(
        self,
        category: str | None = None,
        names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_names = set(names) if names is not None else None
        return [
            tool.as_openai_tool()
            for tool in self._tool_specs
            if (category is None or tool.category == category)
            and (allowed_names is None or tool.name in allowed_names)
        ]

    async def call(self, name: str, args: dict[str, Any]) -> str:
        """Compatibility API used by the current Agent Loop."""

        result = await self.execute(name, args)
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def execute(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        allowed_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        if allowed_tools is not None and name not in allowed_tools:
            return {
                "status": "error",
                "errorType": "TOOL_NOT_ALLOWED",
                "tool": name,
                "allowedTools": sorted(allowed_tools),
                "retryable": False,
                "message": "当前 Skill 不允许调用该工具。",
            }

        tool = self._tool_map.get(name)
        if tool is None:
            return {
                "status": "error",
                "errorType": "UNKNOWN_TOOL",
                "tool": name,
                "message": f"未知技能: {name}",
            }

        try:
            validated = tool.schema_model.model_validate(payload)
        except ValidationError as exc:
            return {
                "status": "error",
                "errorType": "INVALID_TOOL_ARGUMENTS",
                "tool": name,
                "message": f"技能 {name} 参数校验失败。",
                "details": [
                    {
                        "type": item.get("type"),
                        "loc": list(item.get("loc", ())),
                        "msg": item.get("msg"),
                    }
                    for item in exc.errors()
                ],
            }

        execution = asyncio.create_task(self._invoke(tool, validated))
        try:
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            # Once a blocking hardware handler starts, let it settle before the
            # caller cancellation escapes.  A later idempotency phase can build
            # on this single execution boundary.
            try:
                await execution
            except Exception:
                pass
            raise

    async def _invoke(self, tool: ToolSpec, validated: BaseModel) -> dict[str, Any]:
        try:
            # Preserve the current controller execution semantics in this
            # first refactor: synchronous handlers run in the Agent event-loop
            # thread, exactly as they did in the previous async MCP wrappers.
            # Moving blocking UHD work to a managed executor is a separate
            # lifecycle/concurrency change and should not be hidden here.
            raw = tool.handler(validated)
            if inspect.isawaitable(raw):
                raw = await raw
        except Exception as exc:
            return {
                "status": "error",
                "errorType": "TOOL_EXECUTION_FAILED",
                "tool": tool.name,
                "message": f"技能 {tool.name} 执行失败: {exc}",
            }

        if isinstance(raw, dict):
            return raw
        return {"status": "success", "result": raw}


__all__ = ["ToolInput", "ToolRegistry", "ToolRisk", "ToolSpec"]
