from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolSpec:
    name: str
    description: str
    schema_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]
    category: str = "general"

    def as_openai_tool(self) -> dict[str, Any]:
        schema = self.schema_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    def __init__(self, tool_specs: list[ToolSpec]) -> None:
        self._tool_specs = list(tool_specs)
        self._tool_map = {tool.name: tool for tool in tool_specs}

    def as_openai_tools(
        self,
        *,
        names: set[str] | None = None,
        categories: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in self._tool_specs:
            if names is not None and tool.name not in names:
                continue
            if categories is not None and tool.category not in categories:
                continue
            tools.append(tool.as_openai_tool())
        return tools

    def has_tool(self, name: str) -> bool:
        return name in self._tool_map

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._tool_map.get(name)

    def execute(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self._tool_map.get(name)
        if tool is None:
            return {"status": "error", "error": f"未知技能: {name}"}
        try:
            validated = tool.schema_model.model_validate(payload)
        except ValidationError as exc:
            return {
                "status": "error",
                "error": f"技能 {name} 参数校验失败。",
                "details": exc.errors(),
            }

        try:
            return tool.handler(validated)
        except Exception as exc:  # pragma: no cover - guardrail for tool handlers
            return {
                "status": "error",
                "error": f"技能 {name} 执行失败: {exc}",
            }
