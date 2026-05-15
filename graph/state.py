from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class SDRState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    mode: str
    active_model: str
    temperature: float
    initial_route: str
    next_action: str
    iteration_count: int
    hardware_results: list[dict[str, Any]]
    knowledge_context: str
    final_reply: str
    tool_logs: list[str]
    timing: dict[str, Any]
    error: str | None


class SDRGraphConfig(TypedDict, total=False):
    configurable: dict[str, str]
