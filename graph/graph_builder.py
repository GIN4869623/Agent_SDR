from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.state import SDRState


def _route_after_supervisor(state: SDRState) -> str:
    if state.get("error"):
        return "finalize"
    next_action = state.get("next_action", "respond")
    if next_action == "call_hardware":
        return "hardware_tools"
    elif next_action == "consult_knowledge":
        return "knowledge_agent"
    else:
        return "finalize"


def _route_after_tools(state: SDRState) -> str:
    if state.get("error"):
        return "finalize"
    next_action = state.get("next_action", "respond")
    if next_action == "respond":
        return "finalize"
    return "supervisor"


def build_sdr_graph(
    supervisor_node: Callable,
    hardware_node: Callable,
    knowledge_node: Callable,
    finalize_node: Callable,
) -> Any:
    builder = StateGraph(SDRState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("hardware_tools", hardware_node)
    builder.add_node("knowledge_agent", knowledge_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("finalize", END)

    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "hardware_tools": "hardware_tools",
            "knowledge_agent": "knowledge_agent",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "hardware_tools",
        _route_after_tools,
        {
            "supervisor": "supervisor",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "knowledge_agent",
        _route_after_tools,
        {
            "supervisor": "supervisor",
            "finalize": "finalize",
        },
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
