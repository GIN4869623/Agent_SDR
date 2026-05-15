from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from graph.state import SDRState
from tools import ToolRegistry


CONSULT_KNOWLEDGE_NAME = "consult_knowledge_agent"
STOP_HARDWARE_NAME = "stop_hardware_task"


def create_hardware_node(tool_registry: ToolRegistry):
    def hardware_node(state: SDRState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"next_action": "respond"}

        last_message = messages[-1]
        raw_tool_calls = getattr(last_message, "tool_calls", None) or []
        if isinstance(last_message, dict):
            raw_tool_calls = last_message.get("tool_calls", [])
        if not raw_tool_calls:
            return {"next_action": "respond"}

        results: list[dict[str, Any]] = []
        logs: list[str] = list(state.get("tool_logs", []))
        tool_messages: list[ToolMessage] = []

        for tc in raw_tool_calls:
            # Normalise to {name, args, id} regardless of source format
            if isinstance(tc, dict):
                # LangGraph-converted format: {"name": "...", "args": {...}, "id": "..."}
                # or OpenAI format: {"function": {"name": "...", "arguments": "..."}}
                if "function" in tc:
                    fn = tc["function"]
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    tc_id = tc.get("id", "")
                else:
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tc_id = tc.get("id", "")
            else:
                # LangChain ToolCall object
                tool_name = getattr(tc, "name", "")
                raw_args = getattr(tc, "args", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                tc_id = getattr(tc, "id", "")

            if tool_name == CONSULT_KNOWLEDGE_NAME:
                continue

            result = tool_registry.execute(tool_name, args)

            if (
                result.get("status") == "error"
                and "硬件正忙" in str(result.get("error", ""))
                and tool_name != STOP_HARDWARE_NAME
                and tool_registry.has_tool(STOP_HARDWARE_NAME)
            ):
                logs.append("检测到 USRP 正忙，自动尝试停止后台任务后重试。")
                stop_result = tool_registry.execute(STOP_HARDWARE_NAME, {})
                logs.append(
                    f"自动调用 {STOP_HARDWARE_NAME}: "
                    f"{stop_result.get('message') or stop_result.get('error') or stop_result.get('status')}"
                )
                result = tool_registry.execute(tool_name, args)
                result["recovery"] = {
                    "auto_stop_attempted": True,
                    "stop_result": stop_result,
                }

            results.append({"tool": tool_name, "args": args, "result": result})
            logs.append(f"技能 {tool_name} 执行完成，结果: {result.get('status', 'unknown')}。")

            tool_messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tc_id,
                name=tool_name,
            ))

        return {
            "messages": tool_messages,
            "hardware_results": state.get("hardware_results", []) + results,
            "tool_logs": logs,
            "next_action": "respond",
        }

    return hardware_node
