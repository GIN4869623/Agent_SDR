from __future__ import annotations

import json
import re
from typing import Any

from config.settings import Settings
from graph.state import SDRState
from prompts.system_prompts import build_supervisor_system_prompt
from tools import ToolRegistry


CONSULT_KNOWLEDGE_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "consult_knowledge_agent",
        "description": (
            "Call the cloud knowledge agent to retrieve SDR theory documentation "
            "and analyze hardware execution results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The knowledge question or analysis request for the knowledge agent.",
                },
                "analyze_results": {
                    "type": "boolean",
                    "description": "Set to true to have the knowledge agent analyze recent hardware results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


# === helpers (from old proven agent) ===

def _extract_user_instruction(messages: list[Any]) -> str:
    """Return the LAST user message (not first — MemorySaver accumulates history)."""
    for msg in reversed(messages):
        content = ""
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                content = msg.get("content", "")
        elif hasattr(msg, "content"):
            typ = getattr(msg, "type", "")
            if typ == "human":
                content = getattr(msg, "content", "")
        if content:
            return str(content)[:2000]
    return "SDR知识查询"


def _extract_message_text(instruction: str) -> str:
    """Extract the actual message body from a send command."""
    m = re.search(r'[""]([^""]+)[""]', instruction)
    if m:
        return m.group(1).strip()
    m = re.search(
        r'(?:发送文本|发送文字|发送消息|收发文本|收发文字|收发消息|'
        r'发送|传输|收发|发射|送|发)'
        r'\s*[:：]?\s*'
        r'(.+?)'
        r'(?:\s*(?:并|然后|再|且|以及)\s*(?:接收|解调|验证|显示|查看|可视化|框图)'
        r'|\s*(?:用|采用|使用|以)\s*(?:bpsk|qpsk|16\s*-?\s*qam|2\s*-?\s*fsk|fsk)'
        r'|\s*$)',
        instruction,
        flags=re.I,
    )
    if m:
        text = m.group(1).strip().rstrip('，。、.')
        text = re.sub(r'^\s*(?:bpsk|qpsk|16\s*-?\s*qam|2\s*-?\s*fsk|fsk)\s*', '', text, flags=re.I)
        text = re.sub(r'^\s*(?:文本|文字|消息)\s*', '', text)
        if text and len(text) < 200:
            return text
    return instruction


def _extract_freq(instruction: str, default: float = 2_400_000_000.0) -> float:
    """Extract frequency in Hz from instruction text."""
    m = re.search(r'(\d+\.?\d*)\s*([gGmMkK]?[hH][zZ])', instruction)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit.startswith("G"):
            return val * 1e9
        elif unit.startswith("M"):
            return val * 1e6
        elif unit.startswith("K"):
            return val * 1e3
    return default


def _regex_select_tool(instruction: str) -> tuple[str, dict[str, Any]]:
    """Regex-based tool selection and parameter extraction.
    Used as fallback when LLM tool calling fails.
    """
    text_lower = instruction.lower()

    if re.search(r'停止|关闭|停掉|停下', text_lower):
        return ("stop_hardware_task", {})

    if re.search(r'自适应|根据.{0,5}snr|根据.{0,5}信道质量|自动选择.*调制', text_lower, re.I):
        return ("adaptive_modulation_transmit", {"text": _extract_message_text(instruction)})

    if re.search(r'干净.{0,3}信道|最优.{0,3}频点|干净.{0,3}频点|找.{0,4}(?:干扰|噪声|底噪).{0,3}(?:小|低|少)', text_lower):
        msg = _extract_message_text(instruction)
        has_tone = bool(re.search(r'正弦|tone|单音|可视化|波形|频谱|框图', text_lower))
        return ("auto_optimal_transmit", {"text": msg, "enable_tone": has_tone or msg == instruction})

    if re.search(r'扫描|扫频|底噪|频谱.*扫描|扫.{0,2}频', text_lower):
        return ("perform_physical_scan", {"center_freq_hz": _extract_freq(instruction)})

    if re.search(r'正弦|tone|单音|可视化|波形.*频谱|频谱.*波形|框图', text_lower):
        freq = _extract_freq(instruction)
        tone_freq = _extract_freq(instruction, 100_000.0)
        m = re.search(r'(\d+\.?\d*)\s*([kK]?[hH][zZ])', instruction)
        if m:
            tone_freq = _extract_freq(instruction, 100_000.0)
        return ("tone_loopback_visualize", {"center_freq_hz": freq, "tone_freq_hz": tone_freq})

    if re.search(r'发送|收发|传输|发出去|发射|送', text_lower):
        msg = _extract_message_text(instruction)
        mod = "2-FSK"
        if re.search(r'16\s*[-]?\s*qam', text_lower):
            mod = "16-QAM"
        elif re.search(r'qpsk', text_lower):
            mod = "QPSK"
        elif re.search(r'(?:2\s*[-]?\s*)?fsk', text_lower):
            mod = "2-FSK"
        elif re.search(r'bpsk', text_lower):
            mod = "BPSK"
        return ("text_fsk_send_and_receive", {"text": msg, "modulation_scheme": mod})

    return ("auto_optimal_transmit", {"text": instruction})


# === supervisor node factory ===

def create_supervisor_node(
    ollama_client: Any,
    tool_registry: ToolRegistry,
    settings: Settings,
):
    hardware_tools = tool_registry.as_openai_tools(categories={"hardware"})
    all_tools = hardware_tools + [CONSULT_KNOWLEDGE_TOOL_DEF]

    def _build_tool_call_dict(name: str, args: dict, call_id: str = "") -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id or f"call_{name}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        }

    def _call_ollama_with_tools(
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        tool_choice: str = "auto",
    ) -> Any | None:
        """Call Ollama with tools. Returns response or None on failure."""
        try:
            return ollama_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=all_tools,
                tool_choice=tool_choice,
                stream=False,
            )
        except Exception:
            if tool_choice != "auto":
                # Retry with auto
                try:
                    return ollama_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        tools=all_tools,
                        tool_choice="auto",
                        stream=False,
                    )
                except Exception:
                    return None
            return None

    def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
        """Extract tool_calls from Ollama response as list of dicts."""
        msg = response.choices[0].message
        raw = getattr(msg, "tool_calls", None) or []
        result: list[dict[str, Any]] = []
        for tc in raw:
            tc_id = getattr(tc, "id", "")
            fn = getattr(tc, "function", None)
            if fn is not None:
                name = getattr(fn, "name", "")
                arguments = getattr(fn, "arguments", "{}")
            else:
                name = tc.get("name", "") if isinstance(tc, dict) else ""
                arguments = tc.get("arguments", "{}") if isinstance(tc, dict) else "{}"
            result.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        return result

    def supervisor_node(state: SDRState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0)
        initial_route = state.get("initial_route", "chat")
        active_model = state.get("active_model", settings.model_think)
        temperature = state.get("temperature", settings.temperature_think)
        messages = state.get("messages", [])
        hw_results = state.get("hardware_results", [])

        # Max turns guard
        if iteration >= settings.max_tool_turns:
            return {
                "messages": [{"role": "assistant", "content": ""}],
                "next_action": "respond",
                "iteration_count": iteration + 1,
            }

        # === Knowledge route (iteration 0): inject consult_knowledge_agent directly ===
        if initial_route == "knowledge" and iteration == 0:
            instruction = _extract_user_instruction(messages)
            return {
                "messages": [_build_tool_call_dict(
                    "consult_knowledge_agent",
                    {"query": instruction, "analyze_results": False},
                    "call_knowledge",
                )],
                "next_action": "consult_knowledge",
                "iteration_count": iteration + 1,
            }

        # === Hardware route (iteration 0): LLM tool calling ===
        if initial_route == "hardware" and iteration == 0:
            instruction = _extract_user_instruction(messages)

            # Build messages in OLD proven format (plain dicts, not LangChain objects).
            # Two system messages like old agent's _build_messages() + _route_note().
            api_messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_supervisor_system_prompt()},
                {"role": "system", "content": (
                    "当前请求已被系统判定为硬件执行动作。你必须调用一个硬件工具来实际执行物理层操作，"
                    "不要仅用文字描述。可用的硬件工具: perform_physical_scan, tone_loopback_visualize, "
                    "stop_hardware_task, text_fsk_send_and_receive, adaptive_modulation_transmit, "
                    "auto_optimal_transmit。如果需要分析执行结果或查询理论知识，调用 consult_knowledge_agent。"
                )},
                {"role": "user", "content": instruction},
            ]

            # Try LLM with tool_choice="required" (matching old agent's approach)
            response = _call_ollama_with_tools(api_messages, active_model, temperature, "required")
            tool_calls = _response_tool_calls(response) if response else []

            # Validate tool calls have non-empty names
            valid_tool_calls = [
                tc for tc in tool_calls
                if tc.get("function", {}).get("name", "") in {
                    "consult_knowledge_agent",
                    "perform_physical_scan",
                    "tone_loopback_visualize",
                    "stop_hardware_task",
                    "text_fsk_send_and_receive",
                    "adaptive_modulation_transmit",
                    "auto_optimal_transmit",
                }
            ]

            if not valid_tool_calls:
                # LLM failed — fall back to regex
                tool_name, args = _regex_select_tool(instruction)
                assistant_dict = _build_tool_call_dict(tool_name, args, f"call_{tool_name}")
            else:
                assistant_dict = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": valid_tool_calls,
                }

            tc_list = assistant_dict.get("tool_calls", [])
            tc_names = [tc.get("function", {}).get("name", "") for tc in tc_list]
            next_action = "consult_knowledge" if "consult_knowledge_agent" in tc_names else "call_hardware"

            return {
                "messages": [assistant_dict],
                "next_action": next_action,
                "iteration_count": iteration + 1,
            }

        # === Subsequent iterations: LLM decides next step ===
        # Convert state messages to plain dicts for Ollama
        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_supervisor_system_prompt()},
        ]
        for m in messages:
            if isinstance(m, dict):
                api_messages.append(m)
            elif hasattr(m, "type") and hasattr(m, "content"):
                role = "user" if getattr(m, "type", "") == "human" else "assistant"
                api_messages.append({"role": role, "content": getattr(m, "content", "") or ""})

        # Check if hardware results need analysis
        needs_analysis = False
        for r in hw_results:
            result = r.get("result", {})
            if result.get("status") == "error":
                needs_analysis = True
                break
            if result.get("estimated_snr_db") or result.get("probe_snr_db"):
                needs_analysis = True
                break

        if needs_analysis:
            instruction = _extract_user_instruction(messages)
            return {
                "messages": [_build_tool_call_dict(
                    "consult_knowledge_agent",
                    {"query": instruction, "analyze_results": True},
                    "call_analyze",
                )],
                "next_action": "consult_knowledge",
                "iteration_count": iteration + 1,
            }

        # No more tools needed — let finalize generate the response
        return {
            "messages": [{"role": "assistant", "content": ""}],
            "next_action": "respond",
            "iteration_count": iteration + 1,
        }

    return supervisor_node
