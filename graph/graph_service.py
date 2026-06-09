from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from core.memory import ConversationMemory
from core.rag import LocalKnowledgeBase
from graph.graph_builder import build_sdr_graph
from graph.hardware_node import create_hardware_node
from graph.knowledge_node import create_knowledge_node
from graph.state import SDRState, SDRGraphConfig
from graph.supervisor_node import create_supervisor_node
from prompts.system_prompts import build_system_prompt
from skill import SkillRegistry
from tools import ToolRegistry


_HARDWARE_OBJECT_HINT = re.compile(
    r"(usrp|硬件|物理层|tone|回环|链路|波形|频谱|底噪|噪声|文本收发|文本发送|"
    r"正弦|正弦波|单音|连续波|信号|sine|sinusoid|fsk|bpsk|qpsk|16-qam|"
    r"频点|频率|频段|信道|干净信道|最优信道|干净频点|最优频点|增益|采样率|"
    r"可视化|框图|无线|发出去|视频流|video.stream|video_stream|gnu.radio|vlc)",
    re.I,
)
_HARDWARE_EXECUTION_HINT = re.compile(
    r"(请|帮我|给我|麻烦|直接|立即|现在)?[\s:：]*(启动|停止|执行|发送|发射|传输|"
    r"发\s*(?=正弦|单音|tone|sine|信号|波|[A-Za-z0-9\"'\"“‘])|发出去|输出|收发|扫描|扫频|扫噪|开始|"
    r"扫.{0,4}频|找.{0,8}(干净|最优|最佳)|关闭|打开|测量|测试|显示|查看|运行)",
    re.I,
)
_EXPLANATION_HINT = re.compile(
    r"(什么是|是什么|为什么|为何|请解释|解释一下|说明一下|介绍一下|介绍|"
    r"原理|流程|区别|比较|文档|手册|资料|知识库|知识|参数含义|怎么理解|如何理解|"
    r"怎么用|如何使用|讲讲|说说|告诉我|跟我讲|跟我聊聊|关于.{0,6}(知识|原理|概念|理论)|"
    r"想了解|想.{0,4}(了解|知道|学习|认识)|有没有.{0,6}(文档|资料|教程|手册)|"
    r"讲一下|说一下|聊一下|问一下|你知道|你知道什么是|知道.{0,6}吗|提供.{0,6}知识)",
    re.I,
)
_TOOL_NAME_HINT = re.compile(
    r"(search_sdr_knowledge|tone_loopback|perform_physical|text_fsk|adaptive_modulation|"
    r"auto_optimal|stop_hardware|start_video_stream|stop_video_stream|video_stream)",
    re.I,
)

_MARKDOWN_CODE_BLOCK = re.compile(r"```[^\n]*\n([\s\S]*?)```")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MARKDOWN_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_MARKDOWN_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_MARKDOWN_ORDERED_LIST = re.compile(r"^\s*\d+[.)]\s+", re.M)
_MARKDOWN_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.M)
_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|~~)(.*?)\1")
_MARKDOWN_SINGLE_STAR = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MARKDOWN_INLINE_CODE = re.compile(r"`([^`]+)`")


def _plain_text_reply(text: str) -> str:
    plain = str(text or "")
    if not plain:
        return ""

    plain = _MARKDOWN_CODE_BLOCK.sub(lambda match: match.group(1).strip(), plain)
    plain = _MARKDOWN_IMAGE.sub(lambda match: match.group(1).strip(), plain)
    plain = _MARKDOWN_LINK.sub(lambda match: f"{match.group(1)}（{match.group(2)}）", plain)
    plain = _MARKDOWN_RULE.sub("", plain)
    plain = _MARKDOWN_TABLE_SEPARATOR.sub("", plain)
    plain = _MARKDOWN_HEADING.sub("", plain)
    plain = _MARKDOWN_BLOCKQUOTE.sub("", plain)
    plain = _MARKDOWN_BULLET.sub("", plain)
    plain = _MARKDOWN_ORDERED_LIST.sub("", plain)
    plain = _MARKDOWN_INLINE_CODE.sub(lambda match: match.group(1), plain)

    previous = None
    while previous != plain:
        previous = plain
        plain = _MARKDOWN_EMPHASIS.sub(lambda match: match.group(2), plain)
        plain = _MARKDOWN_SINGLE_STAR.sub(lambda match: match.group(1), plain)

    cleaned_lines: list[str] = []
    for line in plain.splitlines():
        normalized = line.strip()
        if not normalized:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if "|" in normalized:
            cells = [cell.strip() for cell in normalized.strip("|").split("|") if cell.strip()]
            if len(cells) > 1:
                normalized = "，".join(cells)
        cleaned_lines.append(normalized)

    plain = "\n".join(cleaned_lines).strip()
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain


class GraphService:
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        knowledge_base: LocalKnowledgeBase,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.tool_registry = tool_registry
        self.memory = memory
        self.knowledge_base = knowledge_base

        try:
            from openai import OpenAI
            self.ollama_client = OpenAI(
                api_key=settings.api_key,
                base_url=settings.ollama_base_url,
            )
        except Exception:
            self.ollama_client = None

        supervisor_node = create_supervisor_node(
            self.ollama_client, tool_registry, settings, skill_registry,
        )
        hardware_node = create_hardware_node(tool_registry)
        knowledge_node = create_knowledge_node(
            knowledge_base, settings.bailian_api_key, settings,
        )

        self.graph = build_sdr_graph(
            supervisor_node, hardware_node, knowledge_node, self._finalize_node,
        )

    def normalize_mode(self, mode: str) -> str:
        mode_key = (mode or "think").strip().lower()
        return mode_key if mode_key in self.settings.model_map else "think"

    def resolve_model_name(self, mode: str) -> str:
        return self.settings.model_map[self.normalize_mode(mode)]

    def resolve_temperature(self, mode: str, override: float | None) -> float:
        if override is not None:
            return max(0.0, min(2.0, float(override)))
        return float(self.settings.temperature_map[self.normalize_mode(mode)])

    @staticmethod
    def _classify_route(instruction: str) -> str:
        text = instruction.strip()
        if not text:
            return "chat"
        lowered = text.lower()
        mentions_tool = bool(_TOOL_NAME_HINT.search(lowered))
        mentions_hardware = bool(
            mentions_tool
            or _HARDWARE_OBJECT_HINT.search(lowered)
            or _HARDWARE_EXECUTION_HINT.search(text)
        )
        asks_explanation = bool(_EXPLANATION_HINT.search(text))
        explicit_execution = bool(_HARDWARE_EXECUTION_HINT.search(text)) or (
            mentions_tool and not asks_explanation
        )
        if asks_explanation:
            # Theory/knowledge questions go to knowledge agent
            return "knowledge"
        if explicit_execution and mentions_hardware:
            return "hardware"
        if mentions_hardware and not asks_explanation:
            return "hardware"
        return "chat"

    def _build_initial_state(
        self,
        session_id: str,
        instruction: str,
        mode: str,
        temperature: float,
    ) -> SDRState:
        selected_mode = self.normalize_mode(mode)
        route = self._classify_route(instruction)
        return {
            "messages": [HumanMessage(content=instruction)],
            "session_id": session_id,
            "mode": selected_mode,
            "active_model": self.resolve_model_name(selected_mode),
            "temperature": temperature,
            "initial_route": route,
            "next_action": "",
            "iteration_count": 0,
            "hardware_results": [],
            "knowledge_context": "",
            "final_reply": "",
            "tool_logs": [],
            "timing": {},
            "error": None,
        }

    @staticmethod
    def _msg_to_dict(msg: Any) -> dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        role = getattr(msg, "role", None)
        if not role:
            typ = getattr(msg, "type", "unknown")
            # Normalise LangChain type names to OpenAI role names
            role = {"ai": "assistant", "human": "user", "tool": "tool", "system": "system"}.get(typ, typ)
        content = getattr(msg, "content", "") or ""
        result: dict[str, Any] = {"role": role, "content": content}
        tc = getattr(msg, "tool_calls", None)
        if tc:
            result["tool_calls"] = tc
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id:
            result["tool_call_id"] = tc_id
        tc_name = getattr(msg, "name", None)
        if tc_name:
            result["name"] = tc_name
        return result

    def _finalize_node(self, state: SDRState) -> dict[str, Any]:
        """Pass-through node: LLM reply generation is done in chat_stream() for proper streaming."""
        return {
            "final_reply": state.get("final_reply", ""),
            "next_action": "end",
            "timing": dict(state.get("timing", {})),
        }

    def chat(
        self,
        session_id: str,
        instruction: str,
        mode: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        reply_parts: list[str] = []
        tool_logs: list[str] = []
        final_payload: dict[str, Any] | None = None

        for event in self.chat_stream(
            session_id=session_id,
            instruction=instruction,
            mode=mode,
            temperature=temperature,
        ):
            event_type = event.get("type")
            if event_type == "delta" and event.get("text"):
                reply_parts.append(str(event["text"]))
            elif event_type == "tool_log" and event.get("line"):
                tool_logs.append(str(event["line"]))
            elif event_type == "done":
                final_payload = event
            elif event_type == "error":
                selected_mode = self.normalize_mode(mode)
                return {
                    "status": "error",
                    "message": str(event.get("message") or "Unknown error"),
                    "reply": _plain_text_reply("".join(reply_parts)),
                    "hardware_logs": tool_logs,
                    "updated_history": self.memory.get_history(session_id),
                    "active_mode": selected_mode,
                    "active_model": self.resolve_model_name(selected_mode),
                    "active_temperature": self.resolve_temperature(selected_mode, temperature),
                }

        if final_payload is None:
            selected_mode = self.normalize_mode(mode)
            return {
                "status": "success",
                "reply": _plain_text_reply("".join(reply_parts)),
                "hardware_logs": tool_logs,
                "updated_history": self.memory.get_history(session_id),
                "active_mode": selected_mode,
                "active_model": self.resolve_model_name(selected_mode),
                "active_temperature": self.resolve_temperature(selected_mode, temperature),
                "timing": None,
            }

        return {
            "status": final_payload.get("status", "success"),
            "reply": _plain_text_reply(final_payload.get("reply", "".join(reply_parts))),
            "hardware_logs": final_payload.get("hardware_logs", tool_logs),
            "updated_history": final_payload.get(
                "updated_history", self.memory.get_history(session_id)
            ),
            "active_mode": final_payload.get("active_mode", self.normalize_mode(mode)),
            "active_model": final_payload.get("active_model", self.resolve_model_name(mode)),
            "active_temperature": final_payload.get(
                "active_temperature", self.resolve_temperature(mode, temperature)
            ),
            "active_route": final_payload.get("active_route", "langgraph"),
            "timing": final_payload.get("timing"),
        }

    def chat_stream(
        self,
        session_id: str,
        instruction: str,
        mode: str,
        temperature: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self.ollama_client is None:
            yield {
                "type": "error",
                "message": "未安装 OpenAI SDK，请先执行 pip install -r requirements.txt。",
            }
            return

        selected_mode = self.normalize_mode(mode)
        active_temperature = self.resolve_temperature(selected_mode, temperature)
        active_model = self.resolve_model_name(selected_mode)

        yield {
            "type": "meta",
            "active_mode": selected_mode,
            "active_model": active_model,
            "active_temperature": active_temperature,
            "active_route": "langgraph",
        }

        route = self._classify_route(instruction)

        if route == "chat":
            # Fast path: no graph, direct Ollama streaming
            try:
                system_msg = SystemMessage(content=build_system_prompt())
                messages = [system_msg, HumanMessage(content=instruction)]
                stream = self.ollama_client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=active_temperature,
                    stream=True,
                )
                reply_parts: list[str] = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        reply_parts.append(text)
                        yield {"type": "delta", "text": text}
                final_reply = _plain_text_reply("".join(reply_parts))
            except Exception as exc:
                yield {"type": "error", "message": str(exc)}
                return

            history = self.memory.get_history(session_id)
            history.append({"role": "user", "content": instruction})
            history.append({"role": "assistant", "content": final_reply})
            self.memory.replace(session_id, history)
            yield {
                "type": "done",
                "status": "success",
                "reply": final_reply,
                "hardware_logs": [],
                "updated_history": history,
                "active_mode": selected_mode,
                "active_model": active_model,
                "active_temperature": active_temperature,
                "active_route": "chat",
                "timing": {"route": "chat", "final_reply_source": "stream_direct"},
            }
            return

        initial_state = self._build_initial_state(
            session_id, instruction, mode, active_temperature,
        )

        exec_logs: list[str] = []
        final_reply = ""
        hw_results: list[dict[str, Any]] = []
        knowledge_text = ""
        request_started_at = time.perf_counter()

        try:
            config: SDRGraphConfig = {"configurable": {"thread_id": session_id}}
            for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    node_logs = node_output.get("tool_logs", [])
                    for log_line in node_logs:
                        exec_logs.append(log_line)
                        yield {"type": "tool_log", "line": log_line}

                    # Collect results from nodes
                    if node_name == "hardware_tools":
                        hw_results = node_output.get("hardware_results", hw_results)
                    elif node_name == "knowledge_agent":
                        knowledge_text = node_output.get("knowledge_context", knowledge_text)

                    if node_name == "finalize":
                        final_reply = node_output.get("final_reply", "")
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        # Generate natural-language reply via streaming LLM call
        if not final_reply and (hw_results or knowledge_text):
            try:
                # Use gemma3 for fast summarisation (no think block, ~3s per reply)
                summary_model = "gemma3:latest"

                # Build a minimal summarisation prompt
                intro = "请用中文一句话总结以下SDR实验结果。只提及实际数据，不编造。"
                if hw_results:
                    intro += "\n硬件结果:\n" + json.dumps(hw_results, ensure_ascii=False, indent=2)
                if knowledge_text:
                    intro += "\n知识库检索:\n" + knowledge_text

                reply_parts: list[str] = []
                stream = self.ollama_client.chat.completions.create(
                    model=summary_model,
                    messages=[{"role": "user", "content": intro}],
                    temperature=0.1,
                    stream=True,
                    timeout=30.0,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        reply_parts.append(text)
                        yield {"type": "delta", "text": text}
                final_reply = "".join(reply_parts)
            except Exception:
                # Fallback to active model or simple summary
                try:
                    if hw_results:
                        resp = self.ollama_client.chat.completions.create(
                            model=active_model,
                            messages=[{
                                "role": "user",
                                "content": "用中文一句话总结: " + json.dumps(hw_results, ensure_ascii=False),
                            }],
                            temperature=0.1,
                            stream=False,
                            timeout=60.0,
                        )
                        final_reply = resp.choices[0].message.content or ""
                    else:
                        final_reply = knowledge_text or ""
                except Exception:
                    if hw_results:
                        final_reply = "\n".join(
                            f"{r['tool']}: {r.get('result', {}).get('message', '')}"
                            for r in hw_results
                        )
                    else:
                        final_reply = knowledge_text or "操作完成。"

        final_reply = _plain_text_reply(final_reply)

        history = self.memory.get_history(session_id)
        history.append({"role": "user", "content": instruction})
        history.append({"role": "assistant", "content": final_reply})
        self.memory.replace(session_id, history)

        timing = {
            "route": "langgraph",
            "total_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
            "final_reply_source": "graph_finalize",
        }

        yield {
            "type": "done",
            "status": "success",
            "reply": final_reply,
            "hardware_logs": exec_logs,
            "updated_history": history,
            "active_mode": selected_mode,
            "active_model": active_model,
            "active_temperature": active_temperature,
            "active_route": "langgraph",
            "timing": timing,
        }
