# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATED — replaced by src/agent/loop.py
#
# This file is no longer used by the main code path.  The new Claude Code
# architecture uses AgentLoop (src/agent/loop.py) with MCP tool registry
# instead of regex-based routing + ToolRegistry.
#
# Kept for reference and potential fallback during migration.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Literal

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from config.settings import Settings
from core.memory import ConversationMemory
from prompts.system_prompts import build_system_prompt
from tools import ToolRegistry


RouteName = Literal["hardware", "knowledge", "chat"]
HardwareTaskName = Literal[
    "stop_hardware",
    "rx_noise_scan",
    "fixed_text_fsk",
    "adaptive_text_transmit",
    "tone_visualization",
    "cognitive_text_transmit",
    "cognitive_adaptive_text",
    "cognitive_tone_visualization",
]


@dataclass(frozen=True)
class HardwareTaskIntent:
    task: HardwareTaskName
    tool_name: str
    args: dict[str, Any]
    description: str

HARDWARE_TOOL_CATEGORY = "hardware"
KNOWLEDGE_TOOL_CATEGORY = "knowledge"
KNOWLEDGE_TOOL_NAME = "search_sdr_knowledge"
STOP_HARDWARE_TOOL_NAME = "stop_hardware_task"
DEFAULT_CENTER_FREQ_HZ = 2.4e9
DEFAULT_TEXT_MODULATION = "2-FSK"
DEFAULT_TONE_FREQ_HZ = 100e3
_HARDWARE_TASK_DESCRIPTIONS: dict[HardwareTaskName, str] = {
    "stop_hardware": "停止当前占用 USRP 的后台物理层任务",
    "rx_noise_scan": "接收端底噪/功率测量实验，只打开 RX 接收链路",
    "fixed_text_fsk": "固定调制文本无线发送实验，默认调制为 2-FSK",
    "adaptive_text_transmit": "先探测链路质量，再按 SNR 选择调制方式的文本发送实验",
    "tone_visualization": "连续单音/正弦波发射与频谱可视化实验",
    "cognitive_text_transmit": "先搜索较干净频点，再执行固定调制文本发送的认知无线实验，默认调制为 2-FSK",
    "cognitive_adaptive_text": "先搜索较干净频点，再按链路质量自适应发送文本的认知无线实验",
    "cognitive_tone_visualization": "先搜索较干净频点，再启动单音频谱观察的认知无线实验",
}

_TOOL_NAME_HINT = re.compile(
    r"(search_sdr_knowledge|tone_loopback|perform_physical|text_fsk|adaptive_modulation|"
    r"auto_optimal|stop_hardware)",
    re.I,
)
_HARDWARE_OBJECT_HINT = re.compile(
    r"(usrp|硬件|物理层|tone|回环|链路|波形|频谱|底噪|噪声|文本收发|文本发送|"
    r"正弦|正弦波|单音|连续波|信号|sine|sinusoid|fsk|bpsk|qpsk|16-qam|"
    r"频点|频率|频段|信道|干净信道|最优信道|干净频点|最优频点|增益|采样率|"
    r"可视化|框图|无线|发出去)",
    re.I,
)
_HARDWARE_EXECUTION_HINT = re.compile(
    r"(请|帮我|给我|麻烦|直接|立即|现在)?[\s:：]*(启动|停止|执行|发送|发射|传输|"
    r"发\s*(?=正弦|单音|tone|sine|信号|波|[A-Za-z0-9\"'“‘])|发出去|输出|收发|扫描|扫频|扫噪|开始|"
    r"扫.{0,4}频|找.{0,8}(干净|最优|最佳)|关闭|打开|测量|测试|显示|查看|运行)",
    re.I,
)
_EXPLANATION_HINT = re.compile(
    r"(什么是|是什么|为什么|为何|请解释|解释一下|说明一下|介绍一下|介绍|"
    r"原理|流程|区别|比较|文档|手册|资料|知识库|参数含义|怎么理解|如何理解|"
    r"怎么用|如何使用)",
    re.I,
)
_THEORY_ONLY_SHORT = re.compile(
    r"^(什么是|是什么|为何|为什么|请解释|解释一下|说明一下|介绍|辨析|比较|区别)[\s\S]{0,120}[?？。]?$"
)
_IDENTITY_QUERY_HINT = re.compile(
    r"(你是谁|你的身份|你是什么模型|你是哪个模型|你用的什么模型|你是什么大模型|"
    r"who are you|what model are you|which model are you)",
    re.I,
)
_TEXT_PAYLOAD_TOOL_NAMES = {
    "text_fsk_send_and_receive",
    "adaptive_modulation_transmit",
    "auto_optimal_transmit",
}
_TEXT_FIELD_ALIASES = ("message", "content", "payload", "data", "msg")
_TEXT_QUOTE_HINT = re.compile(r"[\"'“‘](.+?)[\"'”’]")
_TEXT_EXPLICIT_HINTS = (
    re.compile(
        r"把\s*([^，。；;\n]+?)\s*(?:通过|用).{0,8}?(?:无线|fsk|链路).{0,8}?发出去",
        re.I,
    ),
    re.compile(
        r"(?:文本|内容|消息|字符串|message|text)\s*(?:为|是|:|：|=)\s*([^，。；;\n]+)",
        re.I,
    ),
    re.compile(
        r"(?:发送|发射|收发|传输).{0,8}?(?:文本|内容|消息|字符串|message|text)"
        r"\s*(?:为|是|:|：|=)?\s*([^，。；;\n]+)",
        re.I,
    ),
    re.compile(r"(?:发送|发射|收发|传输)\s*([^，。；;\n]+)", re.I),
    re.compile(r"(?:给我|帮我|请)?\s*发\s*([^，。；;\n]+)", re.I),
)
_TEXT_PARAM_BOUNDARY = re.compile(
    r"(中心频率|频率|采样率|符号率|频偏|增益|带宽|信道|频点|搜索跨度|"
    r"tone|Tone|正弦|单音|hz|Hz|kHz|MHz|GHz|db|dB)"
)
_TONE_REQUEST_HINT = re.compile(r"(tone|正弦|正弦波|单音|连续波|sine|sinusoid)", re.I)
_VISUALIZATION_REQUEST_HINT = re.compile(
    r"(可视化|可视化框|框图|图形|窗口|波形|频谱|监视|显示|认知|认知型|干净信道|"
    r"最优信道|最佳信道|干净频点|最优频点)",
    re.I,
)
_SIGNAL_VISUALIZATION_HINT = re.compile(
    r"(可视化|可视化框|框图|图形|窗口|波形|频谱|监视|显示)",
    re.I,
)
_STOP_REQUEST_HINT = re.compile(r"(停止|关闭|终止|释放|stop)", re.I)
_SCAN_REQUEST_HINT = re.compile(r"(扫频|扫描|扫噪|扫.{0,4}频|底噪|噪声|测量|测试)", re.I)
_OPTIMAL_REQUEST_HINT = re.compile(
    r"(干净信道|干净频段|干净频带|最优信道|最佳信道|最优频段|最佳频段|干净频点|"
    r"最优频点|最佳频点|认知|找.{0,12}(干净|最优|最佳)|"
    r"(选择|挑选|寻找|搜索|切换到|切到).{0,12}(干净|最优|最佳).{0,8}(频段|频带|信道|频点|频率))",
    re.I,
)
_ADAPTIVE_REQUEST_HINT = re.compile(
    r"(自适应|adaptive|amc|自动选择|自动切换|调整调制|信道质量|链路质量|按\s*snr|根据\s*snr|snr)",
    re.I,
)
_DIRECT_MODULATION_HINT = re.compile(
    r"(2\s*[- ]?\s*fsk|fsk|bpsk|qpsk|16\s*[- ]?\s*qam|16qam|qam16)",
    re.I,
)
_TEXT_TRANSMIT_HINT = re.compile(
    r"(文本|消息|字符串|message|text|发送|发射|传输|收发|无线|发出去|fsk|发\s*(?=[A-Za-z0-9\"'“‘]))",
    re.I,
)
_CENTER_FREQ_HINT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ghz|g|mhz|m|khz|k|hz)",
    re.I,
)
_CENTER_FREQ_CONTEXT_HINT = re.compile(
    r"(?:中心频率|载频|射频|工作频率|工作频点|频点|center\s*(?:freq(?:uency)?)?|carrier|rf)"
    r"\s*(?:为|是|:|：|=|在|到)?\s*(\d+(?:\.\d+)?)\s*(ghz|g|mhz|m|khz|k|hz)",
    re.I,
)
_TONE_FREQ_CONTEXT_HINTS = (
    re.compile(
        r"(?:tone|正弦波?|单音|连续波|基带(?:频率)?|音调)"
        r".{0,12}?(\d+(?:\.\d+)?)\s*(ghz|g|mhz|m|khz|k|hz)",
        re.I,
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(ghz|g|mhz|m|khz|k|hz).{0,8}?"
        r"(?:tone|正弦波?|单音|连续波)",
        re.I,
    ),
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


class SDRAgentService:
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
    ) -> None:
        self.settings = settings
        self.tool_registry = tool_registry
        self.memory = memory
        self.client = None
        if OpenAI is not None:
            self.client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.ollama_base_url,
            )
        self._tool_support_cache: dict[str, bool] = {}

    def normalize_mode(self, mode: str) -> str:
        mode_key = (mode or "think").strip().lower()
        return mode_key if mode_key in self.settings.model_map else "think"

    def resolve_model_name(self, mode: str) -> str:
        return self.settings.model_map[self.normalize_mode(mode)]

    def resolve_temperature(self, mode: str, override: float | None) -> float:
        if override is not None:
            return max(0.0, min(2.0, float(override)))
        normalized_mode = self.normalize_mode(mode)
        return float(self.settings.temperature_map[normalized_mode])

    @staticmethod
    def _elapsed_ms(started_at: float, finished_at: float | None = None) -> float:
        stop = time.perf_counter() if finished_at is None else finished_at
        return round((stop - started_at) * 1000.0, 2)

    @staticmethod
    def _resolve_route(instruction: str) -> RouteName:
        text = instruction.strip()
        if not text:
            return "chat"

        lowered = text.lower()
        asks_explanation = bool(_THEORY_ONLY_SHORT.match(text) or _EXPLANATION_HINT.search(text))
        mentions_tool_name = bool(_TOOL_NAME_HINT.search(lowered))
        mentions_hardware = bool(
            mentions_tool_name
            or _HARDWARE_OBJECT_HINT.search(lowered)
            or _HARDWARE_EXECUTION_HINT.search(text)
        )
        explicit_execution = bool(_HARDWARE_EXECUTION_HINT.search(text)) or (
            mentions_tool_name and not asks_explanation
        )

        if asks_explanation and not explicit_execution:
            return "knowledge" if mentions_hardware else "chat"
        if explicit_execution and mentions_hardware:
            return "hardware"
        if mentions_hardware and not asks_explanation:
            return "hardware"
        return "chat"

    def _trim_history(
        self,
        session_id: str,
        *,
        route: RouteName,
        mode: str,
    ) -> list[dict[str, str]]:
        history = self.memory.get_history(session_id)
        if route == "hardware":
            return history[-self.settings.session_history_limit :]

        limit = (
            self.settings.fast_history_messages
            if self.normalize_mode(mode) == "fast"
            else self.settings.general_history_messages
        )
        if limit <= 0:
            return []
        return history[-limit:]

    @staticmethod
    def _is_identity_query(instruction: str) -> bool:
        return bool(_IDENTITY_QUERY_HINT.search(instruction.strip()))

    def _build_identity_reply(self, *, model_name: str, mode: str) -> str:
        mode_label = self.normalize_mode(mode)
        return (
            "我是 OpenClaw SDR 智能控制中枢，负责协助完成 USRP 物理层实验、"
            "状态查询、文本链路验证和 SDR 理论知识检索。"
            f"当前这轮对话由本地部署模型 {model_name} 驱动，当前模式为 {mode_label}。"
        )

    @staticmethod
    def _plain_text_reply(text: str) -> str:
        """Convert common Markdown-shaped model output into readable plain Chinese text."""
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

    @staticmethod
    def _is_tools_unsupported_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "does not support tools" in message or "support tools" in message

    def _tool_fallback_model(self, requested_model: str) -> str:
        fallback_model = self.resolve_model_name("think")
        return fallback_model if fallback_model and fallback_model != requested_model else requested_model

    def _resolve_active_model(self, *, route: RouteName, requested_model: str) -> str:
        if route == "chat":
            return requested_model

        if self._tool_support_cache.get(requested_model, True):
            return requested_model
        return self._tool_fallback_model(requested_model)

    @staticmethod
    def _freq_match_to_hz(match: re.Match[str]) -> float:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"ghz", "g"}:
            return value * 1e9
        if unit in {"mhz", "m"}:
            return value * 1e6
        if unit in {"khz", "k"}:
            return value * 1e3
        return value

    @classmethod
    def _extract_center_freq_hz(cls, instruction: str) -> float | None:
        explicit = _CENTER_FREQ_CONTEXT_HINT.search(instruction)
        if explicit:
            return cls._freq_match_to_hz(explicit)

        for match in _CENTER_FREQ_HINT.finditer(instruction):
            freq_hz = cls._freq_match_to_hz(match)
            unit = match.group(2).lower()
            if unit in {"ghz", "g"} or freq_hz >= 100e6:
                return freq_hz
        return None

    @classmethod
    def _extract_tone_freq_hz(cls, instruction: str) -> float | None:
        for pattern in _TONE_FREQ_CONTEXT_HINTS:
            match = pattern.search(instruction)
            if match:
                freq_hz = cls._freq_match_to_hz(match)
                if freq_hz < 10e6:
                    return freq_hz

        if not _TONE_REQUEST_HINT.search(instruction):
            return None

        low_freqs: list[float] = []
        for match in _CENTER_FREQ_HINT.finditer(instruction):
            freq_hz = cls._freq_match_to_hz(match)
            if freq_hz < 10e6:
                low_freqs.append(freq_hz)
        return low_freqs[-1] if low_freqs else None

    @staticmethod
    def _normalize_requested_modulation(raw: Any) -> str | None:
        normalized = str(raw or "").strip().upper()
        compact = normalized.replace("_", "").replace("-", "").replace(" ", "")
        if compact in {"", "FSK", "2FSK"}:
            return DEFAULT_TEXT_MODULATION
        if compact == "BPSK":
            return "BPSK"
        if compact == "QPSK":
            return "QPSK"
        if compact in {"16QAM", "QAM16"}:
            return "16-QAM"
        return None

    @classmethod
    def _extract_requested_modulation(cls, instruction: str) -> str | None:
        match = _DIRECT_MODULATION_HINT.search(instruction)
        if not match:
            return None
        return cls._normalize_requested_modulation(match.group(1))

    @classmethod
    def _apply_common_hardware_args(
        cls,
        *,
        instruction: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        patched = dict(args)
        center_freq_hz = cls._extract_center_freq_hz(instruction)
        if center_freq_hz is not None:
            patched["center_freq_hz"] = center_freq_hz
        tone_freq_hz = cls._extract_tone_freq_hz(instruction)
        if tone_freq_hz is not None:
            patched["tone_freq_hz"] = tone_freq_hz
        return patched

    @classmethod
    def _classify_hardware_intent(cls, instruction: str) -> HardwareTaskIntent | None:
        text = instruction.strip()
        if not text:
            return None

        args = cls._apply_common_hardware_args(instruction=text, args={})

        if _STOP_REQUEST_HINT.search(text):
            task: HardwareTaskName = "stop_hardware"
            return HardwareTaskIntent(
                task=task,
                tool_name=STOP_HARDWARE_TOOL_NAME,
                args={},
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        wants_optimal = bool(_OPTIMAL_REQUEST_HINT.search(text))
        wants_tone = bool(_TONE_REQUEST_HINT.search(text))
        wants_visualization = bool(_VISUALIZATION_REQUEST_HINT.search(text))
        wants_signal_visualization = bool(_SIGNAL_VISUALIZATION_HINT.search(text))
        requested_modulation = cls._extract_requested_modulation(text)
        wants_adaptive = bool(_ADAPTIVE_REQUEST_HINT.search(text)) and requested_modulation is None
        wants_text_tx = bool(_TEXT_TRANSMIT_HINT.search(text))
        wants_scan = bool(_SCAN_REQUEST_HINT.search(text))

        if wants_optimal:
            tx_text = "" if wants_tone else cls._extract_text_payload(text)
            if tx_text:
                args["text"] = tx_text
            if wants_tone or wants_signal_visualization or not tx_text:
                args["enable_tone"] = True
            elif requested_modulation:
                args["modulation"] = requested_modulation
            if wants_adaptive:
                args["use_adaptive_modulation"] = True
            if wants_tone or wants_signal_visualization or not tx_text:
                task = "cognitive_tone_visualization"
            elif wants_adaptive:
                task = "cognitive_adaptive_text"
            else:
                task = "cognitive_text_transmit"
            return HardwareTaskIntent(
                task=task,
                tool_name="auto_optimal_transmit",
                args=args,
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        if wants_tone or wants_visualization:
            task = "tone_visualization"
            return HardwareTaskIntent(
                task=task,
                tool_name="tone_loopback_visualize",
                args=args,
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        if wants_adaptive:
            args["text"] = cls._extract_text_payload(text) or "OpenClaw SDR adaptive test"
            task = "adaptive_text_transmit"
            return HardwareTaskIntent(
                task=task,
                tool_name="adaptive_modulation_transmit",
                args=args,
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        if wants_scan:
            task = "rx_noise_scan"
            return HardwareTaskIntent(
                task=task,
                tool_name="perform_physical_scan",
                args=args,
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        if wants_text_tx:
            args["text"] = cls._extract_text_payload(text) or text
            if requested_modulation:
                args["modulation"] = requested_modulation
            task = "fixed_text_fsk"
            return HardwareTaskIntent(
                task=task,
                tool_name="text_fsk_send_and_receive",
                args=args,
                description=_HARDWARE_TASK_DESCRIPTIONS[task],
            )

        return None

    @classmethod
    def _infer_hardware_tool_call(cls, instruction: str) -> tuple[str, dict[str, Any]] | None:
        intent = cls._classify_hardware_intent(instruction)
        if intent is None:
            return None
        return intent.tool_name, dict(intent.args)

    @staticmethod
    def _route_note(route: RouteName) -> str:
        if route == "hardware":
            return (
                "当前请求已被系统判定为明确硬件执行动作。"
                "第一步必须调用最小必要的硬件技能，禁止跳过技能直接回答，"
                "也不要调用知识检索工具。"
            )
        if route == "knowledge":
            return (
                "当前请求已被系统判定为理论/资料问答。"
                "第一步必须调用 `search_sdr_knowledge`，禁止跳过知识检索直接作答，"
                "也不要调用任何硬件技能。"
            )
        return "当前请求已被系统判定为普通对话。不要调用任何工具，直接回答。"

    def _build_messages(
        self,
        *,
        history: list[dict[str, str]],
        instruction: str,
        route: RouteName,
        extra_system_messages: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "system", "content": self._route_note(route)},
        ]
        for item in extra_system_messages or []:
            messages.append({"role": "system", "content": item})
        messages.extend(history)
        messages.append({"role": "user", "content": instruction})
        return messages

    def _create_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        temperature: float,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        assert self.client is not None
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def _chat_completion_with_tools(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any],
        temperature: float,
    ) -> Any:
        try:
            return self._create_completion(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )
        except Exception:
            if tool_choice != "auto":
                return self._create_completion(
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,
                )
            raise

    def _iter_stream_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
    ) -> Iterator[dict[str, Any]]:
        """Stream one chat completion; yields client events and a final _assistant sentinel."""
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)
        tc_parts: dict[int, dict[str, str]] = defaultdict(
            lambda: {"id": "", "name": "", "arguments": ""}
        )
        content_parts: list[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = int(getattr(tc, "index", 0) or 0)
                    if getattr(tc, "id", None):
                        tc_parts[index]["id"] = tc.id or ""
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            tc_parts[index]["name"] = fn.name or ""
                        if getattr(fn, "arguments", None):
                            tc_parts[index]["arguments"] += fn.arguments

        tool_calls_list: list[dict[str, Any]] = []
        for idx in sorted(tc_parts.keys()):
            part = tc_parts[idx]
            if not part["name"] and not part["arguments"]:
                continue
            tool_calls_list.append(
                {
                    "id": part["id"],
                    "type": "function",
                    "function": {"name": part["name"], "arguments": part["arguments"]},
                }
            )

        content = self._plain_text_reply("".join(content_parts))
        assistant_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls_list:
            assistant_dict["tool_calls"] = tool_calls_list

        if content:
            yield from self._yield_text_as_deltas(content)
        yield {"type": "_assistant", "message": assistant_dict}

    def _yield_text_as_deltas(self, text: str, chunk: int = 64) -> Iterator[dict[str, Any]]:
        if not text:
            return
        for idx in range(0, len(text), chunk):
            yield {"type": "delta", "text": text[idx : idx + chunk]}

    @staticmethod
    def _is_hardware_busy_result(result: dict[str, Any]) -> bool:
        return result.get("status") == "error" and "硬件正忙" in str(result.get("error", ""))

    @staticmethod
    def _clean_extracted_text(raw: Any) -> str:
        text = str(raw or "").strip().strip("\"'“”‘’")
        text = _TEXT_PARAM_BOUNDARY.split(text, maxsplit=1)[0].strip()
        text = re.sub(r"^(?:文本|内容|消息|字符串|message|text)\s*(?:为|是|:|：|=)?", "", text, flags=re.I)
        text = text.strip().strip("\"'“”‘’")
        if text in {"", "文本", "内容", "消息", "字符串", "数据"}:
            return ""
        return text

    @classmethod
    def _extract_text_payload(cls, instruction: str) -> str:
        quoted = _TEXT_QUOTE_HINT.search(instruction)
        if quoted:
            cleaned = cls._clean_extracted_text(quoted.group(1))
            if cleaned:
                return cleaned

        for pattern in _TEXT_EXPLICIT_HINTS:
            match = pattern.search(instruction)
            if not match:
                continue
            cleaned = cls._clean_extracted_text(match.group(1))
            if cleaned:
                return cleaned
        return ""

    @classmethod
    def _repair_tool_args(
        cls,
        *,
        tool_name: str,
        args: dict[str, Any],
        instruction: str,
    ) -> tuple[dict[str, Any], list[str]]:
        repaired = dict(args)
        notes: list[str] = []
        explicit_fixed_modulation = False

        if tool_name == KNOWLEDGE_TOOL_NAME:
            query_value = str(repaired.get("query") or "").strip()
            if not query_value:
                repaired["query"] = instruction.strip()
                notes.append("已使用用户原始问题补全 `query` 参数。")
            if "top_k" not in repaired:
                repaired["top_k"] = 4 if len(instruction) > 24 else 3
                notes.append("已按问题长度补全 `top_k` 参数。")

        if tool_name in _TEXT_PAYLOAD_TOOL_NAMES:
            tone_only_auto_task = (
                tool_name == "auto_optimal_transmit"
                and _TONE_REQUEST_HINT.search(instruction)
                and not _TEXT_QUOTE_HINT.search(instruction)
                and not re.search(r"(文本|内容|消息|字符串|message|text)", instruction, re.I)
            )
            text_value = "" if tone_only_auto_task else str(repaired.get("text") or "").strip()
            for alias in _TEXT_FIELD_ALIASES:
                alias_value = repaired.pop(alias, None)
                if tone_only_auto_task or text_value or alias_value is None:
                    continue
                text_value = str(alias_value).strip()
                notes.append(f"已将参数 `{alias}` 归一为 `text`。")

            if not text_value and not tone_only_auto_task:
                extracted = cls._extract_text_payload(instruction)
                if extracted:
                    text_value = extracted
                    notes.append("已从用户指令中补全 `text` 参数。")

            if text_value:
                repaired["text"] = text_value
            elif tone_only_auto_task:
                repaired["text"] = ""

        if tool_name in {"text_fsk_send_and_receive", "auto_optimal_transmit"} and not (
            tool_name == "auto_optimal_transmit" and not str(repaired.get("text") or "").strip()
        ):
            requested_modulation = cls._extract_requested_modulation(instruction)
            raw_modulation = repaired.get("modulation")
            normalized_modulation = cls._normalize_requested_modulation(raw_modulation)
            explicit_fixed_modulation = requested_modulation is not None or (
                raw_modulation not in {None, ""} and normalized_modulation is not None
            )
            if requested_modulation is not None:
                repaired["modulation"] = requested_modulation
                if normalized_modulation != requested_modulation:
                    notes.append(f"已按用户指令补全固定调制方式为 `{requested_modulation}`。")
            elif normalized_modulation is not None:
                repaired["modulation"] = normalized_modulation
            else:
                repaired["modulation"] = DEFAULT_TEXT_MODULATION
                if raw_modulation not in {None, ""}:
                    notes.append("检测到无法识别的固定调制方式，已回退到默认文本调制 2-FSK。")
                else:
                    notes.append("未检测到固定调制方式，已使用默认文本调制 2-FSK。")

        if tool_name in {"tone_loopback_visualize", "auto_optimal_transmit"}:
            parsed_center = cls._extract_center_freq_hz(instruction)
            parsed_tone = cls._extract_tone_freq_hz(instruction)

            if parsed_center is not None:
                repaired["center_freq_hz"] = parsed_center
            elif tool_name == "tone_loopback_visualize":
                current_center = float(repaired.get("center_freq_hz") or 0)
                if current_center <= 10e6:
                    repaired["center_freq_hz"] = DEFAULT_CENTER_FREQ_HZ
                    notes.append("未检测到有效射频中心频率，已使用默认中心频率 2.4 GHz。")

            if parsed_tone is not None:
                repaired["tone_freq_hz"] = parsed_tone
            elif tool_name == "tone_loopback_visualize":
                current_tone = float(repaired.get("tone_freq_hz") or 0)
                if current_tone <= 0:
                    repaired["tone_freq_hz"] = DEFAULT_TONE_FREQ_HZ
                    notes.append("未检测到 Tone 频率，已使用默认基带正弦波频率 100 kHz。")

        if tool_name == "auto_optimal_transmit" and (
            _TONE_REQUEST_HINT.search(instruction)
            or (
                _VISUALIZATION_REQUEST_HINT.search(instruction)
                and not str(repaired.get("text") or "").strip()
            )
        ):
            if not bool(repaired.get("enable_tone", False)):
                repaired["enable_tone"] = True
                notes.append("检测到认知选频可视化/Tone 意图，已补全 `enable_tone=true`。")
        if (
            tool_name == "auto_optimal_transmit"
            and bool(repaired.get("use_adaptive_modulation"))
            and explicit_fixed_modulation
        ):
            repaired["use_adaptive_modulation"] = False
            notes.append("检测到用户明确指定了固定调制方式，已关闭按 SNR 自动选调制。")

        return repaired, notes

    @staticmethod
    def _result_requests_visualization(tool_name: str, result: dict[str, Any]) -> bool:
        if result.get("status") != "success":
            return False
        if tool_name == "tone_loopback_visualize":
            return True
        tone_result = result.get("tone_result")
        if isinstance(tone_result, dict) and tone_result.get("status") == "success":
            return True
        return bool(result.get("visualization_active", False))

    def _execute_tool_with_recovery(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        result = self.tool_registry.execute(tool_name, args)
        logs: list[str] = []
        tool_spec = self.tool_registry.get_tool(tool_name)

        if (
            tool_spec is not None
            and tool_spec.category == HARDWARE_TOOL_CATEGORY
            and tool_name != STOP_HARDWARE_TOOL_NAME
            and self._is_hardware_busy_result(result)
            and self.tool_registry.has_tool(STOP_HARDWARE_TOOL_NAME)
        ):
            logs.append("检测到 USRP 正忙，系统自动尝试停止后台任务后重试一次。")
            stop_result = self.tool_registry.execute(STOP_HARDWARE_TOOL_NAME, {})
            logs.append(
                f"自动调用 {STOP_HARDWARE_TOOL_NAME}: "
                f"{stop_result.get('message') or stop_result.get('error') or stop_result.get('status')}"
            )
            retried_result = self.tool_registry.execute(tool_name, args)
            logs.append(
                f"自动重试 {tool_name}: "
                f"{retried_result.get('message') or retried_result.get('error') or retried_result.get('status')}"
            )
            wrapped_result = dict(retried_result)
            wrapped_result["recovery"] = {
                "auto_stop_attempted": True,
                "stop_result": stop_result,
            }
            result = wrapped_result

        return result, logs

    def _build_knowledge_context(self, result: dict[str, Any]) -> str:
        hits = result.get("results") or []
        if not hits:
            return (
                "本地知识库没有检索到直接匹配结果。请基于已有上下文谨慎回答，"
                "并明确说明本次没有命中本地资料。"
            )

        lines = [
            "以下是本地知识库检索结果，请优先基于这些资料组织答案，并尽量指出来源："
        ]
        for idx, item in enumerate(hits, start=1):
            lines.append(
                f"{idx}. 来源: {item.get('source', 'unknown')} | "
                f"得分: {item.get('score', '--')}\n"
                f"摘要: {item.get('snippet', '')}"
            )
        return "\n".join(lines)

    def _route_tools(self, route: RouteName) -> list[dict[str, Any]]:
        if route == "hardware":
            return self.tool_registry.as_openai_tools(
                categories={HARDWARE_TOOL_CATEGORY}
            )
        if route == "knowledge":
            return self.tool_registry.as_openai_tools(
                names={KNOWLEDGE_TOOL_NAME},
                categories={KNOWLEDGE_TOOL_CATEGORY},
            )
        return []

    @staticmethod
    def _tool_retry_note(route: RouteName) -> str:
        if route == "hardware":
            return (
                "系统校验：本轮属于硬件执行请求。你必须先返回一个或多个硬件技能调用，"
                "禁止直接给出自然语言结论或假设硬件已经执行。"
            )
        if route == "knowledge":
            return (
                "系统校验：本轮属于理论/资料问答。你必须先调用 `search_sdr_knowledge`，"
                "拿到检索结果后再组织回答，禁止直接凭记忆作答。"
            )
        return "系统校验：如需工具，必须先返回 tool call。"

    def _format_tool_result_for_model(self, tool_name: str, result: dict[str, Any]) -> str:
        if tool_name == KNOWLEDGE_TOOL_NAME:
            return self._build_knowledge_context(result)
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _format_hz(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "未提供"
        if abs(numeric) >= 1e9:
            return f"{numeric / 1e9:.3f} GHz"
        if abs(numeric) >= 1e6:
            return f"{numeric / 1e6:.3f} MHz"
        if abs(numeric) >= 1e3:
            return f"{numeric / 1e3:.3f} kHz"
        return f"{numeric:.2f} Hz"

    @staticmethod
    def _format_db(value: Any) -> str:
        try:
            return f"{float(value):.2f} dB"
        except (TypeError, ValueError):
            return "未提供"

    @staticmethod
    def _describe_noise_power(power_db: Any) -> str:
        try:
            power = float(power_db)
        except (TypeError, ValueError):
            return "当前没有可用的噪声功率读数。"

        if power <= -70:
            quality = "很干净"
        elif power <= -55:
            quality = "较干净"
        elif power <= -40:
            quality = "中等"
        else:
            quality = "偏嘈杂"
        return f"测得平均接收功率为 {power:.2f} dB，按当前未校准的 IQ 功率估计，信道噪声水平属于{quality}；数值越低，通常表示这个频点越安静。"

    @staticmethod
    def _modulation_reason(modulation: Any, snr_db: Any) -> str:
        try:
            snr = float(snr_db)
        except (TypeError, ValueError):
            return "未拿到有效 SNR，因此只能依据工具返回的调制结果说明。"

        selected = str(modulation or "").upper()
        if selected == "BPSK":
            return f"探测 SNR 为 {snr:.2f} dB，低于 -18 dB 阈值，所以选择 BPSK，以优先保证鲁棒性。"
        if selected == "QPSK":
            return f"探测 SNR 为 {snr:.2f} dB，处在 -18 dB 到 -15 dB 区间，所以选择 QPSK，在可靠性和速率之间折中。"
        if selected == "16-QAM":
            return f"探测 SNR 为 {snr:.2f} dB，高于 -15 dB 阈值，所以选择 16-QAM，以利用较好的链路质量提高频谱效率。"
        return f"探测 SNR 为 {snr:.2f} dB，工具返回的调制方式是 {modulation}。"

    def _build_scan_reply(self, result: dict[str, Any], args: dict[str, Any]) -> str:
        center = self._format_hz(result.get("center_freq_hz", args.get("center_freq_hz", 2.4e9)))
        bandwidth = self._format_hz(result.get("bandwidth_hz", args.get("bandwidth_hz", 1e6)))
        channel = result.get("chan", args.get("chan", 0))
        duration = result.get("duration_s", args.get("duration_s", 0.2))
        captured = result.get("captured_samples", 0)
        noise = self._describe_noise_power(result.get("measured_power_dB"))
        return (
            f"扫噪已完成，中心频率是 {center}，接收带宽是 {bandwidth}，接收通道是 {channel}，采样时长约 {float(duration):.2f} 秒，"
            f"实际捕获 {captured} 个 IQ 样本。{noise}"
        )

    def _build_text_fsk_reply(self, result: dict[str, Any], args: dict[str, Any]) -> str:
        sent_text = result.get("sent_text") or args.get("text") or ""
        center = self._format_hz(result.get("center_freq_hz", args.get("center_freq_hz", 2.4e9)))
        modulation = result.get("selected_modulation") or args.get("modulation") or DEFAULT_TEXT_MODULATION
        samp_rate = self._format_hz(args.get("samp_rate", 1e6))
        sym_rate = self._format_hz(args.get("sym_rate", 10e3))
        f_dev = self._format_hz(args.get("f_dev", 25e3))
        tx_gain = args.get("tx_gain", 20.0)
        rx_gain = args.get("rx_gain", 20.0)
        captured = result.get("captured_samples", 0)
        decoded = result.get("decoded_texts") or []
        decoded_ok = bool(result.get("decoded_ok"))
        decode_supported = bool(result.get("decode_supported", modulation == DEFAULT_TEXT_MODULATION))
        estimated_snr = result.get("estimated_snr_db")

        if str(modulation).upper() != DEFAULT_TEXT_MODULATION.upper():
            snr_text = ""
            if estimated_snr is not None:
                snr_text = f" 当前回采链路估计 SNR 约为 {float(estimated_snr):.2f} dB。"
            return (
                f"文本链路已执行，发送内容是 {sent_text}。本次按你指定的 {modulation} 调制发送，中心频率是 {center}，"
                f"采样率是 {samp_rate}，发射增益和接收增益分别是 {tx_gain} dB 和 {rx_gain} dB。"
                f"接收端捕获了 {captured} 个 IQ 样本。"
                f"{'当前固定调制路径已完成发射与回采，但还没有做文本级解码确认。' if not decode_supported else ''}{snr_text}"
            )

        if decoded_ok:
            decoded_text = "、".join(str(item) for item in decoded)
            link_status = f"接收端成功解码，解码文本为 {decoded_text}。"
        else:
            link_status = "接收端这次没有成功解码出文本，可能与回环路径、增益、同步或当前信道噪声有关。"

        return (
            f"文本链路已执行，发送内容是 {sent_text}。本次使用 2-FSK 调制，中心频率是 {center}，"
            f"采样率是 {samp_rate}，符号率是 {sym_rate}，频偏是 {f_dev}，发射增益和接收增益分别是 {tx_gain} dB 和 {rx_gain} dB。"
            f"接收端捕获了 {captured} 个 IQ 样本。{link_status}"
        )

    def _build_adaptive_reply(self, result: dict[str, Any], args: dict[str, Any]) -> str:
        center = self._format_hz(result.get("center_freq_hz", args.get("center_freq_hz", 2.4e9)))
        selected = result.get("selected_modulation", "未知调制")
        snr = result.get("probe_snr_db")
        text = args.get("text") or ""
        reason = self._modulation_reason(selected, snr)
        final_result = result.get("final_result") if isinstance(result.get("final_result"), dict) else {}
        final_snr = final_result.get("estimated_snr_db")
        final_note = ""
        if final_snr is not None:
            final_note = f"正式发送阶段估计 SNR 为 {self._format_db(final_snr)}。"
        return (
            f"自适应调制发送已完成，发送内容是 {text}，中心频率是 {center}。"
            f"系统先用 BPSK 探测链路质量，然后选择 {selected} 作为正式发送调制方式。"
            f"{reason}{final_note}"
        )

    def _build_auto_optimal_reply(self, result: dict[str, Any], args: dict[str, Any]) -> str:
        best_freq = self._format_hz(result.get("best_freq_hz", args.get("center_freq_hz", 2.4e9)))
        min_power = result.get("min_power_db")
        scan_details = result.get("scan_details") if isinstance(result.get("scan_details"), dict) else {}
        scanned_count = len(scan_details)
        scan_text = (
            f"本次共比较了 {scanned_count} 个候选频点，最低接收功率为 {self._format_db(min_power)}。"
            if scanned_count
            else "本次没有返回详细候选频点列表。"
        )
        reply = (
            f"认知选频已完成，系统选择的较干净频点是 {best_freq}。"
            f"{scan_text}选择依据是接收功率越低，当前频点越可能安静、干扰越少。"
        )

        tx_result = result.get("tx_result")
        if isinstance(tx_result, dict):
            if args.get("use_adaptive_modulation"):
                reply += self._build_adaptive_reply(tx_result, {**args, "center_freq_hz": result.get("best_freq_hz")})
            else:
                reply += self._build_text_fsk_reply(tx_result, {**args, "center_freq_hz": result.get("best_freq_hz")})

        tone_result = result.get("tone_result")
        if isinstance(tone_result, dict):
            if tone_result.get("status") == "success":
                reply += f"同时已经在该频点启动 Tone 可视化，Tone 频率是 {self._format_hz(tone_result.get('tone_freq_hz', args.get('tone_freq_hz', 100e3)))}，波形和频谱会推送到可视化窗口。"
            else:
                reply += f"Tone 可视化启动失败，原因是 {tone_result.get('error') or tone_result.get('message') or '未知错误'}。"

        return reply

    def _build_tool_execution_reply(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        fallback_used: bool = False,
    ) -> str:
        status = result.get("status", "unknown")
        message = result.get("message") or result.get("error") or "技能已返回结果。"

        if status == "success":
            if tool_name == "perform_physical_scan":
                return self._plain_text_reply(self._build_scan_reply(result, args))
            if tool_name == "text_fsk_send_and_receive":
                return self._plain_text_reply(self._build_text_fsk_reply(result, args))
            if tool_name == "adaptive_modulation_transmit":
                return self._plain_text_reply(self._build_adaptive_reply(result, args))
            if tool_name == "auto_optimal_transmit":
                return self._plain_text_reply(self._build_auto_optimal_reply(result, args))
            if tool_name == "tone_loopback_visualize":
                center = self._format_hz(result.get("center_freq_hz", args.get("center_freq_hz", 2.4e9)))
                tone = self._format_hz(result.get("tone_freq_hz", args.get("tone_freq_hz", 100e3)))
                return self._plain_text_reply(
                    f"Tone 可视化已启动，中心频率是 {center}，基带 Tone 频率是 {tone}。系统会持续发送单音并接收回环数据，波形和频谱会显示在可视化窗口中。"
                )

        param_text = "无额外参数" if not args else json.dumps(args, ensure_ascii=False)
        if status == "success":
            return self._plain_text_reply(
                f"已执行技能 {tool_name}。关键参数是 {param_text}。执行结果：{message}"
            )
        return self._plain_text_reply(
            f"技能 {tool_name} 已尝试执行。关键参数是 {param_text}。但实际执行失败，原因是 {message}"
        )

    def chat(
        self,
        session_id: str,
        instruction: str,
        mode: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("未安装 OpenAI SDK，请先执行 `pip install -r requirements.txt`。")

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
                raise RuntimeError(str(event.get("message") or "Unknown error"))

        if final_payload is None:
            selected_mode = self.normalize_mode(mode)
            return {
                "status": "success",
                "reply": self._plain_text_reply("".join(reply_parts)),
                "hardware_logs": tool_logs,
                "updated_history": self.memory.get_history(session_id),
                "active_mode": selected_mode,
                "active_model": self.resolve_model_name(selected_mode),
                "active_temperature": self.resolve_temperature(selected_mode, temperature),
                "timing": None,
            }

        return {
            "status": final_payload.get("status", "success"),
            "reply": self._plain_text_reply(final_payload.get("reply", "".join(reply_parts))),
            "hardware_logs": final_payload.get("hardware_logs", tool_logs),
            "updated_history": final_payload.get("updated_history", self.memory.get_history(session_id)),
            "active_mode": final_payload.get("active_mode", self.normalize_mode(mode)),
            "active_model": final_payload.get("active_model", self.resolve_model_name(mode)),
            "active_temperature": final_payload.get(
                "active_temperature",
                self.resolve_temperature(mode, temperature),
            ),
            "active_route": final_payload.get("active_route", self._resolve_route(instruction)),
            "timing": final_payload.get("timing"),
        }

    def chat_stream(
        self,
        session_id: str,
        instruction: str,
        mode: str,
        temperature: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self.client is None:
            yield {
                "type": "error",
                "message": "未安装 OpenAI SDK，请先执行 `pip install -r requirements.txt`。",
            }
            return

        selected_mode = self.normalize_mode(mode)
        active_temperature = self.resolve_temperature(selected_mode, temperature)
        model_name = self.resolve_model_name(selected_mode)
        route = self._resolve_route(instruction)
        if self._is_identity_query(instruction):
            route = "chat"
        hardware_intent = self._classify_hardware_intent(instruction) if route == "hardware" else None
        active_model_name = self._resolve_active_model(route=route, requested_model=model_name)

        yield {
            "type": "meta",
            "active_mode": selected_mode,
            "active_model": active_model_name,
            "active_temperature": active_temperature,
            "active_route": route,
        }

        history = self._trim_history(
            session_id,
            route=route,
            mode=selected_mode,
        )
        exec_logs: list[str] = []
        final_reply = ""
        current_messages: list[dict[str, Any]] = []
        last_tool_name = ""
        last_tool_args: dict[str, Any] = {}
        last_tool_result: dict[str, Any] = {}
        request_started_at = time.perf_counter()
        timing: dict[str, Any] = {
            "route": route,
            "history_messages": len(history),
            "knowledge_lookup_ms": 0.0,
            "tool_decision_ms": 0.0,
            "tool_execution_ms": 0.0,
            "final_reply_ms": 0.0,
            "first_token_ms": None,
            "planning_rounds": 0,
            "tool_calls": [],
            "final_reply_source": None,
        }

        try:
            if self._is_identity_query(instruction):
                final_reply = self._build_identity_reply(
                    model_name=model_name,
                    mode=selected_mode,
                )
                current_messages = self._build_messages(
                    history=history,
                    instruction=instruction,
                    route="chat",
                )
                current_messages.append({"role": "assistant", "content": final_reply})
                timing["final_reply_source"] = "identity_shortcut"
                timing["first_token_ms"] = 0.0
                timing["final_reply_ms"] = 0.0
                for event in self._yield_text_as_deltas(final_reply):
                    yield event
            elif route in {"knowledge", "hardware"}:
                current_messages = self._build_messages(
                    history=history,
                    instruction=instruction,
                    route=route,
                )
                routed_tools = self._route_tools(route)
                if not routed_tools:
                    raise RuntimeError(f"当前路由 `{route}` 没有可用技能。")

                executed_tool_count = 0
                if hardware_intent is not None:
                    intent_line = f"系统将本次请求理解为：{hardware_intent.description}。"
                    exec_logs.append(intent_line)
                    yield {"type": "tool_log", "line": intent_line}

                for turn in range(self.settings.max_tool_turns):
                    tool_choice: str | dict[str, Any] = "required" if executed_tool_count == 0 else "auto"
                    planning_started_at = time.perf_counter()
                    try:
                        response = self._chat_completion_with_tools(
                            model_name=active_model_name,
                            messages=current_messages,
                            tools=routed_tools,
                            tool_choice=tool_choice,
                            temperature=active_temperature,
                        )
                    except Exception as exc:
                        if not self._is_tools_unsupported_error(exc):
                            raise

                        self._tool_support_cache[active_model_name] = False
                        fallback_model = self._tool_fallback_model(active_model_name)
                        if fallback_model == active_model_name:
                            raise RuntimeError(
                                f"当前模型 `{active_model_name}` 不支持工具调用，且没有可用的回退模型。"
                            ) from exc

                        fallback_line = (
                            f"模型 {active_model_name} 不支持工具调用，系统已自动切换到 "
                            f"{fallback_model} 执行技能。"
                        )
                        exec_logs.append(fallback_line)
                        yield {"type": "tool_log", "line": fallback_line}

                        active_model_name = fallback_model
                        yield {
                            "type": "meta",
                            "active_mode": selected_mode,
                            "active_model": active_model_name,
                            "active_temperature": active_temperature,
                            "active_route": route,
                        }
                        response = self._chat_completion_with_tools(
                            model_name=active_model_name,
                            messages=current_messages,
                            tools=routed_tools,
                            tool_choice=tool_choice,
                            temperature=active_temperature,
                        )
                    timing["tool_decision_ms"] = round(
                        float(timing["tool_decision_ms"]) + self._elapsed_ms(planning_started_at),
                        2,
                    )
                    timing["planning_rounds"] = turn + 1
                    message = response.choices[0].message
                    assistant_dict = self._assistant_message_to_dict(message)
                    tool_calls = assistant_dict.get("tool_calls")
                    current_messages.append(assistant_dict)

                    if not tool_calls:
                        if executed_tool_count == 0:
                            if current_messages and current_messages[-1] is assistant_dict:
                                current_messages.pop()
                            inferred_tool_call = None
                            if route == "hardware":
                                inferred_tool_call = (
                                    (hardware_intent.tool_name, dict(hardware_intent.args))
                                    if hardware_intent is not None
                                    else self._infer_hardware_tool_call(instruction)
                                )
                            elif route == "knowledge":
                                inferred_tool_call = (
                                    KNOWLEDGE_TOOL_NAME,
                                    {
                                        "query": instruction,
                                        "top_k": 4 if len(instruction) > 24 else 3,
                                    },
                                )
                            if inferred_tool_call is not None:
                                tool_name, inferred_args = inferred_tool_call
                                inferred_args, repair_logs = self._repair_tool_args(
                                    tool_name=tool_name,
                                    args=inferred_args,
                                    instruction=instruction,
                                )
                                for repair_line in repair_logs:
                                    exec_logs.append(repair_line)
                                    yield {"type": "tool_log", "line": repair_line}

                                fallback_line = f"系统触发技能 {tool_name}，参数: {inferred_args}"
                                exec_logs.append(fallback_line)
                                yield {"type": "tool_log", "line": fallback_line}

                                tool_started_at = time.perf_counter()
                                result, recovery_logs = self._execute_tool_with_recovery(
                                    tool_name,
                                    inferred_args,
                                )
                                last_tool_name = tool_name
                                last_tool_args = dict(inferred_args)
                                last_tool_result = dict(result)
                                tool_elapsed_ms = self._elapsed_ms(tool_started_at)
                                executed_tool_count += 1
                                timing["tool_execution_ms"] = round(
                                    float(timing["tool_execution_ms"]) + tool_elapsed_ms,
                                    2,
                                )
                                if tool_name == KNOWLEDGE_TOOL_NAME:
                                    timing["knowledge_lookup_ms"] = round(
                                        float(timing["knowledge_lookup_ms"]) + tool_elapsed_ms,
                                        2,
                                    )
                                timing["tool_calls"].append(
                                    {
                                        "name": tool_name,
                                        "hardware_task": hardware_intent.task if hardware_intent else None,
                                        "duration_ms": tool_elapsed_ms,
                                        "status": result.get("status", "unknown"),
                                        "fallback_used": True,
                                        "auto_stop_attempted": bool(
                                            (result.get("recovery") or {}).get(
                                                "auto_stop_attempted",
                                                False,
                                            )
                                        ),
                                    }
                                )
                                duration_line = (
                                    f"技能 {tool_name} 执行耗时 {tool_elapsed_ms:.0f} ms，"
                                    f"结果 {result.get('status', 'unknown')}。"
                                )
                                exec_logs.append(duration_line)
                                yield {"type": "tool_log", "line": duration_line}
                                for recovery_line in recovery_logs:
                                    exec_logs.append(recovery_line)
                                    yield {"type": "tool_log", "line": recovery_line}
                                if self._result_requests_visualization(tool_name, result):
                                    yield {"type": "visualization", "action": "open"}

                                final_reply = self._build_tool_execution_reply(
                                    tool_name=tool_name,
                                    args=inferred_args,
                                    result=result,
                                    fallback_used=True,
                                )
                                current_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": final_reply,
                                    }
                                )
                                timing["final_reply_source"] = "backend_tool_fallback"
                                if final_reply:
                                    for event in self._yield_text_as_deltas(final_reply):
                                        yield event
                                break

                            current_messages.append(
                                {
                                    "role": "system",
                                    "content": self._tool_retry_note(route),
                                }
                            )
                            continue

                        if route == "hardware" and last_tool_name and last_tool_result:
                            if current_messages and current_messages[-1] is assistant_dict:
                                current_messages.pop()
                            break

                        final_reply = self._plain_text_reply(assistant_dict.get("content") or "")
                        assistant_dict["content"] = final_reply
                        timing["final_reply_source"] = "tool_decision"
                        if final_reply:
                            for event in self._yield_text_as_deltas(final_reply):
                                yield event
                        break

                    for tool_call in tool_calls:
                        fn = tool_call.get("function") or {}
                        tool_name = fn.get("name") or ""
                        forced_by_intent = False
                        if (
                            route == "hardware"
                            and hardware_intent is not None
                            and tool_name != hardware_intent.tool_name
                        ):
                            correction_line = (
                                f"模型请求的技能 {tool_name or '(empty)'} 与物理实验意图不一致，"
                                f"已改为 {hardware_intent.tool_name}。"
                            )
                            exec_logs.append(correction_line)
                            yield {"type": "tool_log", "line": correction_line}
                            tool_name = hardware_intent.tool_name
                            forced_by_intent = True
                        tool_spec = self.tool_registry.get_tool(tool_name)
                        expected_category = (
                            HARDWARE_TOOL_CATEGORY
                            if route == "hardware"
                            else KNOWLEDGE_TOOL_CATEGORY
                        )
                        if tool_spec is None or tool_spec.category != expected_category:
                            result = {
                                "status": "error",
                                "error": f"模型请求了无效技能 {tool_name or '(empty)'}，本轮没有执行任何底层程序。",
                            }
                            last_tool_name = tool_name or "(empty)"
                            last_tool_args = {}
                            last_tool_result = dict(result)
                            timing["tool_calls"].append(
                                {
                                    "name": last_tool_name,
                                    "duration_ms": 0.0,
                                    "status": "error",
                                    "invalid_tool": True,
                                }
                            )
                            invalid_line = result["error"]
                            exec_logs.append(invalid_line)
                            yield {"type": "tool_log", "line": invalid_line}
                            current_messages.append(
                                {
                                    "tool_call_id": tool_call.get("id", ""),
                                    "role": "tool",
                                    "name": last_tool_name,
                                    "content": json.dumps(result, ensure_ascii=False),
                                }
                            )
                            continue
                        if forced_by_intent and hardware_intent is not None:
                            args = dict(hardware_intent.args)
                        else:
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except json.JSONDecodeError:
                                args = {}

                        args, repair_logs = self._repair_tool_args(
                            tool_name=tool_name,
                            args=args,
                            instruction=instruction,
                        )
                        for repair_line in repair_logs:
                            exec_logs.append(repair_line)
                            yield {"type": "tool_log", "line": repair_line}

                        line = f"触发技能 {tool_name}，参数: {args}"
                        exec_logs.append(line)
                        yield {"type": "tool_log", "line": line}

                        tool_started_at = time.perf_counter()
                        result, recovery_logs = self._execute_tool_with_recovery(tool_name, args)
                        last_tool_name = tool_name
                        last_tool_args = dict(args)
                        last_tool_result = dict(result)
                        tool_elapsed_ms = self._elapsed_ms(tool_started_at)
                        executed_tool_count += 1
                        timing["tool_execution_ms"] = round(
                            float(timing["tool_execution_ms"]) + tool_elapsed_ms,
                            2,
                        )
                        if tool_name == KNOWLEDGE_TOOL_NAME:
                            timing["knowledge_lookup_ms"] = round(
                                float(timing["knowledge_lookup_ms"]) + tool_elapsed_ms,
                                2,
                            )
                        timing["tool_calls"].append(
                            {
                                "name": tool_name,
                                "hardware_task": hardware_intent.task if hardware_intent else None,
                                "intent_corrected": forced_by_intent,
                                "duration_ms": tool_elapsed_ms,
                                "status": result.get("status", "unknown"),
                                "auto_stop_attempted": bool(
                                    (result.get("recovery") or {}).get("auto_stop_attempted", False)
                                ),
                            }
                        )
                        duration_line = (
                            f"技能 {tool_name} 执行耗时 {tool_elapsed_ms:.0f} ms，"
                            f"结果 {result.get('status', 'unknown')}。"
                        )
                        exec_logs.append(duration_line)
                        yield {"type": "tool_log", "line": duration_line}
                        for recovery_line in recovery_logs:
                            exec_logs.append(recovery_line)
                            yield {"type": "tool_log", "line": recovery_line}
                        if self._result_requests_visualization(tool_name, result):
                            yield {"type": "visualization", "action": "open"}

                        current_messages.append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": tool_name,
                                "content": self._format_tool_result_for_model(tool_name, result),
                            }
                        )

                if not final_reply and executed_tool_count == 0:
                    if timing["tool_calls"] and last_tool_name and last_tool_result:
                        final_reply = self._build_tool_execution_reply(
                            tool_name=last_tool_name,
                            args=last_tool_args,
                            result=last_tool_result,
                            fallback_used=False,
                        )
                        timing["final_reply_source"] = "tool_error_guard"
                    else:
                        raise RuntimeError("模型未按要求优先调用技能，系统已中止本轮请求。")

                if (
                    not final_reply
                    and route == "hardware"
                    and executed_tool_count > 0
                    and last_tool_name
                    and last_tool_result
                ):
                    final_reply = self._build_tool_execution_reply(
                        tool_name=last_tool_name,
                        args=last_tool_args,
                        result=last_tool_result,
                        fallback_used=False,
                    )
                    timing["final_reply_source"] = "tool_result_summary"
                    if final_reply:
                        for event in self._yield_text_as_deltas(final_reply):
                            yield event

                if not final_reply:
                    reply_started_at = time.perf_counter()
                    for event in self._iter_stream_completion(
                        model_name=active_model_name,
                        messages=current_messages,
                        temperature=active_temperature,
                        tools=None,
                    ):
                        if event.get("type") == "delta" and timing["first_token_ms"] is None:
                            timing["first_token_ms"] = self._elapsed_ms(reply_started_at)
                        if event.get("type") == "_assistant":
                            final_reply = event["message"].get("content") or ""
                            continue
                        yield event
                    timing["final_reply_ms"] = self._elapsed_ms(reply_started_at)
                    timing["final_reply_source"] = "stream_completion"
            else:
                current_messages = self._build_messages(
                    history=history,
                    instruction=instruction,
                    route=route,
                )
                reply_started_at = time.perf_counter()
                for event in self._iter_stream_completion(
                    model_name=active_model_name,
                    messages=current_messages,
                    temperature=active_temperature,
                    tools=None,
                ):
                    if event.get("type") == "delta" and timing["first_token_ms"] is None:
                        timing["first_token_ms"] = self._elapsed_ms(reply_started_at)
                    if event.get("type") == "_assistant":
                        final_reply = event["message"].get("content") or ""
                        continue
                    yield event
                    timing["final_reply_ms"] = self._elapsed_ms(reply_started_at)
                    timing["final_reply_source"] = "stream_completion"

            if (
                route in {"knowledge", "hardware"}
                and timing["tool_calls"]
                and not any(call.get("status") == "success" for call in timing["tool_calls"])
                and last_tool_name
                and last_tool_result
            ):
                final_reply = self._build_tool_execution_reply(
                    tool_name=last_tool_name,
                    args=last_tool_args,
                    result=last_tool_result,
                    fallback_used=bool(timing["tool_calls"][-1].get("fallback_used", False)),
                )
                timing["final_reply_source"] = "tool_error_guard"

            final_reply = self._plain_text_reply(final_reply)
            updated_history = self._extract_history(current_messages, final_reply)
            self.memory.replace(session_id, updated_history)
            timing["tool_count"] = len(timing["tool_calls"])
            timing["total_ms"] = self._elapsed_ms(request_started_at)

            yield {
                "type": "done",
                "status": "success",
                "reply": final_reply,
                "hardware_logs": exec_logs,
                "updated_history": updated_history,
                "active_mode": selected_mode,
                "active_model": active_model_name,
                "active_temperature": active_temperature,
                "active_route": route,
                "timing": timing,
            }
        except Exception as exc:  # pragma: no cover - surfaced to client
            yield {"type": "error", "message": str(exc)}

    def _assistant_message_to_dict(self, message: Any) -> dict[str, Any]:
        tool_calls = []
        for tool_call in getattr(message, "tool_calls", []) or []:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": self._plain_text_reply(message.content or ""),
        }
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return payload

    def _extract_history(
        self,
        messages: list[dict[str, Any]],
        final_reply: str,
    ) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and not item.get("tool_calls"):
                cleaned.append({"role": role, "content": content})

        if not cleaned or cleaned[-1]["role"] != "assistant" or cleaned[-1]["content"] != final_reply:
            cleaned.append({"role": "assistant", "content": final_reply})
        return cleaned
