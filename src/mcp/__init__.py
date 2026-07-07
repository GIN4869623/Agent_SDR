"""MCP Server — single source of truth for all SDR tools, resources, and prompts.

This module replaces the old tools/ directory and mcp_server/sdr_server.py.
Tools are defined once here; the Agent Loop uses them via `get_tools()` /
`call_tool()`, and external agents consume them via SSE or stdio transport.

Architecture:
    MCPToolRegistry  ←  programmatic access (Agent Loop)
    FastMCP          ←  protocol transport (SSE / stdio for external agents)
    Both backed by the same handlers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from hardware.video_stream_controller import VideoStreamController


# ═══════════════════════════════════════════════════════════════════════════
# Tool Definition
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolDef:
    """Lightweight tool definition — one source of truth."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable  # async (... ) -> str
    category: str = "general"


class MCPToolRegistry:
    """Holds all tool definitions.  Used by both Agent Loop and MCP transport."""

    def __init__(self, controller: SDRController | None = None) -> None:
        self._tools: list[ToolDef] = []
        self._by_name: dict[str, ToolDef] = {}
        self._controller = controller

    def register(self, tool: ToolDef) -> None:
        self._tools.append(tool)
        self._by_name[tool.name] = tool

    def get_all(self) -> list[ToolDef]:
        return list(self._tools)

    def get_by_category(self, category: str) -> list[ToolDef]:
        return [t for t in self._tools if t.category == category]

    def get(self, name: str) -> ToolDef | None:
        return self._by_name.get(name)

    async def call(self, name: str, args: dict) -> str:
        """Execute a tool by name. Returns JSON string."""
        tool = self._by_name.get(name)
        if tool is None:
            return json.dumps({"status": "error", "error": f"未知技能: {name}"})
        try:
            return await tool.handler(**args)
        except Exception as exc:
            return json.dumps(
                {"status": "error", "error": f"技能 {name} 执行失败: {exc}"}
            )

    def is_hardware_busy(self) -> bool:
        """Check if the USRP has any active background task.

        Returns False when no controller is available (e.g. dry-run).
        """
        if self._controller is None:
            return False
        return self._controller.has_background_task()

    def as_openai_tools(self, category: str | None = None) -> list[dict]:
        """Convert tools to OpenAI function-calling format."""
        tools = self.get_by_category(category) if category else self._tools
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


def create_sdr_mcp(
    controller: SDRController,
    knowledge_base: LocalKnowledgeBase,
    video_controller: VideoStreamController | None = None,
) -> tuple[FastMCP, MCPToolRegistry]:
    """Build the FastMCP server + programmatic registry.

    Returns:
        (mcp_server, tool_registry)
        - mcp_server: FastMCP instance for SSE/stdio transport
        - tool_registry: MCPToolRegistry for Agent Loop direct access
    """

    mcp = FastMCP(
        name="Agent SDR Platform",
        instructions=(
            "Agent SDR Platform — manage USRP X300 software-defined radio hardware. "
            "Use tools to scan spectrum, transmit/receive text with FSK/BPSK/QPSK/16-QAM, "
            "generate and visualize tones, stream video via GNU Radio, and query the "
            "SDR knowledge base.  Default center frequency: 2.4 GHz.  "
            "Default tone frequency: 100 kHz."
        ),
    )
    registry = MCPToolRegistry(controller=controller)

    # ── Helper: register tool in both FastMCP and MCPToolRegistry ──

    def _register(
        name: str,
        description: str,
        parameters: dict,
        category: str = "hardware",
    ):
        def decorator(fn):
            # Register with FastMCP (for external agents via SSE/stdio)
            mcp_tool = mcp.tool(name=name, description=description)(fn)
            # Register with ToolRegistry (for internal Agent Loop)
            registry.register(
                ToolDef(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=fn,
                    category=category,
                )
            )
            return mcp_tool

        return decorator

    # ─────────────────────────────────────────────────────────────────────
    # Hardware Tools
    # ─────────────────────────────────────────────────────────────────────

    @_register(
        name="perform_physical_scan",
        description="扫描指定频点附近的底噪和信号功率，返回各频点的接收功率 (dB)。",
        parameters={
            "type": "object",
            "properties": {
                "center_freq_hz": {
                    "type": "number",
                    "description": "中心频率，单位 Hz，默认 2.4e9。",
                    "default": 2_400_000_000.0,
                },
                "bandwidth_hz": {
                    "type": "number",
                    "description": "扫描带宽，单位 Hz。",
                    "default": 1_000_000.0,
                },
                "duration_s": {
                    "type": "number",
                    "description": "接收采样持续时间，单位秒。",
                    "default": 0.2,
                },
                "chan": {
                    "type": "integer",
                    "description": "接收通道索引。",
                    "default": 0,
                },
            },
            "required": [],
        },
    )
    async def perform_physical_scan(
        center_freq_hz: float = 2_400_000_000.0,
        bandwidth_hz: float = 1_000_000.0,
        duration_s: float = 0.2,
        chan: int = 0,
    ) -> str:
        result = controller.perform_physical_scan(
            center_freq_hz=center_freq_hz,
            bandwidth_hz=bandwidth_hz,
            duration_s=duration_s,
            chan=chan,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_register(
        name="tone_loopback_visualize",
        description=(
            "发射正弦波/Tone/单音信号，启动持续后台发送并通过实时波形与频谱窗口观察。"
            "默认中心频率 2.4 GHz，基带 Tone 100 kHz。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "center_freq_hz": {
                    "type": "number",
                    "description": "射频中心频率 Hz，默认 2.4e9。",
                    "default": 2_400_000_000.0,
                },
                "tone_freq_hz": {
                    "type": "number",
                    "description": "基带 Tone 频率 Hz，默认 100e3。",
                    "default": 100_000.0,
                },
                "samp_rate": {
                    "type": "number",
                    "description": "采样率 Hz。",
                    "default": 1_000_000.0,
                },
            },
            "required": [],
        },
    )
    async def tone_loopback_visualize(
        center_freq_hz: float = 2_400_000_000.0,
        tone_freq_hz: float = 100_000.0,
        samp_rate: float = 1_000_000.0,
    ) -> str:
        result = controller.tone_loopback_visualize(
            center_freq_hz=center_freq_hz,
            tone_freq_hz=tone_freq_hz,
            samp_rate=samp_rate,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_register(
        name="stop_hardware_task",
        description="停止 USRP 当前后台任务（Tone 回环或长期运行的操作），释放硬件资源。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    async def stop_hardware_task() -> str:
        result = controller.stop_hardware_task()
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_register(
        name="text_fsk_send_and_receive",
        description=(
            "使用指定固定调制方式发送文本并通过 USRP 接收解调。"
            "默认 2-FSK；用户可指定 BPSK、QPSK 或 16-QAM。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要发送的文本内容，必须从用户指令中提取。",
                },
                "modulation_scheme": {
                    "type": "string",
                    "description": "调制方式：2-FSK, BPSK, QPSK, 16-QAM。",
                    "default": "2-FSK",
                },
                "center_freq_hz": {
                    "type": "number",
                    "description": "中心频率 Hz。",
                    "default": 2_400_000_000.0,
                },
                "samp_rate": {
                    "type": "number",
                    "description": "采样率 Hz。",
                    "default": 1_000_000.0,
                },
                "sym_rate": {
                    "type": "number",
                    "description": "符号率 Baud。",
                    "default": 10_000.0,
                },
                "f_dev": {"type": "number", "description": "频偏 Hz。", "default": 25_000.0},
                "tx_gain": {
                    "type": "number",
                    "description": "发射增益 dB。",
                    "default": 20.0,
                },
                "rx_gain": {
                    "type": "number",
                    "description": "接收增益 dB。",
                    "default": 20.0,
                },
                "amp": {
                    "type": "number",
                    "description": "发送信号幅度 0-1。",
                    "default": 0.6,
                },
                "packet_repetitions": {
                    "type": "integer",
                    "description": "包重复次数。",
                    "default": 3,
                },
            },
            "required": ["text"],
        },
    )
    async def text_fsk_send_and_receive(
        text: str,
        modulation_scheme: str = "2-FSK",
        center_freq_hz: float = 2_400_000_000.0,
        samp_rate: float = 1_000_000.0,
        sym_rate: float = 10_000.0,
        f_dev: float = 25_000.0,
        tx_gain: float = 20.0,
        rx_gain: float = 20.0,
        amp: float = 0.6,
        packet_repetitions: int = 3,
    ) -> str:
        result = controller.text_fsk_send_and_receive(
            text=text,
            modulation_scheme=modulation_scheme,
            center_freq_hz=center_freq_hz,
            samp_rate=samp_rate,
            sym_rate=sym_rate,
            f_dev=f_dev,
            tx_gain=tx_gain,
            rx_gain=rx_gain,
            amp=amp,
            packet_repetitions=packet_repetitions,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_register(
        name="adaptive_modulation_transmit",
        description=(
            "先探测信道 SNR，根据信道质量自适应选择 BPSK/QPSK/16-QAM，"
            "发送文本并接收解调。仅在用户明确要求自适应或根据信道质量选择调制时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "正式发送的文本内容。",
                },
                "center_freq_hz": {
                    "type": "number",
                    "description": "中心频率 Hz。",
                    "default": 2_400_000_000.0,
                },
                "probe_text": {
                    "type": "string",
                    "description": "链路探测文本。",
                    "default": "channel_probe",
                },
            },
            "required": ["text"],
        },
    )
    async def adaptive_modulation_transmit(
        text: str,
        center_freq_hz: float = 2_400_000_000.0,
        probe_text: str = "channel_probe",
    ) -> str:
        result = controller.adaptive_modulation_transmit(
            text=text,
            center_freq_hz=center_freq_hz,
            probe_text=probe_text,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_register(
        name="auto_optimal_transmit",
        description=(
            "自动寻找干扰最小的干净频点，在最优信道上发送文本和/或发射正弦波。"
            "支持认知选频、Tone 可视化和自适应调制。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要发送的文本，若只需可视化则留空。",
                    "default": "",
                },
                "modulation_scheme": {
                    "type": "string",
                    "description": "调制方式，默认 2-FSK。",
                    "default": "2-FSK",
                },
                "center_freq_hz": {
                    "type": "number",
                    "description": "目标工作频率 Hz。",
                    "default": 2_400_000_000.0,
                },
                "search_span_hz": {
                    "type": "number",
                    "description": "频点搜索跨度 Hz。",
                    "default": 30_000_000.0,
                },
                "enable_tone": {
                    "type": "boolean",
                    "description": "是否在最优频点启动 Tone 可视化。",
                    "default": False,
                },
                "tone_freq_hz": {
                    "type": "number",
                    "description": "Tone 基带频率 Hz。",
                    "default": 100_000.0,
                },
                "use_adaptive_modulation": {
                    "type": "boolean",
                    "description": "是否按 SNR 自动切换调制方式。",
                    "default": False,
                },
            },
            "required": [],
        },
    )
    async def auto_optimal_transmit(
        text: str = "",
        modulation_scheme: str = "2-FSK",
        center_freq_hz: float = 2_400_000_000.0,
        search_span_hz: float = 30_000_000.0,
        enable_tone: bool = False,
        tone_freq_hz: float = 100_000.0,
        use_adaptive_modulation: bool = False,
    ) -> str:
        result = controller.auto_optimal_transmit(
            text=text,
            modulation_scheme=modulation_scheme,
            center_freq_hz=center_freq_hz,
            search_span_hz=search_span_hz,
            enable_tone=enable_tone,
            tone_freq_hz=tone_freq_hz,
            use_adaptive_modulation=use_adaptive_modulation,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────────────────────────────
    # Knowledge Tool
    # ─────────────────────────────────────────────────────────────────────

    @_register(
        name="search_sdr_knowledge",
        description="从云端知识库检索 SDR 理论、设备规格、实验文档和项目资料。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的理论问题、术语或设备说明。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量。",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        category="knowledge",
    )
    async def search_sdr_knowledge(query: str, top_k: int = 3) -> str:
        result = knowledge_base.search(query, top_k=top_k)
        return json.dumps(
            {
                "status": "success",
                "message": f"已返回 {len(result)} 条知识检索结果。",
                "results": [
                    {"source": hit.source, "snippet": hit.snippet, "score": hit.score}
                    for hit in result
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Video Tools (optional)
    # ─────────────────────────────────────────────────────────────────────

    if video_controller is not None:

        @_register(
            name="start_video_stream",
            description=(
                "启动基于 GNU Radio + USRP 的 BPSK 视频流无线传输。"
                "需要 GNU Radio 环境和图形显示器。"
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )
        async def start_video_stream() -> str:
            result = video_controller.start()
            return json.dumps(result, ensure_ascii=False, indent=2)

        @_register(
            name="stop_video_stream",
            description="停止 GNU Radio 视频流后台进程，并清理 VLC 播放器。",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        async def stop_video_stream() -> str:
            result = video_controller.stop()
            return json.dumps(result, ensure_ascii=False, indent=2)

        @_register(
            name="video_stream_status",
            description="查询当前视频流任务是否在运行。",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        async def video_stream_status() -> str:
            state = video_controller.get_state()
            return json.dumps(state, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────────────────────────────
    # Resources
    # ─────────────────────────────────────────────────────────────────────

    @mcp.resource(
        uri="sdr://hardware/status",
        name="USRP Hardware Status",
        description="USRP 当前连接状态、中心频率、采样率、增益和调制方式。",
    )
    async def hardware_status() -> str:
        snapshot = controller.get_hardware_snapshot()
        return json.dumps(snapshot, ensure_ascii=False, indent=2)

    @mcp.resource(
        uri="sdr://hardware/visualization",
        name="Visualization Snapshot",
        description="当前可视化窗口的 IQ 波形和 FFT 频谱实时数据。",
    )
    async def visualization_snapshot() -> str:
        snap = controller.get_visualization_snapshot()
        for key in (
            "iq_inphase",
            "iq_quadrature",
            "fft_freq_khz",
            "fft_magnitude_db",
        ):
            if key in snap and hasattr(snap[key], "tolist"):
                snap[key] = snap[key].tolist()
        return json.dumps(snap, ensure_ascii=False, indent=2)

    @mcp.resource(
        uri="sdr://knowledge/{query}",
        name="SDR Knowledge Query",
        description="从百炼云端知识库检索 SDR 相关文档资料。",
    )
    async def knowledge_query(query: str) -> str:
        result = knowledge_base.query(query)
        return result or f"知识库未检索到与「{query}」匹配的内容。"

    # ─────────────────────────────────────────────────────────────────────
    # Prompts — auto-generated from skill .md files (single source of truth)
    # ─────────────────────────────────────────────────────────────────────

    import re as _re
    import yaml as _yaml
    from pathlib import Path as _Path

    _SKILLS_DIR = _Path(__file__).resolve().parent.parent.parent / "skill" / "skills"

    for _md_path in sorted(_SKILLS_DIR.glob("*.md")):
        _content = _md_path.read_text(encoding="utf-8")
        # Parse YAML frontmatter
        _fm_match = _re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", _content, _re.DOTALL)
        if not _fm_match:
            continue
        _frontmatter = _yaml.safe_load(_fm_match.group(1)) or {}
        _name = _frontmatter.get("name", _md_path.stem)
        _desc = _frontmatter.get("description", _name)
        _body = _fm_match.group(2).strip()
        # Strip leading markdown heading
        _body = _re.sub(r"^#\s+.*\n+", "", _body).strip()

        # Register prompt — closure capture is safe here because each iteration
        # creates a new scope via the decorator call pattern.
        def _make_prompt_fn(body: str = _body):
            async def _prompt_fn() -> str:
                return body
            return _prompt_fn

        mcp.prompt(name=_name, description=_desc)(_make_prompt_fn())

    # Meta-prompt: hardware result diagnosis (no single skill covers this)
    @mcp.prompt(
        name="sdr-diagnose",
        description="硬件结果诊断 — 分析 SNR、解码状态并给出参数建议。",
    )
    async def prompt_diagnose() -> str:
        return (
            "请根据刚才的硬件执行结果，分析信道质量（SNR、接收功率），"
            "判断当前调制方式是否合适，如果不合适请给出具体的参数调整建议。"
        )

    return mcp, registry
