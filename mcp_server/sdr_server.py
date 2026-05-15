"""MCP server exposing SDR hardware tools, knowledge base, and hardware status
as standard MCP tools, resources, and prompts.

Mount on FastAPI via:
    from mcp_server.sdr_server import create_sdr_mcp_server
    app.mount("/mcp", create_sdr_mcp_server(tool_registry, knowledge_base, controller))
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from tools import ToolRegistry


def create_sdr_mcp_server(
    tool_registry: ToolRegistry,
    knowledge_base: LocalKnowledgeBase,
    controller: SDRController,
) -> Any:
    """Build and return an ASGI app (FastMCP.sse_app) for FastAPI mounting."""

    mcp = FastMCP(
        name="SDR Control Platform",
        instructions=(
            "SDR Control Platform — manage USRP X300 software-defined radio hardware. "
            "Use tools to scan spectrum, transmit/receive text with FSK/BPSK/QPSK/16-QAM, "
            "generate and visualize tones, and query SDR knowledge base. "
            "Use resources to check hardware status and visualization data. "
            "All frequencies default to 2.4 GHz center, 100 kHz tone."
        ),
    )

    # ── Tools ──────────────────────────────────────────────────────────

    @mcp.tool(
        name="perform_physical_scan",
        description="扫描指定频点附近的底噪和信号功率，返回各频点的接收功率 (dB)。",
    )
    async def perform_physical_scan(
        center_freq_hz: float = 2_400_000_000.0,
        bandwidth_hz: float = 1_000_000.0,
        duration_s: float = 0.2,
        chan: int = 0,
    ) -> str:
        result = tool_registry.execute("perform_physical_scan", {
            "center_freq_hz": center_freq_hz,
            "bandwidth_hz": bandwidth_hz,
            "duration_s": duration_s,
            "chan": chan,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="tone_loopback_visualize",
        description=(
            "发射正弦波/Tone/单音信号，启动持续后台发送并通过实时波形与频谱窗口观察。"
            "默认中心频率 2.4 GHz，基带 Tone 100 kHz。"
        ),
    )
    async def tone_loopback_visualize(
        center_freq_hz: float = 2_400_000_000.0,
        tone_freq_hz: float = 100_000.0,
        samp_rate: float = 1_000_000.0,
    ) -> str:
        result = tool_registry.execute("tone_loopback_visualize", {
            "center_freq_hz": center_freq_hz,
            "tone_freq_hz": tone_freq_hz,
            "samp_rate": samp_rate,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="stop_hardware_task",
        description="停止 USRP 当前后台任务（Tone 回环或长期运行的操作），释放硬件资源。",
    )
    async def stop_hardware_task() -> str:
        result = tool_registry.execute("stop_hardware_task", {})
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="text_fsk_send_and_receive",
        description=(
            "使用指定固定调制方式发送文本并通过 USRP 接收解调。"
            "默认调制方式为 2-FSK；用户可指定 BPSK、QPSK 或 16-QAM。"
        ),
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
        result = tool_registry.execute("text_fsk_send_and_receive", {
            "text": text,
            "modulation_scheme": modulation_scheme,
            "center_freq_hz": center_freq_hz,
            "samp_rate": samp_rate,
            "sym_rate": sym_rate,
            "f_dev": f_dev,
            "tx_gain": tx_gain,
            "rx_gain": rx_gain,
            "amp": amp,
            "packet_repetitions": packet_repetitions,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="adaptive_modulation_transmit",
        description=(
            "先探测信道 SNR，根据信道质量自适应选择 BPSK/QPSK/16-QAM，"
            "发送文本并接收解调。仅在用户明确要求自适应或根据信道质量选择调制时使用。"
        ),
    )
    async def adaptive_modulation_transmit(
        text: str,
        center_freq_hz: float = 2_400_000_000.0,
        probe_text: str = "channel_probe",
    ) -> str:
        result = tool_registry.execute("adaptive_modulation_transmit", {
            "text": text,
            "center_freq_hz": center_freq_hz,
            "probe_text": probe_text,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="auto_optimal_transmit",
        description=(
            "自动寻找干扰最小的干净频点，在最优信道上发送文本和/或发射正弦波。"
            "支持认知选频、Tone 可视化和自适应调制。"
        ),
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
        result = tool_registry.execute("auto_optimal_transmit", {
            "text": text,
            "modulation_scheme": modulation_scheme,
            "center_freq_hz": center_freq_hz,
            "search_span_hz": search_span_hz,
            "enable_tone": enable_tone,
            "tone_freq_hz": tone_freq_hz,
            "use_adaptive_modulation": use_adaptive_modulation,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="search_sdr_knowledge",
        description="从云端知识库检索 SDR 理论、设备规格、实验文档和项目资料。",
    )
    async def search_sdr_knowledge(
        query: str,
        top_k: int = 3,
    ) -> str:
        result = tool_registry.execute("search_sdr_knowledge", {
            "query": query,
            "top_k": top_k,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ── Resources ──────────────────────────────────────────────────────

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
        # Convert numpy arrays to lists for JSON serialisation
        for key in ("iq_inphase", "iq_quadrature", "fft_freq_khz", "fft_magnitude_db"):
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

    # ── Prompts ─────────────────────────────────────────────────────────

    @mcp.prompt(
        name="sdr-scan",
        description="频谱感知场景提示词 — 扫描指定频段寻找干净信道。",
    )
    async def prompt_scan() -> str:
        return "请帮我扫描 2.4 GHz 附近的底噪，找出接收功率最低、干扰最小的干净频点。"

    @mcp.prompt(
        name="sdr-transmit",
        description="文本无线收发场景提示词 — 用指定调制方式发送文本并验证解码。",
    )
    async def prompt_transmit(modulation: str = "2-FSK", text: str = "Hello") -> str:
        return f"请用 {modulation} 调制发送文本「{text}」并通过 USRP 接收解调，验证解码是否正确。"

    @mcp.prompt(
        name="sdr-visualize",
        description="信号可视化场景提示词 — 发射正弦波并显示实时波形频谱。",
    )
    async def prompt_visualize(freq_khz: str = "100") -> str:
        return (
            f"请发射一个 {freq_khz} kHz 的正弦波/Tone 信号，"
            f"中心频率 2.4 GHz，并打开可视化窗口显示实时 IQ 波形和 FFT 频谱。"
        )

    @mcp.prompt(
        name="sdr-diagnose",
        description="硬件结果诊断场景提示词 — 分析 SNR、解码状态并给出参数建议。",
    )
    async def prompt_diagnose() -> str:
        return (
            "请根据刚才的硬件执行结果，分析信道质量（SNR、接收功率），"
            "判断当前调制方式是否合适，如果不合适请给出具体的参数调整建议。"
        )

    # ── Return ASGI app ─────────────────────────────────────────────────
    return mcp.sse_app()
