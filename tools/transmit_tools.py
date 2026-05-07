from __future__ import annotations

from pydantic import Field

from hardware.sdr_controller import (
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_TEXT_PACKET_REPETITIONS,
    DEFAULT_TONE_FREQ_HZ,
    SDRController,
)
from tools import ToolInput, ToolSpec


class FskSendRequest(ToolInput):
    text: str = Field(min_length=1, description="必须从用户指令中提取出的要发送文本内容。")
    modulation_scheme: str = Field(
        default="2-FSK",
        description="固定文本收发调制方式。用户未指定时使用 2-FSK；明确指定时可用 BPSK、QPSK 或 16-QAM。",
    )
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, description="中心频率，单位 Hz。")
    samp_rate: float = Field(default=1e6, description="采样率，单位 Hz。")
    sym_rate: float = Field(default=10e3, description="符号率，单位 Baud。")
    f_dev: float = Field(default=25e3, description="频偏，单位 Hz。")
    tx_gain: float = Field(default=20.0, description="发射增益，单位 dB。")
    rx_gain: float = Field(default=20.0, description="接收增益，单位 dB。")
    amp: float = Field(default=0.6, description="发送信号幅度，范围建议 0 到 1。")
    packet_repetitions: int = Field(
        default=DEFAULT_TEXT_PACKET_REPETITIONS,
        ge=1,
        le=8,
        description="同一个文本包重复发送次数，用于提高短包解码成功率。",
    )


class AdaptiveTransmitRequest(ToolInput):
    text: str = Field(min_length=1, description="正式发送的文本内容，必须从用户指令中提取。")
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, description="中心频率，单位 Hz。")
    probe_text: str = Field(default="channel_probe", description="用于 BPSK 探测并解调校验的链路探测文本。")


class AutoOptimalTransmitRequest(ToolInput):
    text: str = Field(default="", description="若需要发送文本，填写文本内容；否则留空。")
    modulation_scheme: str = Field(default="2-FSK", description="非自适应文本发送时使用的固定调制方式，默认 2-FSK。")
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, description="目标工作频率，单位 Hz。")
    search_span_hz: float = Field(default=30e6, description="频点搜索跨度，单位 Hz。")
    enable_tone: bool = Field(default=False, description="是否在最优频点启动 Tone/正弦波可视化。")
    tone_freq_hz: float = Field(default=DEFAULT_TONE_FREQ_HZ, description="Tone 基带频率，单位 Hz；用户未指定时保持默认 100 kHz。")
    use_adaptive_modulation: bool = Field(
        default=False,
        description="是否按 SNR 自动切换 BPSK / QPSK / 16-QAM。",
    )


def build_transmit_tools(controller: SDRController) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="text_fsk_send_and_receive",
            description=(
                "执行固定调制的普通文本无线收发，必须把用户要发送的内容填入 text。"
                "用户未指定调制方式时默认使用 2-FSK；用户明确说用 BPSK、QPSK 或 16-QAM 发送文本时，也必须使用本工具并设置 modulation_scheme。"
                "如果用户说根据信道质量、链路质量、SNR、自适应、自动选择或调整调制方式，不要使用本工具。"
            ),
            schema_model=FskSendRequest,
            handler=lambda payload: controller.text_fsk_send_and_receive(**payload.model_dump()),
            category="hardware",
        ),
        ToolSpec(
            name="adaptive_modulation_transmit",
            description=(
                "先探测链路质量/SNR，再自动选择 BPSK、QPSK 或 16-QAM 完成文本发送、接收和对应调制方式解调。"
                "用户说根据信道质量、链路质量、SNR、自适应、自动选择调制方式或调整调制方式时必须使用本工具。"
            ),
            schema_model=AdaptiveTransmitRequest,
            handler=lambda payload: controller.adaptive_modulation_transmit(**payload.model_dump()),
            category="hardware",
        ),
        ToolSpec(
            name="auto_optimal_transmit",
            description=(
                "认知选频工具：先搜索更干净的频点，再执行文本发送或 Tone/正弦波可视化；"
                "非自适应文本发送默认使用 2-FSK，用户指定 BPSK、QPSK 或 16-QAM 时设置 modulation_scheme；"
                "用户要求找干净信道并显示可视化框图/波形/频谱时设置 enable_tone=true。"
            ),
            schema_model=AutoOptimalTransmitRequest,
            handler=lambda payload: controller.auto_optimal_transmit(**payload.model_dump()),
            category="hardware",
        ),
    ]
