from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from hardware.sdr_controller import (
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_TONE_FREQ_HZ,
    DEFAULT_TONE_SAMPLE_RATE_HZ,
    SDRController,
)
from tools import ToolInput, ToolSpec
from tools.policy import (
    MAX_SCAN_CAPTURE_SAMPLES,
    RF_CENTER_MAX_HZ,
    RF_CENTER_MIN_HZ,
    SAMPLE_RATE_MAX_HZ,
    SAMPLE_RATE_MIN_HZ,
    SCAN_BANDWIDTH_MAX_HZ,
    SCAN_BANDWIDTH_MIN_HZ,
)


class ScanRequest(ToolInput):
    center_freq_hz: float = Field(
        default=DEFAULT_CENTER_FREQ_HZ,
        ge=RF_CENTER_MIN_HZ,
        le=RF_CENTER_MAX_HZ,
        description="中心频率，单位 Hz；当前 USRP-2943R 允许 1.2-6.0 GHz。",
    )
    bandwidth_hz: float = Field(
        default=1e6,
        ge=SCAN_BANDWIDTH_MIN_HZ,
        le=SCAN_BANDWIDTH_MAX_HZ,
        description="扫描带宽，单位 Hz；允许 100 kHz-40 MHz。",
    )
    duration_s: float = Field(
        default=0.2,
        ge=0.02,
        le=10.0,
        description="接收采样持续时间，单位秒。",
    )
    chan: Literal[0, 1] = Field(default=0, description="接收通道，只允许 0 或 1。")

    @model_validator(mode="after")
    def validate_capture_size(self) -> "ScanRequest":
        if self.bandwidth_hz * self.duration_s > MAX_SCAN_CAPTURE_SAMPLES:
            raise ValueError(
                f"bandwidth_hz * duration_s 不能超过 {MAX_SCAN_CAPTURE_SAMPLES} 个采样点"
            )
        return self


class ToneLoopbackRequest(ToolInput):
    center_freq_hz: float = Field(
        default=DEFAULT_CENTER_FREQ_HZ,
        ge=RF_CENTER_MIN_HZ,
        le=RF_CENTER_MAX_HZ,
        description="射频中心频率，单位 Hz；当前允许 1.2-6.0 GHz。",
    )
    tone_freq_hz: float = Field(
        default=DEFAULT_TONE_FREQ_HZ,
        gt=0,
        description="基带 Tone/正弦波频率，单位 Hz。",
    )
    samp_rate: float = Field(
        default=DEFAULT_TONE_SAMPLE_RATE_HZ,
        ge=SAMPLE_RATE_MIN_HZ,
        le=SAMPLE_RATE_MAX_HZ,
        description="采样率，单位 Hz；允许 100 kHz-40 MHz。",
    )

    @model_validator(mode="after")
    def validate_nyquist(self) -> "ToneLoopbackRequest":
        if self.tone_freq_hz >= self.samp_rate / 2:
            raise ValueError("tone_freq_hz 必须小于 samp_rate/2，避免混叠")
        return self


class StopHardwareTaskRequest(ToolInput):
    pass


def build_physical_tools(controller: SDRController) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="perform_physical_scan",
            description="扫描指定频点附近的底噪和信号功率，返回各频点的接收功率 (dB)。",
            schema_model=ScanRequest,
            handler=lambda request: controller.perform_physical_scan(**request.model_dump()),
            category="hardware",
            risk="read_only",
            retry_safe=True,
        ),
        ToolSpec(
            name="tone_loopback_visualize",
            description=(
                "发射正弦波/Tone/单音信号，启动持续后台发送并通过实时波形与频谱窗口观察。"
                "默认中心频率 2.4 GHz，基带 Tone 100 kHz。"
            ),
            schema_model=ToneLoopbackRequest,
            handler=lambda request: controller.tone_loopback_visualize(**request.model_dump()),
            category="hardware",
            risk="transmit",
            retry_safe=False,
        ),
        ToolSpec(
            name="stop_hardware_task",
            description="停止 USRP 当前后台任务（Tone 回环或长期运行的操作），释放硬件资源。",
            schema_model=StopHardwareTaskRequest,
            handler=lambda _request: controller.stop_hardware_task(),
            category="hardware",
            risk="control",
            retry_safe=True,
        ),
    ]
