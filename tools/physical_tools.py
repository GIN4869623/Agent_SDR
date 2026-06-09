from __future__ import annotations

from pydantic import Field

from hardware.sdr_controller import (
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_TONE_FREQ_HZ,
    DEFAULT_TONE_SAMPLE_RATE_HZ,
    SDRController,
)
from tools import ToolInput, ToolSpec


class ScanRequest(ToolInput):
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, description="Center frequency in Hz.")
    bandwidth_hz: float = Field(default=1e6, description="Scan bandwidth in Hz.")
    duration_s: float = Field(default=0.2, ge=0.02, le=10.0, description="接收采样持续时间，单位秒。")
    chan: int = Field(default=0, description="Receive channel index.")


class ToneLoopbackRequest(ToolInput):
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, description="射频中心频率，单位 Hz；用户未指定时必须保持默认 2.4 GHz。")
    tone_freq_hz: float = Field(default=DEFAULT_TONE_FREQ_HZ, description="基带 Tone/正弦波频率，单位 Hz；用户未指定时必须保持默认 100 kHz。")
    samp_rate: float = Field(default=DEFAULT_TONE_SAMPLE_RATE_HZ, description="采样率，单位 Hz。")


class StopHardwareTaskRequest(ToolInput):
    pass


def build_physical_tools(controller: SDRController) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="perform_physical_scan",
            description=(
                "扫噪/测底噪工具，用于测量目标频点附近的接收功率和噪声水平。"
            ),
            schema_model=ScanRequest,
            handler=lambda payload: controller.perform_physical_scan(**payload.model_dump()),
            category="hardware",
        ),
        ToolSpec(
            name="tone_loopback_visualize",
            description=(
                "启动连续 Tone/正弦波回环任务，并把实时波形和 FFT 数据发布到可视化窗口。"
                "默认参数固定为射频中心频率 2.4 GHz、基带正弦波 100 kHz；"
                "只有用户明确给出对应参数时才修改。"
            ),
            schema_model=ToneLoopbackRequest,
            handler=lambda payload: controller.tone_loopback_visualize(**payload.model_dump()),
            category="hardware",
        ),
        ToolSpec(
            name="stop_hardware_task",
            description="Stop the current background hardware task and release the USRP device.",
            schema_model=StopHardwareTaskRequest,
            handler=lambda payload: controller.stop_hardware_task(),
            category="hardware",
        ),
    ]
