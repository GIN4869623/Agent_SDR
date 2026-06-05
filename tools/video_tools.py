from __future__ import annotations

from hardware.video_stream_controller import VideoStreamController
from tools import ToolInput, ToolSpec


class StartVideoStreamRequest(ToolInput):
    pass


class StopVideoStreamRequest(ToolInput):
    pass


class VideoStreamStatusRequest(ToolInput):
    pass


def build_video_tools(controller: VideoStreamController) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="start_video_stream",
            description=(
                "启动基于 GNU Radio + USRP 的 BPSK 视频流无线传输。"
                "会自动运行 /usr/bin/python3 -u bpsk_video_stream.py，"
                "唤起 GNU Radio Qt 窗口和 VLC 播放器显示接收画面。"
                "需要 GNU Radio 环境和图形显示器。"
            ),
            schema_model=StartVideoStreamRequest,
            handler=lambda payload: controller.start(),
            category="hardware",
        ),
        ToolSpec(
            name="stop_video_stream",
            description=(
                "停止当前运行的 GNU Radio 视频流后台进程，并清理 VLC 播放器。"
            ),
            schema_model=StopVideoStreamRequest,
            handler=lambda payload: controller.stop(),
            category="hardware",
        ),
        ToolSpec(
            name="video_stream_status",
            description="查询当前视频流任务是否在运行。",
            schema_model=VideoStreamStatusRequest,
            handler=lambda payload: controller.get_state(),
            category="hardware",
        ),
    ]
