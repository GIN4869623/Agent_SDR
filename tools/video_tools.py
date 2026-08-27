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
    def video_status(_request: VideoStreamStatusRequest) -> dict:
        return {"status": "success", **controller.get_state()}

    return [
        ToolSpec(
            name="start_video_stream",
            description=(
                "启动基于 GNU Radio + USRP 的 BPSK 视频流无线传输。"
                "需要 GNU Radio 环境和图形显示器。"
            ),
            schema_model=StartVideoStreamRequest,
            handler=lambda _request: controller.start(),
            category="hardware",
            risk="transmit",
            retry_safe=False,
        ),
        ToolSpec(
            name="stop_video_stream",
            description="停止 GNU Radio 视频流后台进程，并清理 VLC 播放器。",
            schema_model=StopVideoStreamRequest,
            handler=lambda _request: controller.stop(),
            category="hardware",
            risk="control",
            retry_safe=True,
        ),
        ToolSpec(
            name="video_stream_status",
            description="查询当前视频流任务是否在运行。",
            schema_model=VideoStreamStatusRequest,
            handler=video_status,
            category="hardware",
            risk="read_only",
            retry_safe=True,
        ),
    ]
