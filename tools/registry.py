"""Build the ToolRegistry shared by the local Agent and MCP transport."""

from __future__ import annotations

from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from hardware.video_stream_controller import VideoStreamController
from tools import ToolRegistry, ToolSpec
from tools.knowledge_tools import build_knowledge_tools
from tools.physical_tools import build_physical_tools
from tools.transmit_tools import build_transmit_tools
from tools.video_tools import build_video_tools


def create_tool_registry(
    controller: SDRController,
    knowledge_base: LocalKnowledgeBase,
    video_controller: VideoStreamController | None = None,
) -> ToolRegistry:
    """Create protocol-independent tool definitions exactly once."""

    specs: list[ToolSpec] = []
    specs.extend(build_physical_tools(controller))
    specs.extend(build_transmit_tools(controller))
    specs.extend(build_knowledge_tools(knowledge_base))
    if video_controller is not None:
        specs.extend(build_video_tools(video_controller))
    return ToolRegistry(specs, hardware_busy=controller.has_background_task)
