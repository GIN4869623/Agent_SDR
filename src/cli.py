"""MCP stdio entry point — for Claude Code and other external agents.

Usage:
    python -m src.cli

This starts an MCP server over stdio.  Claude Code (or any MCP-compatible
agent) can connect directly by adding to its MCP config:

    {
      "mcpServers": {
        "sdr": {
          "command": "python",
          "args": ["-m", "src.cli"],
          "cwd": "/home/wenchaoxia/Desktop/Agent_SDR"
        }
      }
    }

When launched this way, only the MCP server is active — no web UI, no
agent loop.  The connecting agent provides its own intelligence and uses
the SDR tools/resources/prompts exposed here.
"""

from __future__ import annotations

import asyncio
import sys

from src.config import settings
from src.mcp import create_sdr_mcp

# Bootstrap hardware + knowledge (same as server.py)
from core.memory import ConversationMemory
from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from hardware.video_stream_controller import VideoStreamController


async def main() -> None:
    # Init domain layer
    controller = SDRController(settings.usrp_ip)
    knowledge_base = LocalKnowledgeBase(
        app_id=settings.bailian_app_id,
        api_key=settings.bailian_api_key,
    )
    video_controller = VideoStreamController(
        release_cb=controller.disconnect_usrp,
        reconnect_cb=controller.reconnect_usrp,
    )

    # Build MCP server
    mcp_server, _tool_registry = create_sdr_mcp(
        controller=controller,
        knowledge_base=knowledge_base,
        video_controller=video_controller,
    )

    # Run over stdio — FastMCP 1.28+ unified API
    await mcp_server.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
