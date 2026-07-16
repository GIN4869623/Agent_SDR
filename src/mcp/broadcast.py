"""MCP notification broadcast — push events to connected clients via SSE.

This module replaces the aiohttp HTTP POST approach.  Instead of opening a
separate HTTP connection to the console, events are pushed through the
existing MCP SSE transport that clients already use.

Architecture:
    server.py  ──monkey-patch──►  captures SseServerTransport
    loop.py    ──broadcast_event──►  writes JSON-RPC notifications
                                     to all connected SSE clients
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCNotification, JSONRPCMessage

logger = logging.getLogger(__name__)

# ── Global transport reference ──────────────────────────────────────────
# Set by server.py after MCP SSE transport is created via monkey-patch.

_transport: Any = None


def set_transport(transport: Any) -> None:
    """Store the SseServerTransport instance for later broadcasts."""
    global _transport
    _transport = transport


def broadcast_event(event: dict[str, Any]) -> bool:
    """Push an agent-activity event to all connected MCP clients.

    Returns True if at least one client received the event, False if
    no clients are connected or the transport is unavailable.

    Never raises — failures are logged and swallowed.
    """
    transport = _transport
    if transport is None:
        return False

    try:
        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/agent_activity",
            params=event,
        )
        msg = SessionMessage(message=JSONRPCMessage(notification))

        sent = False
        for sid, writer in list(transport._read_stream_writers.items()):
            try:
                writer.send_nowait(msg)
                sent = True
            except Exception:
                logger.debug("MCP broadcast: failed to write to session %s", sid)

        return sent
    except Exception:
        logger.debug("MCP broadcast failed", exc_info=True)
        return False
