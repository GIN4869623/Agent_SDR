"""Compatibility entry point for the LangGraph-backed agent.

Console event helpers stay in this module because the HTTP server and existing
integrations import them from here. The orchestration lives in
``src.agent.graph``.
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

CONSOLE_URL: str = os.getenv("CONSOLE_EVENT_URL", "http://127.0.0.1:3000")
CONSOLE_TOKEN: str = os.getenv(
    "CONSOLE_EVENT_TOKEN", "dev-local-token-change-in-production"
)


def _agent_id_for_tool(tool_name: str) -> str:
    """Map a canonical tool to its console agent node."""
    return "rag" if tool_name == "search_sdr_knowledge" else "usrp"


async def _post_console_event(
    agent_id: str,
    run_id: str,
    session_id: str,
    phase: str,
    task: str,
    tool_name: str,
    detail: str = "",
) -> None:
    """Send one best-effort event to the multi-agent console."""
    try:
        payload = {
            "type": "agent.activity",
            "agentId": agent_id,
            "runId": run_id,
            "sessionId": session_id,
            "phase": phase,
            "task": task,
            "toolName": tool_name,
            "detail": detail,
        }
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{CONSOLE_URL}/api/integrations/agent-events",
                json=payload,
                headers={
                    "Authorization": f"Bearer {CONSOLE_TOKEN}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning(
                        "Console event rejected (HTTP %s): %s",
                        response.status,
                        body[:200],
                    )
    except asyncio.TimeoutError:
        logger.warning("Console event timeout (%s)", CONSOLE_URL)
    except aiohttp.ClientConnectorError as exc:
        logger.warning("Console unreachable (%s): %s", CONSOLE_URL, exc)
    except aiohttp.ClientError as exc:
        logger.warning("Console event network error: %s", exc)
    except Exception:
        logger.warning("Console event unexpected error", exc_info=True)


def report_to_console(
    agent_id: str,
    run_id: str,
    session_id: str,
    phase: str,
    task: str,
    tool_name: str,
    detail: str = "",
) -> None:
    """Schedule console reporting without blocking the SSE stream."""
    try:
        asyncio.create_task(
            _post_console_event(
                agent_id=agent_id,
                run_id=run_id,
                session_id=session_id,
                phase=phase,
                task=task,
                tool_name=tool_name,
                detail=detail,
            )
        )
    except RuntimeError:
        logger.debug("Console event skipped: no running event loop")


# Imported last so graph.py can call these patchable helpers without creating
# an initialization cycle.
from src.agent.graph import AgentLoop  # noqa: E402

__all__ = ["AgentLoop", "_agent_id_for_tool", "_post_console_event", "report_to_console"]
