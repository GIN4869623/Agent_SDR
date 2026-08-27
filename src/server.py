"""FastAPI + SSE entry point — serves the web UI and MCP SSE endpoint.

Usage:
    uvicorn src.server:app --host 127.0.0.1 --port 8000

Architecture:
    - /api/chat        — Agent Loop (non-streaming)
    - /api/chat/stream — Agent Loop (SSE streaming)
    - /api/health      — Health check
    - /api/hardware_status — Hardware snapshot
    - /mcp             — MCP SSE endpoint (for external agents)
    - /                — Static web UI
"""

from __future__ import annotations

import json
import socket
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.loop import _agent_id_for_tool, _post_console_event
from src.agent.trace import RunTraceStore
from src.config import settings
from src.mcp import create_sdr_mcp

# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

from core.memory import ConversationMemory
from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from hardware.video_stream_controller import VideoStreamController
from tools.registry import create_tool_registry

memory = ConversationMemory(max_messages=settings.session_history_limit)
knowledge_base = LocalKnowledgeBase(
    api_key=settings.bailian_api_key,
    embedding_model=settings.bailian_embedding_model,
    knowledge_dir=settings.knowledge_dir,
    index_dir=settings.knowledge_dir / ".index",
)
controller = SDRController(settings.usrp_ip)
video_controller = VideoStreamController(
    release_cb=controller.disconnect_usrp,
    reconnect_cb=controller.reconnect_usrp,
)

# Build the protocol-independent registry first. Both the local Agent and the
# external MCP transport delegate to this exact execution boundary.
tool_registry = create_tool_registry(
    controller=controller,
    knowledge_base=knowledge_base,
    video_controller=video_controller,
)
mcp_server = create_sdr_mcp(
    tool_registry=tool_registry,
    controller=controller,
    knowledge_base=knowledge_base,
)

# Create skill registry
from src.skills.registry import create_skill_registry

skill_registry = create_skill_registry()
trace_store = RunTraceStore(
    settings.agent_trace_path,
    enabled=settings.agent_trace_enabled,
    max_records=settings.agent_trace_max_records,
)

# Create agent loop
from openai import AsyncOpenAI

from src.agent.loop import AgentLoop

llm_client = AsyncOpenAI(
    api_key=settings.api_key,
    base_url=settings.ollama_base_url,
)

agent_loop = AgentLoop(
    llm_client=llm_client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
    memory=memory,
    trace_store=trace_store,
)

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    instruction: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    mode: str = "think"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ResetSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


# ── Helpers ──────────────────────────────────────────────────────────────


def get_host_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((settings.usrp_ip, 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


# ── REST endpoints ────────────────────────────────────────────────────────


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "models": settings.model_map,
        "knowledge_documents": knowledge_base.document_count,
        "uhd_available": controller.driver_available,
    }


@app.get("/api/hardware_status")
def get_hardware_status() -> dict:
    return {
        "server_ip": get_host_ip(),
        "usrp_01_ip": settings.usrp_ip,
        "usrp_02_ip": "Offline",
        "diagnostics": controller.get_hardware_snapshot(),
        "visualization_active": controller.get_visualization_snapshot().get(
            "active", False
        ),
        "video_stream": video_controller.get_state(),
        "knowledge_documents": knowledge_base.document_count,
    }


@app.get("/api/visualization")
def get_visualization_state() -> dict:
    return controller.get_visualization_snapshot()


@app.post("/api/hardware/stop")
async def stop_hardware_task() -> dict:
    video_result = video_controller.stop()
    hw_result = controller.stop_hardware_task()
    # Flush any pending console completion — the USRP is now idle
    agent_loop._flush_pending_completion()
    return {"hardware": hw_result, "video_stream": video_result}


@app.post("/api/session/reset")
def reset_session(payload: ResetSessionRequest) -> dict:
    memory.reset(payload.session_id)
    return {"status": "success", "message": "Session context cleared."}


@app.get("/api/agent/runs")
def list_agent_runs(limit: int = 20) -> dict:
    """Return the newest structured Agent work logs."""
    return {
        "status": "success",
        "enabled": trace_store.enabled,
        "retention_max_records": trace_store.max_records,
        "retention_trim_to_records": trace_store.trim_to_records,
        "runs": trace_store.list_recent(limit=limit),
    }


@app.get("/api/agent/runs/{run_id}")
def get_agent_run(run_id: str) -> dict:
    """Return one complete work log by run ID."""
    record = trace_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent run log not found")
    return record


# ── Chat endpoints ───────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    """Non-streaming chat — collects all events and returns final result."""
    reply_parts: list[str] = []
    tool_logs: list[str] = []
    final_payload: dict[str, Any] = {}
    run_id = str(uuid.uuid4())
    request_started_at = time.perf_counter()

    # Notify console synchronously — must arrive before run_stream starts
    await _post_console_event(
        agent_id="usrp",
        run_id=run_id,
        session_id=payload.session_id,
        phase="started",
        task=payload.instruction,
        tool_name="",
    )

    try:
        async for event in agent_loop.run_stream(
            instruction=payload.instruction,
            session_id=payload.session_id,
            mode=payload.mode,
            temperature=payload.temperature,
            run_id=run_id,
            request_started_at=request_started_at,
        ):
            t = event.get("type")
            if t == "delta" and event.get("text"):
                reply_parts.append(str(event["text"]))
            elif t == "tool_log" and event.get("line"):
                tool_logs.append(str(event["line"]))
            elif t == "done":
                final_payload = event
            elif t == "error":
                return {
                    "status": "failed",
                    "run_id": event.get("run_id", run_id),
                    "message": event.get("message", "Unknown error"),
                    "diagnosis": event.get("diagnosis"),
                    "reply": "".join(reply_parts),
                    "hardware_logs": tool_logs,
                    "updated_history": memory.get_history(payload.session_id),
                    "active_mode": agent_loop.normalize_mode(payload.mode),
                    "active_model": agent_loop.resolve_model(payload.mode),
                    "active_temperature": agent_loop.resolve_temperature(
                        payload.mode, payload.temperature
                    ),
                }
    except Exception as exc:
        return {
            "status": "failed",
            "run_id": run_id,
            "message": str(exc),
            "reply": "".join(reply_parts),
            "hardware_logs": tool_logs,
            "updated_history": memory.get_history(payload.session_id),
            "active_mode": agent_loop.normalize_mode(payload.mode),
            "active_model": agent_loop.resolve_model(payload.mode),
            "active_temperature": agent_loop.resolve_temperature(
                payload.mode, payload.temperature
            ),
        }

    return {
        "run_id": final_payload.get("run_id", run_id),
        "status": final_payload.get("status", "failed"),
        "termination_reason": final_payload.get("termination_reason"),
        "diagnosis": final_payload.get("diagnosis"),
        "reply": final_payload.get("reply", "".join(reply_parts)),
        "hardware_logs": final_payload.get("hardware_logs", tool_logs),
        "updated_history": final_payload.get(
            "updated_history", memory.get_history(payload.session_id)
        ),
        "active_mode": final_payload.get(
            "active_mode", agent_loop.normalize_mode(payload.mode)
        ),
        "active_model": final_payload.get(
            "active_model", agent_loop.resolve_model(payload.mode)
        ),
        "active_temperature": final_payload.get(
            "active_temperature",
            agent_loop.resolve_temperature(payload.mode, payload.temperature),
        ),
        "active_route": final_payload.get("active_route"),
        "route_confidence": final_payload.get("route_confidence"),
        "skill_matched": final_payload.get("skill_matched"),
        "workflow": final_payload.get("workflow"),
        "required_actions": final_payload.get("required_actions", []),
        "completed_actions": final_payload.get("completed_actions", []),
        "timing": final_payload.get("timing"),
    }


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE streaming endpoint — token deltas, tool logs, and final payload."""

    async def event_gen() -> AsyncIterator[str]:
        import asyncio as _asyncio

        run_id = str(uuid.uuid4())
        request_started_at = time.perf_counter()
        # Notify console synchronously — must arrive before run_stream starts
        await _post_console_event(
            agent_id="usrp",
            run_id=run_id,
            session_id=payload.session_id,
            phase="started",
            task=payload.instruction,
            tool_name="",
        )

        async for event in agent_loop.run_stream(
            instruction=payload.instruction,
            session_id=payload.session_id,
            mode=payload.mode,
            temperature=payload.temperature,
            run_id=run_id,
            request_started_at=request_started_at,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            # Force the event loop to flush after each delta so the browser
            # renders tokens incrementally instead of receiving one big burst.
            if event.get("type") == "delta":
                await _asyncio.sleep(0)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Mount MCP + static ────────────────────────────────────────────────────

# MCP SSE endpoint — external agents connect here
app.mount("/mcp", mcp_server.sse_app())

# Static web UI
app.mount(
    "/", StaticFiles(directory=str(settings.static_dir), html=True), name="web_root"
)
