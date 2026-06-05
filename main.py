from __future__ import annotations

import json
import socket

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import settings
from core.memory import ConversationMemory
from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from hardware.video_stream_controller import VideoStreamController
from tools import ToolRegistry
from tools.knowledge_tools import build_knowledge_tools
from tools.physical_tools import build_physical_tools
from tools.transmit_tools import build_transmit_tools
from tools.video_tools import build_video_tools
from graph.graph_service import GraphService
from mcp_server.sdr_server import create_sdr_mcp_server


memory = ConversationMemory(max_messages=settings.session_history_limit)
knowledge_base = LocalKnowledgeBase(
    app_id=settings.bailian_app_id,
    api_key=settings.bailian_api_key,
)
controller = SDRController(settings.usrp_ip)
video_controller = VideoStreamController()
tool_registry = ToolRegistry(
    [
        *build_physical_tools(controller),
        *build_transmit_tools(controller),
        *build_knowledge_tools(knowledge_base),
        *build_video_tools(video_controller),
    ]
)
graph_service = GraphService(
    settings=settings,
    tool_registry=tool_registry,
    memory=memory,
    knowledge_base=knowledge_base,
)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    instruction: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    mode: str = "think"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ResetSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


def get_host_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((settings.usrp_ip, 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


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
        "visualization_active": controller.get_visualization_snapshot().get("active", False),
        "video_stream": video_controller.get_state(),
        "knowledge_documents": knowledge_base.document_count,
    }


@app.get("/api/visualization")
def get_visualization_state() -> dict:
    return controller.get_visualization_snapshot()


@app.post("/api/hardware/stop")
def stop_hardware_task() -> dict:
    video_result = video_controller.stop()
    hw_result = controller.stop_hardware_task()
    return {
        "hardware": hw_result,
        "video_stream": video_result,
    }


@app.post("/api/session/reset")
def reset_session(payload: ResetSessionRequest) -> dict:
    memory.reset(payload.session_id)
    return {"status": "success", "message": "Session context cleared."}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return graph_service.chat(
            session_id=payload.session_id,
            instruction=payload.instruction,
            mode=payload.mode,
            temperature=payload.temperature,
        )
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "reply": "",
            "hardware_logs": [
                "LangGraph agent processing failed."
            ],
            "updated_history": memory.get_history(payload.session_id),
            "active_mode": graph_service.normalize_mode(payload.mode),
            "active_model": graph_service.resolve_model_name(payload.mode),
            "active_temperature": graph_service.resolve_temperature(payload.mode, payload.temperature),
        }


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE: token deltas (`delta`), tool lines (`tool_log`), final payload (`done`)."""

    def event_gen():
        try:
            for event in graph_service.chat_stream(
                session_id=payload.session_id,
                instruction=payload.instruction,
                mode=payload.mode,
                temperature=payload.temperature,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # pragma: no cover
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.mount(
    "/mcp",
    create_sdr_mcp_server(
        tool_registry=tool_registry,
        knowledge_base=knowledge_base,
        controller=controller,
        video_controller=video_controller,
    ),
)
app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="web_root")
