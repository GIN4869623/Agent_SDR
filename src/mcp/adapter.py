"""Expose the canonical ToolRegistry through FastMCP.

No tool handler or validation rule lives in this module. The standalone Agent
calls ToolRegistry directly, while external agents use this protocol adapter.
"""

from __future__ import annotations

import json
import inspect
import re
from pathlib import Path
from typing import Annotated, Any, Callable

import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core.rag import LocalKnowledgeBase
from hardware.sdr_controller import SDRController
from tools import ToolRegistry, ToolSpec


def create_sdr_mcp(
    tool_registry: ToolRegistry,
    controller: SDRController,
    knowledge_base: LocalKnowledgeBase,
) -> FastMCP:
    """Create an MCP server that delegates every call to ToolRegistry."""

    mcp = FastMCP(
        name="Agent SDR Platform",
        instructions=(
            "Agent SDR Platform — manage USRP X300 software-defined radio hardware. "
            "Use tools to scan spectrum, transmit/receive text with FSK/BPSK/QPSK/16-QAM, "
            "generate and visualize tones, stream video via GNU Radio, query the local "
            "SDR knowledge base, and use web search only as a knowledge fallback. "
            "Default center frequency: 2.4 GHz. Default tone frequency: 100 kHz."
        ),
    )

    def build_mcp_callable(spec: ToolSpec) -> Callable[..., Any]:
        """Build one FastMCP callable from the registry's Pydantic schema."""

        async def invoke(**arguments: Any) -> str:
            return await tool_registry.call(spec.name, arguments)

        parameters: list[inspect.Parameter] = []
        for field_name, field in spec.schema_model.model_fields.items():
            annotation = Annotated[
                field.rebuild_annotation(),
                Field(description=field.description),
            ]
            default = (
                inspect.Parameter.empty
                if field.is_required()
                else field.get_default(call_default_factory=True)
            )
            parameters.append(
                inspect.Parameter(
                    field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        invoke.__name__ = spec.name
        invoke.__qualname__ = spec.name
        invoke.__doc__ = spec.description
        invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters,
            return_annotation=str,
        )
        return invoke

    # The MCP catalogue is generated from the same ToolSpec objects exposed to
    # the internal Agent. No tool names, defaults or parameter constraints are
    # repeated in this transport adapter.
    for spec in tool_registry.all():
        mcp.add_tool(
            build_mcp_callable(spec),
            name=spec.name,
            description=spec.description,
        )

    @mcp.resource(
        uri="sdr://hardware/status",
        name="USRP Hardware Status",
        description="USRP 当前连接状态、中心频率、采样率、增益和调制方式。",
    )
    async def hardware_status() -> str:
        return json.dumps(controller.get_hardware_snapshot(), ensure_ascii=False, indent=2)

    @mcp.resource(
        uri="sdr://hardware/visualization",
        name="Visualization Snapshot",
        description="当前可视化窗口的 IQ 波形和 FFT 频谱实时数据。",
    )
    async def visualization_snapshot() -> str:
        snapshot = controller.get_visualization_snapshot()
        for key in ("iq_inphase", "iq_quadrature", "fft_freq_khz", "fft_magnitude_db"):
            if key in snapshot and hasattr(snapshot[key], "tolist"):
                snapshot[key] = snapshot[key].tolist()
        return json.dumps(snapshot, ensure_ascii=False, indent=2)

    @mcp.resource(
        uri="sdr://knowledge/{query}",
        name="SDR Knowledge Query",
        description="从本地知识库检索 SDR 相关文档资料。",
    )
    async def knowledge_query(query: str) -> str:
        result = knowledge_base.query(query)
        return result or f"知识库未检索到与「{query}」匹配的内容。"

    skills_dir = Path(__file__).resolve().parent.parent.parent / "skill" / "skills"
    for md_path in sorted(skills_dir.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            continue
        frontmatter = yaml.safe_load(match.group(1)) or {}
        name = frontmatter.get("name", md_path.stem)
        description = frontmatter.get("description", name)
        body = re.sub(r"^#\s+.*\n+", "", match.group(2).strip()).strip()

        def make_prompt(prompt_body: str = body):
            async def prompt_fn() -> str:
                return prompt_body

            return prompt_fn

        mcp.prompt(name=name, description=description)(make_prompt())

    @mcp.prompt(
        name="sdr-diagnose",
        description="硬件结果诊断 — 分析 SNR、解码状态并给出参数建议。",
    )
    async def prompt_diagnose() -> str:
        return (
            "请根据刚才的硬件执行结果，分析信道质量（SNR、接收功率），"
            "判断当前调制方式是否合适，如果不合适请给出具体的参数调整建议。"
        )

    return mcp
