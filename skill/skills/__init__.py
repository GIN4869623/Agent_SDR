"""Factory to build all SDR skills from .md definitions.

Each skill is defined by a .md file with YAML frontmatter (triggers, keywords,
system prompt) and a Pydantic schema model for parameter validation.
The handler is resolved at build time from the hardware controllers.
"""

from __future__ import annotations

from pathlib import Path

from skill.skill_spec import SkillSpec, SkillRegistry, load_skill_from_md
from tools.knowledge_tools import SearchKnowledgeRequest
from tools.physical_tools import ScanRequest, StopHardwareTaskRequest, ToneLoopbackRequest
from tools.transmit_tools import AdaptiveTransmitRequest, AutoOptimalTransmitRequest, FskSendRequest
from tools.video_tools import StartVideoStreamRequest

_HERE = Path(__file__).parent

# (md_filename, schema_model) — skill name is read from the .md frontmatter.
_SKILL_DEFS: list[tuple[str, type]] = [
    ("stop.md",                StopHardwareTaskRequest),
    ("physical_scan.md",       ScanRequest),
    ("tone_loopback.md",       ToneLoopbackRequest),
    ("text_transmit.md",       FskSendRequest),
    ("adaptive_transmit.md",   AdaptiveTransmitRequest),
    ("cognitive_select.md",    AutoOptimalTransmitRequest),
    ("video_stream.md",        StartVideoStreamRequest),
    ("knowledge.md",           SearchKnowledgeRequest),
]


def _load_skills() -> dict[str, SkillSpec]:
    """Load all skills from .md files, keyed by skill name."""
    skill_map: dict[str, SkillSpec] = {}
    for md_name, schema_model in _SKILL_DEFS:
        md_path = _HERE / md_name
        skill = load_skill_from_md(
            md_path,
            schema_model=schema_model,
            handler=lambda payload: None,
        )
        skill_map[skill.name] = skill
    return skill_map


def build_all_skills(controller, video_controller=None, knowledge_base=None) -> SkillRegistry:
    """Build all skills with handlers resolved from the hardware controllers.

    Args:
        controller: SDRController instance for hardware tools.
        video_controller: Optional VideoStreamController for video tools.
        knowledge_base: Optional LocalKnowledgeBase for knowledge tools.

    Returns:
        SkillRegistry with all skills registered.
    """
    s = _load_skills()

    # Resolve handlers from controllers
    s["stop_hardware_task"].handler = lambda payload: controller.stop_hardware_task()
    s["perform_physical_scan"].handler = lambda payload: controller.perform_physical_scan(
        **payload.model_dump(),
    )
    s["tone_loopback_visualize"].handler = lambda payload: controller.tone_loopback_visualize(
        **payload.model_dump(),
    )
    s["text_fsk_send_and_receive"].handler = lambda payload: controller.text_fsk_send_and_receive(
        **payload.model_dump(),
    )
    s["adaptive_modulation_transmit"].handler = lambda payload: controller.adaptive_modulation_transmit(
        **payload.model_dump(),
    )
    s["auto_optimal_transmit"].handler = lambda payload: controller.auto_optimal_transmit(
        **payload.model_dump(),
    )

    if video_controller is not None:
        s["start_video_stream"].handler = lambda payload: video_controller.start()

    if knowledge_base is not None:
        s["search_sdr_knowledge"].handler = lambda payload: knowledge_base.search(
            **payload.model_dump(),
        )

    return SkillRegistry(list(s.values()))
