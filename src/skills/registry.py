"""Skill registry factory for the Agent Loop."""

from __future__ import annotations

from skill.skills import build_all_skills
from skill.skill_spec import SkillRegistry


def create_skill_registry() -> SkillRegistry:
    """Build Skill metadata; executable definitions come from MCP only."""
    return build_all_skills()
