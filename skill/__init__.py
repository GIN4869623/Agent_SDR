"""Skill definitions for the SDR control platform (Approach A).

Each skill wraps an existing ToolSpec with matching metadata and a system prompt.
The SkillRegistry is used by the LangGraph supervisor node to match user intent
to skills via regex, with LLM function calling kept as a fallback.
"""

from __future__ import annotations

from skill.skill_spec import SkillSpec, SkillRegistry

__all__ = ["SkillSpec", "SkillRegistry"]
