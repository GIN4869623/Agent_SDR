"""Skill registry factory — loads all .md skills.

This is a thin wrapper around skill/skills/__init__.py:build_all_skills().
In the new architecture, skills are used for:
    1. Regex-based intent matching (before LLM call)
    2. Prompt injection (system prompt enrichment)

The actual tool execution goes through the MCP server, not through skill handlers.
"""

from __future__ import annotations

from typing import Any

from skill.skills import build_all_skills
from skill.skill_spec import SkillRegistry


def create_skill_registry(
    controller: Any = None,
    video_controller: Any = None,
    knowledge_base: Any = None,
) -> SkillRegistry:
    """Build the skill registry from .md files.

    In the Claude Code architecture, skills only provide:
    - Regex matching (trigger_patterns, trigger_keywords)
    - System prompt injection (the .md body)

    Tool execution goes through the MCP server.
    The handler lambdas are kept for backward compatibility but are
    not used in the new agent loop.
    """
    return build_all_skills(controller, video_controller, knowledge_base)
