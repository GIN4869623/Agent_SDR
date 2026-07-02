"""Skill registry — loads .md skill definitions and matches user intent.

Delegates to the existing skill/ module (skill_spec.py + skills/*.md).
This is a thin wrapper that provides the factory for `build_all_skills()`.
"""

from __future__ import annotations

from skill.skill_spec import SkillSpec, SkillRegistry

__all__ = ["SkillSpec", "SkillRegistry"]
