"""SkillSpec – an enhanced ToolSpec for the skill-architecture (Approach A).

Each SkillSpec adds system_prompt and trigger rules to the existing ToolSpec,
so the LangGraph supervisor can match skills by intent and inject skill-specific
prompts into the LLM context, instead of relying on OpenAI function-calling.

Skills can be defined either programmatically (SkillSpec dataclass) or via
markdown files with YAML frontmatter, matching the Claude Code skill pattern.
"""

from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from tools import ToolSpec


@dataclass
class SkillSpec:
    """A self-contained SDR skill wrapping a ToolSpec with matching and prompt metadata."""

    name: str
    description: str
    system_prompt: str                      # Injected into LLM context when skill is matched
    schema_model: type[BaseModel]          # Pydantic model for parameter validation
    handler: Callable[[BaseModel], dict[str, Any]]  # Execution function
    category: str = "hardware"
    trigger_patterns: list[str] = field(default_factory=list)   # Regex patterns for matching
    trigger_keywords: list[str] = field(default_factory=list)   # Keywords for confidence boost
    exclude_patterns: list[str] = field(default_factory=list)   # Patterns that disqualify this skill

    # ── matching ──

    def can_handle(self, instruction: str) -> float:
        """Return confidence score 0.0-1.0.

        Returns 0.0 if any exclude_pattern matches (disqualification).
        Otherwise scores based on trigger_patterns (+0.4 each) and trigger_keywords (+0.1 each).
        """
        # Exclusion check first
        for pattern in self.exclude_patterns:
            if re.search(pattern, instruction, re.I):
                return 0.0

        score = 0.0
        for pattern in self.trigger_patterns:
            if re.search(pattern, instruction, re.I):
                score += 0.4
        lowered = instruction.lower()
        for kw in self.trigger_keywords:
            if kw.lower() in lowered:
                score += 0.1
        return min(score, 1.0)

    # ── conversion ──

    def to_tool_spec(self) -> ToolSpec:
        """Derive a plain ToolSpec for ToolRegistry compatibility."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            schema_model=self.schema_model,
            handler=self.handler,
            category=self.category,
        )

    def as_openai_tool(self) -> dict[str, Any]:
        """Generate OpenAI function-calling tool definition (kept for fallback)."""
        return self.to_tool_spec().as_openai_tool()


class SkillRegistry:
    """Register and look up skills by name, category, or pattern matching."""

    def __init__(self, skills: list[SkillSpec]) -> None:
        self._skills: list[SkillSpec] = list(skills)
        self._by_name: dict[str, SkillSpec] = {s.name: s for s in skills}
        self._by_category: dict[str, list[SkillSpec]] = {}
        for s in skills:
            self._by_category.setdefault(s.category, []).append(s)

    def get(self, name: str) -> SkillSpec | None:
        return self._by_name.get(name)

    def get_by_category(self, category: str) -> list[SkillSpec]:
        return list(self._by_category.get(category, []))

    def all(self) -> list[SkillSpec]:
        return list(self._skills)

    def as_tool_specs(self) -> list[ToolSpec]:
        return [s.to_tool_spec() for s in self._skills]

    def match(
        self,
        instruction: str,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[SkillSpec, float]]:
        """Score all skills in category, sort desc by confidence."""
        pool = self.get_by_category(category) if category else self._skills
        scored = [(s, s.can_handle(instruction)) for s in pool]
        scored = [(s, c) for s, c in scored if c >= min_confidence]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def match_best(
        self,
        instruction: str,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> tuple[SkillSpec, float] | None:
        """Return the single best-match skill, or None."""
        results = self.match(instruction, category=category, min_confidence=min_confidence)
        return results[0] if results else None


# ── Markdown loader: Claude Code style skill definitions ──

def load_skill_from_md(
    md_path: str | Path,
    schema_model: type[BaseModel],
    handler: Callable[[BaseModel], dict[str, Any]],
) -> SkillSpec:
    """Parse a .md skill definition file and return a SkillSpec.

    The .md file format (Claude Code skill style):

        ---
        name: skill_name
        description: one-line description
        category: hardware
        trigger_patterns:
          - "regex1"
          - "regex2"
        trigger_keywords:
          - keyword1
          - keyword2
        exclude_patterns:
          - "exclude_regex"
        ---

        # Skill Title (ignored)

        System prompt body content...
        Detailed instructions for the LLM...

    Args:
        md_path: Path to the .md file.
        schema_model: Pydantic model for parameter validation.
        handler: Execution function.

    Returns:
        SkillSpec populated from the markdown file.
    """
    content = Path(md_path).read_text(encoding="utf-8")

    # Split YAML frontmatter from markdown body
    parts = _split_frontmatter(content)
    if parts is None:
        raise ValueError(f"No YAML frontmatter found in {md_path}")

    raw_frontmatter, body = parts
    metadata = yaml.safe_load(raw_frontmatter) or {}

    # Extract fields from frontmatter
    name = metadata.get("name", "")
    description = metadata.get("description", "")
    category = metadata.get("category", "hardware")
    trigger_patterns = metadata.get("trigger_patterns", [])
    trigger_keywords = metadata.get("trigger_keywords", [])
    exclude_patterns = metadata.get("exclude_patterns", [])

    # Body is the system_prompt (strip leading heading and blank lines)
    system_prompt = _strip_markdown_heading(body)

    if not name:
        raise ValueError(f"Skill name is required in {md_path}")
    if not system_prompt:
        raise ValueError(f"Skill system_prompt is required in {md_path}")

    return SkillSpec(
        name=name,
        description=description,
        system_prompt=system_prompt,
        schema_model=schema_model,
        handler=handler,
        category=category,
        trigger_patterns=trigger_patterns,
        trigger_keywords=trigger_keywords,
        exclude_patterns=exclude_patterns,
    )


def _split_frontmatter(content: str) -> tuple[str, str] | None:
    """Split YAML frontmatter from markdown body. Returns (frontmatter, body) or None."""
    content = content.lstrip()
    if not content.startswith("---"):
        return None

    # Find second ---
    end = content.find("---", 3)
    if end == -1:
        return None

    frontmatter = content[3:end].strip()
    body = content[end + 3:].strip()
    return frontmatter, body


def _strip_markdown_heading(body: str) -> str:
    """Remove leading markdown heading (e.g. '# Title') and blank lines."""
    lines = body.splitlines()
    stripped: list[str] = []
    found_content = False
    for line in lines:
        if not found_content:
            if line.startswith("#") or line.strip() == "":
                continue
            found_content = True
        stripped.append(line)
    return "\n".join(stripped).strip()
