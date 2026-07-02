"""Context assembly — builds the message list for each agent turn.

Claude Code pattern:
    system_prompt + skill_injection + conversation_history + user_message
"""

from __future__ import annotations

from src.prompts import SYSTEM_PROMPT, SKILL_INJECTION_TEMPLATE
from skill.skill_spec import SkillSpec


def build_messages(
    *,
    instruction: str,
    history: list[dict[str, str]] | None = None,
    skill: SkillSpec | None = None,
) -> list[dict[str, str]]:
    """Assemble the full message list for a single agent turn.

    Args:
        instruction: The user's latest message.
        history: Previous conversation turns (user/assistant pairs).
        skill: Matched skill to inject, or None.

    Returns:
        List of message dicts ready for the LLM.
    """
    messages: list[dict[str, str]] = []

    # 1. System prompt
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2. Skill injection (when a skill matches)
    if skill is not None:
        skill_msg = SKILL_INJECTION_TEMPLATE.format(
            skill_name=skill.name,
            skill_description=skill.description,
            skill_prompt=skill.system_prompt,
        )
        messages.append({"role": "system", "content": skill_msg})

    # 3. Conversation history
    for item in history or []:
        messages.append(item)

    # 4. Current user instruction
    messages.append({"role": "user", "content": instruction})

    return messages
