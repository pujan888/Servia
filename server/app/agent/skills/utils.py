from __future__ import annotations

from app.modules.config.schemas import SkillSpec


class SkillText:
    """The two ways a skill is shown to the model"""

    @staticmethod
    def menu_line(skill: SkillSpec) -> str:
        return f"- {skill.key}: {skill.trigger}"

    @staticmethod
    def full(skill: SkillSpec) -> str:
        lines = [
            f"Procedure: {skill.name} ({skill.key})",
            f"Goal: {skill.goal}",
            f"Steps:\n{skill.steps}",
        ]
        if skill.guardrails:
            lines.append(f"Rules for this procedure: {skill.guardrails}")
        if skill.done_when:
            lines.append(f"Done when: {skill.done_when}")
        if skill.escalate_if:
            lines.append(f"Stop and hand over to a person if: {skill.escalate_if}")
        return "\n".join(lines)
