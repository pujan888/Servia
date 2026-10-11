from __future__ import annotations

from dataclasses import dataclass

from app.modules.config.schemas import AgentConfig, SkillSpec
from app.modules.context.schemas import ContextPack


@dataclass(slots=True, frozen=True)
class SkillSelection:
    pinned: tuple[SkillSpec, ...]
    listed: tuple[SkillSpec, ...]
    active_key: str | None


class SkillSelector:
    ESCALATE = "escalate_to_human"
    DEFAULT = "general_inquiry"
    MENU_LIMIT = 20

    def initial(self, pack: ContextPack, config: AgentConfig) -> SkillSelection:
        by_key = {skill.key: skill for skill in config.skills}
        pinned: list[SkillSpec] = []
        focus = self._focus_skill(pack, config) or by_key.get(self.DEFAULT)
        if focus is not None:
            pinned.append(focus)
        escalate = by_key.get(self.ESCALATE)
        if escalate is not None and escalate is not focus:
            pinned.append(escalate)
        pinned_keys = {skill.key for skill in pinned}
        listed = [skill for skill in config.skills if skill.key not in pinned_keys]
        if len(listed) > self.MENU_LIMIT:
            listed = self._for_caller(listed, pack)
        return SkillSelection(
            pinned=tuple(pinned), listed=tuple(listed), active_key=focus.key if focus else None
        )

    def _focus_skill(self, pack: ContextPack, config: AgentConfig) -> SkillSpec | None:
        if pack.focus_service is None:
            return None
        return next(
            (skill for skill in config.skills if skill.service_id == pack.focus_service.id), None
        )

    def _for_caller(self, skills: list[SkillSpec], pack: ContextPack) -> list[SkillSpec]:
        """Return the skill selection for a specific caller."""
        service_ids = {service.id for service in pack.services}
        return [s for s in skills if s.service_id is None or s.service_id in service_ids]


# The goal of SkillSelector is to choose and organize the skills for a particular caller.
#  It picks the most relevant skill as the active one,
#  always keeps escalate_to_human available, and puts the remaining relevant skills into the menu.

# SkillSelector does not execute skills. It prepares the relevant skill menu/context for the agent.