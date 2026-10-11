from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import RunContext
from livekit.agents.llm import Toolset, function_tool

from app.agent.skills.utils import SkillText

if TYPE_CHECKING:
    from app.agent.state import CallState


class SkillLoader:
    """Gives the model a procedure during a call"""

    UNKNOWN = "There is no procedure with that key. Use one from the list you were given."
    ALREADY_ACTIVE = "That procedure is already active. Continue with it."
    PREFACE = "Follow this procedure now. Do not read it to the caller."

    def load(self, state: CallState, key: str) -> str:
        skill = next((s for s in state.config.skills if s.key == key.strip()), None)
        if skill is None:
            return self.UNKNOWN
        if state.active_skill == skill.key:
            return self.ALREADY_ACTIVE
        state.activate_skill(skill.key)
        return f"{self.PREFACE}\n\n{SkillText.full(skill)}"


class SkillToolset(Toolset):
    ID = "skills"

    def __init__(self, loader: SkillLoader) -> None:
        super().__init__(id=self.ID)
        self._loader = loader

    @function_tool
    async def load_skill(self, context: RunContext, key: str) -> str:
        """Get the full procedure for one of the other things you can help with.
        Call this as soon as the caller asks for something on your list that you
        do not already have the procedure for. Use the exact key from the list.
        """
        text = self._loader.load(context.userdata, key)
        refresh = getattr(context.session.current_agent, "refresh_tools", None)
        if refresh is not None:
            await refresh()
        return text


# In LiveKit Agents, RunContext is a context object passed to agent tools and hooks to provide
# access to the current agent session
# , shared user data, and execution state.
