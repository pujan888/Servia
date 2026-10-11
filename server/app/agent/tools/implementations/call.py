from __future__ import annotations

from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import ToolContext, ToolOutcome
from app.core.enums import ToolStatus


class EndCall(ToolImplementation):
    """Asks the transport to end the call after the goodbye has been heard."""

    name = "end_call"
    deterministic = True
    CONTROL = "end_call"

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(ToolStatus.OK, "The call will end now.", control=self.CONTROL)


class EscalateToHuman(ToolImplementation):
    """The model's way to ask for a handoff. The handoff itself is code: the
    model does not choose the words, the queue or the fallback."""

    name = "escalate_to_human"
    deterministic = True
    CONTROL = "escalate"

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(ToolStatus.OK, "The handoff is starting.", control=self.CONTROL)
