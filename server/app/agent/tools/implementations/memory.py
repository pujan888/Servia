from __future__ import annotations

from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import ToolContext, ToolOutcome
from app.agent.utils import SpokenTime
from app.core.database import Database
from app.core.enums import ToolStatus
from app.modules.memory.store import MemoryStore


class GetCallHistory(ToolImplementation):
    """Older call summaries, for when the caller refers back to an earlier call."""

    name = "get_call_history"
    NONE = "There are no earlier calls on record. Do not guess what was said."

    def __init__(self, database: Database, limit: int) -> None:
        self._database = database
        self._limit = limit

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        async with self._database.session() as session:
            summaries = await MemoryStore(session).summaries(
                ctx.organization_id, ctx.caller_id, self._limit, exclude_call_id=ctx.call_id
            )
        if not summaries:
            return ToolOutcome(ToolStatus.EMPTY, self.NONE)
        lines = [f"{SpokenTime.elapsed(s.call_at)}: {s.text}" for s in summaries]
        return ToolOutcome(ToolStatus.OK, "Earlier calls, newest first. " + " | ".join(lines))
