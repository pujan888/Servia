from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.agent.tools.contracts import Refusal, ToolContext, ToolOutcome
from app.core.enums import CallOutcome, SideEffect


class ToolImplementation(ABC):
    """The code behind a tool. It never decides whether it may run; the gateway
    does. It never sees another tenant; the context is already scoped."""

    name: ClassVar[str]
    side_effect: ClassVar[SideEffect] = SideEffect.NONE
    receipt_eta: ClassVar[str] = "soon"
    receipt_outcome: ClassVar[CallOutcome | None] = None
    answers_questions: ClassVar[bool] = False
    deterministic: ClassVar[bool] = False

    def missing_precondition(self, ctx: ToolContext, arguments: dict[str, Any]) -> Refusal | None:
        return None

    @abstractmethod
    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome: ...
