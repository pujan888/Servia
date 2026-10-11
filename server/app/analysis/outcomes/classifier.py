from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.agent.tools.registry import ToolRegistry
from app.analysis.model import AnalysisModel
from app.core.enums import CallOutcome
from app.modules.calls.models import Handoff, TurnRecord
from app.modules.work.models import ActionOutbox


class OutcomeVerdict(BaseModel):
    outcome: Literal["resolved", "unresolved"]
    reason: str


class OutcomeClassifier:
    """Rules first; the model only when no rule applies. What a receipt means is
    declared by the tool implementation that made it, not listed here. A
    completed transfer comes before every other rule."""

    ORDER = (CallOutcome.CALLBACK_PROMISED, CallOutcome.RESOLVED)
    INSTRUCTIONS = (
        "Read the transcript of a customer support call. Did the caller get what they "
        "called for during this call? Answer 'resolved' or 'unresolved', with a short reason."
    )

    def __init__(self, model: AnalysisModel, registry: ToolRegistry) -> None:
        self._model = model
        self._registry = registry

    async def classify(
        self,
        turns: Sequence[TurnRecord],
        receipts: Sequence[ActionOutbox],
        transcript: str,
        handoffs: Sequence[Handoff] = (),
    ) -> tuple[CallOutcome, str]:
        rule = self.by_rules(turns, receipts, handoffs)
        if rule is not None:
            return rule, "rule"
        verdict = await self._model.structured(self.INSTRUCTIONS, transcript, OutcomeVerdict)
        return CallOutcome(verdict.outcome), f"model: {verdict.reason}"

    def by_rules(
        self,
        turns: Sequence[TurnRecord],
        receipts: Sequence[ActionOutbox],
        handoffs: Sequence[Handoff] = (),
    ) -> CallOutcome | None:
        if any(handoff.level == 0 for handoff in handoffs):
            return CallOutcome.ESCALATED
        declared = set()
        for receipt in receipts:
            implementation = self._registry.get(receipt.implementation)
            if implementation is not None and implementation.receipt_outcome is not None:
                declared.add(implementation.receipt_outcome)
        for outcome in self.ORDER:
            if outcome in declared:
                return outcome
        if not any(turn.transcript_in for turn in turns):
            return CallOutcome.ABANDONED
        last = turns[-1]
        if last.transcript_in and not last.text_spoken:
            return CallOutcome.ABANDONED
        return None
