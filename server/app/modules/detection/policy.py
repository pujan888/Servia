from __future__ import annotations

from collections.abc import Sequence

from app.core.config import DetectionSettings
from app.core.enums import Escalation, FlagType, Severity
from app.modules.detection.schemas import Action, CallSignals, Decision, FlagDraft


class EscalationPolicy:
    """The plan's decision table, in order; the first true condition decides.
    The model never decides to escalate. An offer is made once per reason."""

    def __init__(self, settings: DetectionSettings) -> None:
        self._settings = settings

    def decide(
        self, signals: CallSignals, flags: Sequence[FlagDraft], skill_condition: bool
    ) -> Decision:
        types = {flag.type: flag for flag in flags}
        busy = signals.escalation in (Escalation.IN_PROGRESS, Escalation.DONE)
        distress = types.get(FlagType.DISTRESS)
        if distress and distress.severity == Severity.NOW and not busy:
            return Decision(Action.ESCALATE, "distress")
        if FlagType.HUMAN_REQUEST in types and not busy:
            return Decision(Action.ESCALATE, "human_request")
        if FlagType.COMPLIANCE in types and not busy:
            return Decision(Action.ESCALATE, "compliance")
        if FlagType.ABUSE in types and signals.abuse_count >= 2:
            return Decision(Action.END_CALL, "abuse")
        if signals.tool_failures >= self._settings.tool_failure_limit and not busy:
            return Decision(Action.ESCALATE, "tool_failures")
        if skill_condition and not busy:
            return Decision(Action.ESCALATE, "skill_condition")
        if busy:
            return Decision(Action.CONTINUE)
        if signals.frustration > self._settings.frustration_offer:
            return self._offer(signals, Action.OFFER_TRANSFER, "frustration")
        if signals.off_topic_streak >= self._settings.streak_limit:
            return self._offer(signals, Action.OFFER_END, "off_topic")
        if len(signals.unanswered_turns) >= self._settings.unanswered_limit:
            return self._offer(signals, Action.OFFER_TRANSFER, "unanswered")
        return Decision(Action.CONTINUE)

    @staticmethod
    def _offer(signals: CallSignals, action: Action, reason: str) -> Decision:
        if reason in signals.offered:
            return Decision(Action.CONTINUE)
        signals.offered.add(reason)
        return Decision(action, reason)
