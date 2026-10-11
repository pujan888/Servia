from __future__ import annotations

from loguru import logger

from app.agent.state import CallState
from app.analysis.model import AnalysisModel
from app.core.config import DetectionSettings


class BriefingBuilder:
    """What the person taking over hears, in at most 60 words. The facts come
    from the call state and cost nothing. The one sentence about what the caller
    wants is written by the model in the background every few turns, so it is
    ready when an escalation starts; turns after it are added as said."""

    INSTRUCTIONS = (
        "In one sentence of at most 25 words, say what the customer wants and what has been "
        "done so far. No names, no reference codes."
    )
    REASONS = {
        "human_request": "the caller asked for a person",
        "distress": "the caller may be in danger",
        "compliance": "this needs a person by policy",
        "tool_failures": "the systems failed several times",
        "skill_condition": "the task needs a person",
        "tool": "the agent asked for help",
    }

    def __init__(self, model: AnalysisModel, settings: DetectionSettings) -> None:
        self._model = model
        self._settings = settings

    def due(self, state: CallState) -> bool:
        return state.turn_index > 0 and state.turn_index % self._settings.briefing_every == 0

    async def refresh(self, state: CallState) -> None:
        turn = state.turn_index
        lines = "\n".join(f"{speaker}: {text}" for _, speaker, text in state.transcript)
        try:
            text = await self._model.text(self.INSTRUCTIONS, lines)
        except Exception as error:
            logger.warning("briefing refresh failed: {!r}", error)
            return
        state.briefing, state.briefing_turn = " ".join(text.split()[:25]), turn

    def current(self, state: CallState, reason: str) -> str:
        caller = state.pack.caller
        facts = [f"{caller.full_name}, identity {state.identity_level.name.lower()}."]
        tail = []
        if state.references:
            tail.append(f"References: {', '.join(state.references)}.")
        tail.append(f"Reason: {self.REASONS.get(reason, reason.replace('_', ' '))}.")
        tail.append(f"Frustration: {self._level(state.signals.frustration)}.")
        budget = self._settings.briefing_words - self._words(facts + tail)
        story = [state.briefing] if state.briefing else []
        story += [
            f'They said: "{text}"'
            for index, speaker, text in state.transcript
            if speaker == "caller" and index > state.briefing_turn
        ]
        return " ".join(facts + self._fit(story, budget) + tail)

    def _level(self, frustration: float) -> str:
        if frustration > self._settings.frustration_offer:
            return "high"
        return "medium" if frustration >= self._settings.frustration_note else "low"

    @staticmethod
    def _words(parts: list[str]) -> int:
        return sum(len(part.split()) for part in parts)

    @staticmethod
    def _fit(parts: list[str], budget: int) -> list[str]:
        """Keeps the newest raw turns that fit; the summary sentence goes first."""
        words = " ".join(parts).split()
        return [" ".join(words[-budget:])] if budget > 0 and words else []
