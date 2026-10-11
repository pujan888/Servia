from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.analysis.model import AnalysisModel


class JudgeVerdict(BaseModel):
    helpfulness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)


class LLMJudge:
    """Scores one agent reply with the plan's rubric. It is off until it agrees
    with people's scores on enough turns (`judge_calibrate`)."""

    INSTRUCTIONS = (
        "You grade one reply from a phone support agent. Score each item from 1 to 5.\n"
        "- helpfulness: did the reply answer the caller's real question?\n"
        "- faithfulness: is every claim supported by the facts given? A reply that says "
        "it does not know, when the facts do not answer, is fully faithful.\n"
        "- tone: short, clear and polite, as speech."
    )

    def __init__(self, model: AnalysisModel) -> None:
        self._model = model
        self.name = model.name

    async def score(self, conversation: Sequence[str], reply: str, facts: str) -> JudgeVerdict:
        content = (
            f"Facts the agent had:\n{facts or '(none)'}\n\n"
            "Conversation so far:\n" + "\n".join(conversation) + f"\n\nReply to grade:\n{reply}"
        )
        return await self._model.structured(self.INSTRUCTIONS, content, JudgeVerdict)

    @staticmethod
    def agreement(pairs: Sequence[tuple[JudgeVerdict, JudgeVerdict]]) -> float:
        """The share of turns where the judge is within one point of the person
        on all three items."""
        if not pairs:
            return 0.0
        close = sum(
            all(
                abs(getattr(j, f) - getattr(p, f)) <= 1
                for f in ("helpfulness", "faithfulness", "tone")
            )
            for j, p in pairs
        )
        return close / len(pairs)
