from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from loguru import logger

from app.analysis.model import AnalysisModel
from app.analysis.postcall.utils import DayResolver, Sentences


class Summariser:
    """Writes the note the next agent reads. A note that loses a reference is
    useless, so each reference given in the call is checked in code. Dates are
    added by code too."""

    INSTRUCTIONS = (
        "You write a note about a finished phone call, for the agent who takes this "
        "customer's next call.\n"
        "- At most {sentences} sentences. No title. No list.\n"
        "- Say what the caller wanted, what was agreed, and what is still open.\n"
        "- Copy every reference code, name and date exactly.\n"
        "- Write a day as the weekday, such as 'Thursday'. Never write 'today', "
        "'tomorrow' or 'next week'. Do not work out dates.\n"
        "- Use only what the transcript says."
    )
    MISSING = "\n\nYour last note left out these references. Include each one exactly: {codes}."
    FALLBACK = "References: {codes}."

    def __init__(self, model: AnalysisModel, max_sentences: int) -> None:
        self._model = model
        self._max_sentences = max_sentences
        self._instructions = self.INSTRUCTIONS.format(sentences=max_sentences)

    @property
    def model_name(self) -> str:
        return self._model.name

    async def summarise(
        self,
        transcript: str,
        call_day: date,
        skills: Sequence[str],
        references: Sequence[str],
    ) -> str:
        content = (
            f"Call date: {call_day.strftime('%A %-d %B %Y')}\n"
            f"Tasks handled: {', '.join(skills) or 'none'}\n"
            f"References given in the call: {', '.join(references) or 'none'}\n\n"
            f"Transcript:\n{transcript}"
        )
        text = self._shorten(await self._model.text(self._instructions, content))
        missing = self.missing(text, references)
        if missing:
            logger.info("summary lost {}; writing it again", missing)
            retry = content + self.MISSING.format(codes=", ".join(missing))
            text = self._shorten(await self._model.text(self._instructions, retry))
            missing = self.missing(text, references)
        if missing:
            text = f"{text} {self.FALLBACK.format(codes=', '.join(missing))}"
        return DayResolver.resolve(text, call_day)

    def _shorten(self, text: str) -> str:
        return Sentences.first(" ".join(text.split()), self._max_sentences)

    @staticmethod
    def missing(text: str, references: Sequence[str]) -> list[str]:
        return [reference for reference in references if reference not in text]
