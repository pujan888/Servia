from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.analysis.model import AnalysisModel
from app.analysis.postcall.utils import TranscriptText
from app.core.enums import SlotSource, Speaker
from app.modules.calls.models import TranscriptLine
from app.modules.config.schemas import SlotKeySpec
from app.modules.memory.schemas import SlotCandidate
from app.modules.memory.utils import SlotValues


class ExtractedSlot(BaseModel):
    key: str
    value: str
    source: Literal["stated", "inferred"]
    confidence: float
    line: int


class ExtractedSlots(BaseModel):
    slots: list[ExtractedSlot]


class SlotExtractor:
    """Finds values for the tenant's declared keys. The model proposes; code
    accepts only a declared key, a confident value of the right type, and
    evidence on a line the caller said. A hedge on that line makes the value
    inferred, whatever the model said."""

    INSTRUCTIONS = (
        "You find lasting facts about the caller in a call transcript. Use only the keys "
        "listed. Leave a key out if the call does not show it. Most calls show few or none.\n"
        "- source 'stated': the caller said it plainly, as a lasting fact about themselves.\n"
        "- source 'inferred': you worked it out, or the caller hedged it or limited it in "
        "time, with words like 'usually', 'I think', 'this week' or 'for now'.\n"
        "- A single booking or request is not a lasting preference.\n"
        "- confidence: from 0 to 1, how sure you are that the value is true now.\n"
        "- line: the number of the caller's line that shows it."
    )
    HEDGES = re.compile(
        r"\b(usually|normally|mostly|generally|probably|maybe|perhaps|i think|i guess|"
        r"might|sometimes|this week|for now|at the moment|right now)\b",
        re.IGNORECASE,
    )

    def __init__(self, model: AnalysisModel, min_confidence: float) -> None:
        self._model = model
        self._min_confidence = min_confidence

    async def extract(
        self, lines: Sequence[TranscriptLine], slot_keys: Sequence[SlotKeySpec]
    ) -> list[SlotCandidate]:
        if not slot_keys or not any(line.speaker == Speaker.CALLER for line in lines):
            return []
        keys = "\n".join(
            f"- {k.key} ({k.value_type}): {k.description} Example: {k.example}" for k in slot_keys
        )
        content = f"Keys:\n{keys}\n\nTranscript:\n{TranscriptText.render(lines)}"
        result = await self._model.structured(self.INSTRUCTIONS, content, ExtractedSlots)
        return self.accept(result.slots, lines, slot_keys)

    def accept(
        self,
        extracted: Sequence[ExtractedSlot],
        lines: Sequence[TranscriptLine],
        slot_keys: Sequence[SlotKeySpec],
    ) -> list[SlotCandidate]:
        keys = {slot_key.key: slot_key for slot_key in slot_keys}
        by_sequence = {line.sequence: line for line in lines}
        accepted = []
        for slot in extracted:
            spec, line = keys.get(slot.key), by_sequence.get(slot.line)
            if spec is None or slot.confidence < self._min_confidence:
                continue
            if line is None or line.speaker != Speaker.CALLER:
                continue
            value = SlotValues.normalise(slot.value, spec.value_type)
            if value is None:
                continue
            stated = slot.source == "stated" and not self.HEDGES.search(line.text)
            accepted.append(
                SlotCandidate(
                    key=slot.key,
                    value=value,
                    source=SlotSource.STATED if stated else SlotSource.INFERRED,
                    confidence=slot.confidence,
                    evidence_turn=line.turn_index,
                )
            )
        return accepted
