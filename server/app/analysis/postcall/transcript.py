from __future__ import annotations

from collections.abc import Sequence

from app.core.enums import Speaker
from app.modules.calls.models import Call, ToolJournalEntry, TranscriptLine, TurnRecord
from app.modules.work.models import ActionOutbox


class TranscriptBuilder:
    """Makes the transcript from the turn records. The agent's lines are what
    the caller heard (`text_spoken`), not what the model wrote: after an
    interruption the rest was never said."""

    def build(
        self,
        call: Call,
        turns: Sequence[TurnRecord],
        journal: Sequence[ToolJournalEntry],
        receipts: Sequence[ActionOutbox],
    ) -> list[TranscriptLine]:
        references = {receipt.idempotency_key: receipt.reference for receipt in receipts}
        tools_by_turn: dict[int, list[str]] = {}
        for entry in journal:
            if entry.decision == "permitted":
                line = self._tool_line(entry, references.get(entry.idempotency_key))
                tools_by_turn.setdefault(entry.turn_index, []).append(line)

        lines: list[TranscriptLine] = []
        for turn in turns:
            entries = [(Speaker.CALLER, turn.transcript_in)]
            entries += [(Speaker.SYSTEM, text) for text in tools_by_turn.get(turn.turn_index, [])]
            entries.append((Speaker.AGENT, turn.text_spoken))
            for speaker, text in entries:
                if text and text.strip():
                    lines.append(self._line(call, turn.turn_index, len(lines) + 1, speaker, text))
        return lines

    @staticmethod
    def _tool_line(entry: ToolJournalEntry, reference: str | None) -> str:
        text = f"{entry.tool_key} {entry.status}"
        if reference:
            text += f", reference {reference}"
        if entry.arguments:
            text += ": " + "; ".join(f"{key}={value}" for key, value in entry.arguments.items())
        return text

    @staticmethod
    def _line(
        call: Call, turn_index: int, sequence: int, speaker: Speaker, text: str
    ) -> TranscriptLine:
        return TranscriptLine(
            organization_id=call.organization_id,
            call_id=call.id,
            sequence=sequence,
            turn_index=turn_index,
            speaker=speaker,
            text=text.strip(),
        )
