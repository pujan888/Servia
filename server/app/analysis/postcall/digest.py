from __future__ import annotations

from collections.abc import Sequence

from app.analysis.postcall.utils import Sentences
from app.core.config import MemorySettings
from app.core.enums import SlotSource
from app.core.tokens import TokenCounter
from app.modules.config.schemas import SlotKeySpec
from app.modules.memory.models import MemorySlot
from app.modules.memory.schemas import Digest, DigestSlot, SummaryRead


class DigestBuilder:
    """Makes the memory text for the next call: the latest summary and the
    current spoken slots. The next call's open jobs come live from the context
    pack, so they are not copied here, where they would go stale."""

    WHEN_SAMPLE = "earlier today"

    def __init__(self, settings: MemorySettings, counter: TokenCounter) -> None:
        self._settings = settings
        self._counter = counter

    def build(
        self,
        summary: SummaryRead | None,
        slots: Sequence[MemorySlot],
        slot_keys: Sequence[SlotKeySpec],
    ) -> Digest:
        spoken = {slot_key.key: slot_key for slot_key in slot_keys if slot_key.spoken}
        chosen = sorted(
            (slot for slot in slots if slot.key in spoken),
            key=lambda slot: (-SlotSource(slot.source).rank, -slot.confidence),
        )[: self._settings.digest_slots]
        digest = Digest(
            summary=summary.text if summary else None,
            call_at=summary.call_at if summary else None,
            slots=tuple(DigestSlot(label=spoken[s.key].label, value=s.value) for s in chosen),
        )
        return self._fit(digest)

    def tokens(self, digest: Digest) -> int:
        return self._counter.count("\n".join(digest.lines(self.WHEN_SAMPLE)))

    def _fit(self, digest: Digest) -> Digest:
        """Drops the least certain slot first, then the summary's last sentence."""
        while self.tokens(digest) > self._settings.digest_tokens:
            if digest.slots:
                digest = digest.model_copy(update={"slots": digest.slots[:-1]})
                continue
            sentences = Sentences.split(digest.summary or "")
            if len(sentences) <= 1:
                break
            digest = digest.model_copy(update={"summary": " ".join(sentences[:-1])})
        return digest
