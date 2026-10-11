from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import SlotSource


class DigestSlot(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    value: str


class Digest(BaseModel):
    """What the next call is told about earlier calls. Stored as data, so the
    time words are phrased when the call starts, not when the digest was built."""

    model_config = ConfigDict(frozen=True)

    summary: str | None = None
    call_at: datetime | None = None
    slots: tuple[DigestSlot, ...] = ()

    def lines(self, when: str) -> list[str]:
        lines = []
        if self.summary:
            lines.append(f"- Last call, {when}: {self.summary}")
        if self.slots:
            lines.append(
                "- What they told us before: "
                + "; ".join(f"{slot.label}: {slot.value}" for slot in self.slots)
                + "."
            )
        return lines


@dataclass(slots=True, frozen=True)
class LoadedMemory:
    digest: Digest | None
    pending: bool


class SlotDecision(StrEnum):
    INSERT = "insert"
    CONFIRM = "confirm"
    REPLACE = "replace"
    KEEP = "keep"


class SlotCandidate(BaseModel):
    key: str
    value: str
    source: SlotSource
    confidence: float = Field(ge=0, le=1)
    evidence_turn: int | None = None


class SlotChange(BaseModel):
    key: str
    value: str
    source: SlotSource
    decision: SlotDecision
    current: str | None = None


class SlotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    value: str
    source: SlotSource
    confidence: float
    evidence_call_id: uuid.UUID | None
    evidence_turn: int | None
    valid_from: datetime


class SummaryRead(BaseModel):
    call_id: uuid.UUID
    text: str
    call_at: datetime | None


class MemoryRead(BaseModel):
    digest: Digest | None
    slots: list[SlotRead]
    summaries: list[SummaryRead]


class ErasureRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    requested_by: str
    counts: dict[str, int]
    created_at: datetime
