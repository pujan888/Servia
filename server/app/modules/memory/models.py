from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import SlotSource
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class CallSummary(Base, Identified, Timestamped, OrganizationScoped):
    """Episodic memory: what happened in one call, for the next agent."""

    __tablename__ = "call_summaries"

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), unique=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text())
    model: Mapped[str] = mapped_column(String(80))


class MemorySlot(Base, Identified, Timestamped, OrganizationScoped):
    """Semantic memory: one value of one declared key. A replaced value is
    closed, not deleted, so the history stays. At most one value is current."""

    __tablename__ = "memory_slots"
    __table_args__ = (
        Index(
            "uq_memory_slot_current",
            "organization_id",
            "user_id",
            "key",
            unique=True,
            postgresql_where=text("valid_to IS NULL"),
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text())
    source: Mapped[SlotSource] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float())
    evidence_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL")
    )
    evidence_turn: Mapped[int | None] = mapped_column(Integer())
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryDigest(Base, Timestamped, OrganizationScoped):
    """The prepared memory for the next call. One row for each customer."""

    __tablename__ = "digests"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL")
    )


class Erasure(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "erasures"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    requested_by: Mapped[str] = mapped_column(String(200))
    counts: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
