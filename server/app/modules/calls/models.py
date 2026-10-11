from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import CallChannel, CallOutcome, CallStatus, IdentityLevel, Speaker
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Call(Base, Identified, Timestamped, OrganizationScoped):
    """Stores one complete conversation/call and its basic metadata."""

    __tablename__ = "calls"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    service_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="SET NULL")
    )
    room_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[CallStatus] = mapped_column(String(16), default=CallStatus.PENDING, index=True)
    channel: Mapped[CallChannel] = mapped_column(
        String(16), default=CallChannel.WEB, server_default=CallChannel.WEB.value
    )

    identity_level: Mapped[int] = mapped_column(
        Integer(), default=IdentityLevel.UNKNOWN, server_default="0"
    )
    worker_region: Mapped[str | None] = mapped_column(String(64))
    release_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[CallOutcome | None] = mapped_column(String(24))
    analysed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    escalation_reason: Mapped[str | None] = mapped_column(String(64))
    handoff_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class CallContext(Base, Timestamped, OrganizationScoped):
    """Stores the context/knowledge given to the agent when the call starts."""

    __tablename__ = "call_contexts"
    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), primary_key=True
    )
    pack: Mapped[dict[str, Any]] = mapped_column(JSONB)
    prefix: Mapped[str] = mapped_column(Text())
    prefix_hash: Mapped[str] = mapped_column(String(64))
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    truncations: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")


class TurnRecord(Base, Identified, Timestamped, OrganizationScoped):
    """Stores what happened, including timing and LLM usage, for each turn."""

    __tablename__ = "turn_records"
    __table_args__ = (UniqueConstraint("call_id", "turn_index", name="uq_turn_call_index"),)
    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer())
    release_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    end_of_turn_ms: Mapped[int | None] = mapped_column(Integer())
    transcription_ms: Mapped[int | None] = mapped_column(Integer())
    prompt_ms: Mapped[int | None] = mapped_column(Integer())
    llm_ttft_ms: Mapped[int | None] = mapped_column(Integer())
    tts_ttfb_ms: Mapped[int | None] = mapped_column(Integer())
    first_audio_ms: Mapped[int | None] = mapped_column(Integer())
    prefix_hash: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer())
    cached_tokens: Mapped[int | None] = mapped_column(Integer())
    completion_tokens: Mapped[int | None] = mapped_column(Integer())
    transcript_in: Mapped[str | None] = mapped_column(Text())
    text_spoken: Mapped[str | None] = mapped_column(Text())
    tool_set_changed: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    retrieval: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default="[]"
    )
    active_skill: Mapped[str | None] = mapped_column(String(64))
    skill_loaded: Mapped[str | None] = mapped_column(String(64))
    interrupted: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    flags: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default="[]")
    relevance: Mapped[str | None] = mapped_column(String(16))
    frustration: Mapped[float | None] = mapped_column(Float())
    guard_actions: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    cache_hits: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict, server_default="{}")
    optimisations: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")


class ToolJournalEntry(Base, Identified, Timestamped, OrganizationScoped):
    """One row for every tool call, including refusals."""

    __tablename__ = "tool_journal"

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer())
    tool_key: Mapped[str] = mapped_column(String(64), index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    decision: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[int | None] = mapped_column(Integer())
    result_text: Mapped[str | None] = mapped_column(Text())
    idempotency_key: Mapped[str | None] = mapped_column(String(64))
    from_cache: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")


class TranscriptLine(Base, Identified, OrganizationScoped):
    """What was said, as the caller heard it. Built after the call."""

    __tablename__ = "transcript_lines"
    __table_args__ = (UniqueConstraint("call_id", "sequence", name="uq_transcript_call_sequence"),)

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer())
    turn_index: Mapped[int] = mapped_column(Integer())
    speaker: Mapped[Speaker] = mapped_column(String(8))
    text: Mapped[str] = mapped_column(Text())


class Handoff(Base, Identified, Timestamped, OrganizationScoped):
    """One escalation: why, what the person was told, and how far down the
    fallback levels it went."""

    __tablename__ = "handoffs"

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    reason: Mapped[str] = mapped_column(String(64))
    briefing: Mapped[str] = mapped_column(Text())
    level: Mapped[int] = mapped_column(Integer())
    result: Mapped[str] = mapped_column(String(32))
    reference: Mapped[str | None] = mapped_column(String(16))
