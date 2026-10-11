from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import JobStatus, Priority, TaskStatus
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Task(Base, Identified, Timestamped, OrganizationScoped):
    """A unit of work the caller may ring about: an order, a ticket, a job, a request.
    A work item the agent makes is a task with a `kind`, so the next call sees it."""

    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("organization_id", "reference", name="uq_task_org_reference"),
    )

    reference: Mapped[str] = mapped_column(String(16), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    service_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="SET NULL"), index=True
    )
    assignee_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("staff.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    details: Mapped[str | None] = mapped_column(Text())
    status: Mapped[TaskStatus] = mapped_column(String(16), default=TaskStatus.OPEN, index=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    kind: Mapped[str | None] = mapped_column(String(64), index=True)
    priority: Mapped[Priority] = mapped_column(
        String(8), default=Priority.NORMAL, server_default=Priority.NORMAL.value
    )
    queue: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float())
    update_count: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")


class ActionOutbox(Base, Identified, Timestamped, OrganizationScoped):
    """A write the agent asked for. The unique idempotency key means the same
    request can never be stored, or done, twice."""

    __tablename__ = "action_outbox"

    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    reference: Mapped[str] = mapped_column(String(16))
    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    caller_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    identity_level: Mapped[int] = mapped_column(Integer())
    timezone: Mapped[str] = mapped_column(String(64))
    tool_key: Mapped[str] = mapped_column(String(64))
    implementation: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        String(16), default=JobStatus.PENDING, server_default=JobStatus.PENDING.value
    )
    result_text: Mapped[str | None] = mapped_column(Text())
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")


class TaskEvidence(Base, Identified, Timestamped, OrganizationScoped):
    """What a caller said that made or updated a work item."""

    __tablename__ = "task_evidence"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL")
    )
    turn_index: Mapped[int | None] = mapped_column(Integer())
    quote: Mapped[str] = mapped_column(Text())
    reference: Mapped[str | None] = mapped_column(String(16), index=True)
