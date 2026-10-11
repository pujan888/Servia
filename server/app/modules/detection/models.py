from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import ReviewStatus
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class CallerBlock(Base, Identified, Timestamped, OrganizationScoped):
    """A caller the tenant refuses before the call connects: layer 0."""

    __tablename__ = "caller_blocks"
    __table_args__ = (
        UniqueConstraint("organization_id", "identifier", name="uq_block_org_identifier"),
    )

    identifier: Mapped[str] = mapped_column(String(320))
    reason: Mapped[str] = mapped_column(Text())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReviewItem(Base, Identified, Timestamped, OrganizationScoped):
    """A call, or one turn of it, for a person to examine: layer 4."""

    __tablename__ = "review_items"
    __table_args__ = (
        UniqueConstraint(
            "call_id",
            "reason",
            "turn_index",
            name="uq_review_call_reason_turn",
            postgresql_nulls_not_distinct=True,
        ),
    )

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int | None] = mapped_column(Integer())
    reason: Mapped[str] = mapped_column(String(64))
    status: Mapped[ReviewStatus] = mapped_column(
        String(8), default=ReviewStatus.OPEN, server_default=ReviewStatus.OPEN.value, index=True
    )
    reviewer: Mapped[str | None] = mapped_column(String(200))
    note: Mapped[str | None] = mapped_column(Text())
