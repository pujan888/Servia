from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import ReleaseStatus
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class AgentRelease(Base, Identified, Timestamped, OrganizationScoped):
    """One version of all of a tenant's configuration. It never changes: a
    change makes a new release. At most two are live: the stable one, and a
    canary with its share of calls."""

    __tablename__ = "agent_releases"
    __table_args__ = (
        UniqueConstraint("organization_id", "version", name="uq_release_org_version"),
    )

    version: Mapped[int] = mapped_column(Integer())
    hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[ReleaseStatus] = mapped_column(
        String(8), default=ReleaseStatus.DRAFT, server_default=ReleaseStatus.DRAFT.value, index=True
    )
    canary_percent: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")
    created_by: Mapped[str] = mapped_column(String(200))
    live_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")


class ReleaseContent(Base, Identified, OrganizationScoped):
    """One item of a release, with the hash of its content."""

    __tablename__ = "release_contents"

    release_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_releases.id", ondelete="CASCADE"), index=True
    )
    item_type: Mapped[str] = mapped_column(String(32))
    item_key: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer())
    content: Mapped[Any] = mapped_column(JSONB)
    hash: Mapped[str] = mapped_column(String(64))
