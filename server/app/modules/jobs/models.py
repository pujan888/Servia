from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import JobStatus
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Job(Base, Identified, OrganizationScoped, Timestamped):
    """One unit of background work. the table is the queue"""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_queue", "status", "run_after"),)

    kind: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        String(16), default=JobStatus.PENDING, server_default=JobStatus.PENDING.value
    )
    attempts: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_error: Mapped[str | None] = mapped_column(Text())
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
