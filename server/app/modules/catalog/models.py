from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import EnrollmentStatus
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Service(Base, Identified, Timestamped, OrganizationScoped):
    """A field the organization offers: support, delivery, a plan, a product line.

    `instructions` is appended to the persona when a call is about this service,
    which is how one agent covers unrelated use cases.
    """

    __tablename__ = "services"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_service_org_key"),)

    key: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text())
    instructions: Mapped[str | None] = mapped_column(Text())


class Enrollment(Base, Identified, Timestamped, OrganizationScoped):
    """A user joined to a service, plus whatever they told us when they joined."""

    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("user_id", "service_id", name="uq_enrollment_user_service"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[EnrollmentStatus] = mapped_column(
        String(16), default=EnrollmentStatus.ACTIVE, index=True
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
