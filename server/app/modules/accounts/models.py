from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Organization(Base, Identified, Timestamped):
    """A business running an agent. Its persona is data, not code."""

    __tablename__ = "organizations"

    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    persona_name: Mapped[str] = mapped_column(String(80), default="Alex")
    persona_instructions: Mapped[str | None] = mapped_column(Text())
    voice: Mapped[str | None] = mapped_column(String(80))
    config_version: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")
    transcript_days: Mapped[int] = mapped_column(Integer(), default=90, server_default="90")
    phone: Mapped[str | None] = mapped_column(String(32))
    safety_message: Mapped[str | None] = mapped_column(Text())
    blocked_phrases: Mapped[list[str]] = mapped_column(
        ARRAY(String(200)), default=list, server_default="{}"
    )


class User(Base, Identified, Timestamped, OrganizationScoped):
    """A customer. `profile` holds whatever was collected at sign-up."""

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("organization_id", "email", name="uq_user_org_email"),)

    full_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(320), index=True)
    phone: Mapped[str | None] = mapped_column(String(32), index=True)
    locale: Mapped[str] = mapped_column(String(16), default="en-US")
    timezone: Mapped[str | None] = mapped_column(String(64))
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")


class Staff(Base, Identified, Timestamped, OrganizationScoped):
    """Someone work can be assigned to."""

    __tablename__ = "staff"

    full_name: Mapped[str] = mapped_column(String(200))
    team: Mapped[str | None] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(320))
