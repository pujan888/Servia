from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import (
    Eagerness,
    FlagType,
    IdentityLevel,
    LatencyClass,
    Optimisation,
    Priority,
    SlotValueType,
)
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Skill(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_skill_org_key"),)
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    trigger: Mapped[str] = mapped_column(Text())
    goal: Mapped[str] = mapped_column(Text())
    steps: Mapped[str] = mapped_column(Text())
    guardrails: Mapped[str | None] = mapped_column(Text())
    done_when: Mapped[str | None] = mapped_column(Text())
    escalate_if: Mapped[str | None] = mapped_column(Text())
    knowledge_key: Mapped[str | None] = mapped_column(String(64))
    tool_keys: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default="{}"
    )
    service_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id", ondelete="SET NULL")
    )
    system: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")
    position: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")


class ModelProfile(Base, Timestamped, OrganizationScoped):
    __tablename__ = "model_profiles"
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    stt_model: Mapped[str | None] = mapped_column(String(80))
    stt_language: Mapped[str | None] = mapped_column(String(16))
    llm_model: Mapped[str | None] = mapped_column(String(80))
    tts_model: Mapped[str | None] = mapped_column(String(80))
    voice: Mapped[str | None] = mapped_column(String(80))
    eagerness: Mapped[Eagerness] = mapped_column(
        String(8), default=Eagerness.MEDIUM, server_default=Eagerness.MEDIUM.value
    )


class ToolBinding(Base, Identified, Timestamped, OrganizationScoped):
    # ToolBinding is basically a database record that tells your AI system
    #  how a particular tool should behave for a particular customer/tenant
    __tablename__ = "tool_bindings"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_tool_binding_org_key"),)
    key: Mapped[str] = mapped_column(String(64))
    implementation: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text())
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_class: Mapped[LatencyClass] = mapped_column(String(1))
    required_level: Mapped[int] = mapped_column(
        Integer(), default=IdentityLevel.UNKNOWN, server_default="0"
    )
    timeout_ms: Mapped[int] = mapped_column(Integer(), default=2000, server_default="2000")
    max_calls: Mapped[int] = mapped_column(Integer(), default=5, server_default="5")
    filler_category: Mapped[str | None] = mapped_column(String(32))
    fallback_text: Mapped[str] = mapped_column(Text())
    cache_ttl_s: Mapped[int] = mapped_column(Integer(), default=0, server_default="0")
    invalidates: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default="{}"
    )
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


# So ToolBinding = the database configuration that connects a tenant's tool name to its
# implementation
# and tells the system
#  how that tool should be exposed, secured, limited, timed, and handled.


class MemorySlotKey(Base, Identified, Timestamped, OrganizationScoped):
    """One kind of fact the tenant lets the agent remember about a customer.
    Nothing outside this list is ever stored."""

    __tablename__ = "memory_slot_keys"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_slot_key_org_key"),)

    key: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text())
    value_type: Mapped[SlotValueType] = mapped_column(
        String(16), default=SlotValueType.TEXT, server_default=SlotValueType.TEXT.value
    )
    example: Mapped[str] = mapped_column(String(200))
    spoken: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


class EscalationRoute(Base, Identified, Timestamped, OrganizationScoped):
    """Where one type of work item goes. The types listed here are the only
    types the agent may create. `handoff` is the queue for live transfers."""

    __tablename__ = "escalation_routes"
    __table_args__ = (
        UniqueConstraint("organization_id", "work_item_type", name="uq_route_org_type"),
    )

    work_item_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text())
    queue: Mapped[str] = mapped_column(String(64))
    priority: Mapped[Priority] = mapped_column(
        String(8), default=Priority.NORMAL, server_default=Priority.NORMAL.value
    )
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


class DetectionSetting(Base, Identified, Timestamped, OrganizationScoped):
    """A tenant's threshold for one flag type. No row means the platform default."""

    __tablename__ = "detection_settings"
    __table_args__ = (
        UniqueConstraint("organization_id", "flag_type", name="uq_detection_org_type"),
    )

    flag_type: Mapped[FlagType] = mapped_column(String(16))
    threshold: Mapped[float] = mapped_column(Float())
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


class OptimisationFlag(Base, Identified, Timestamped, OrganizationScoped):
    """A tenant's switch for one optimisation: on for this percentage of calls.
    Between 0 and 100 it splits calls, so the latency report can compare."""

    __tablename__ = "optimisation_flags"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_optimisation_org_name"),)

    name: Mapped[Optimisation] = mapped_column(String(32))
    percent: Mapped[int] = mapped_column(Integer())
