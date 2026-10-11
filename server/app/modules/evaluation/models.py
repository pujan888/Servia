from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import GateDecision, Mark
from app.core.models import Base, Identified, OrganizationScoped, Timestamped


class Scenario(Base, Identified, Timestamped, OrganizationScoped):
    """A recorded conversation: what the agent was given, what the caller said,
    what each tool answered, and what must be true of the agent's behaviour.
    It joins the suite only after a person enables it."""

    __tablename__ = "scenarios"

    name: Mapped[str] = mapped_column(String(200))
    source_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL")
    )
    pack: Mapped[dict[str, Any]] = mapped_column(JSONB)
    caller_turns: Mapped[list[str]] = mapped_column(JSONB)
    responses: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    assertions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    review_notes: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")


class SuiteResult(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "suite_results"

    release_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_releases.id", ondelete="CASCADE"), index=True
    )
    baseline_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("suite_results.id", ondelete="SET NULL")
    )
    nightly: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    measurements: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    decision: Mapped[GateDecision | None] = mapped_column(String(8))
    reasons: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")


class ScenarioRun(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "scenario_runs"

    suite_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("suite_results.id", ondelete="CASCADE"), index=True
    )
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    run_number: Mapped[int] = mapped_column(Integer())
    passed: Mapped[bool] = mapped_column(Boolean())
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    turn_ms: Mapped[list[int]] = mapped_column(JSONB)


class AssertionResult(Base, Identified, OrganizationScoped):
    __tablename__ = "assertion_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenario_runs.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer())
    assertion: Mapped[dict[str, Any]] = mapped_column(JSONB)
    passed: Mapped[bool] = mapped_column(Boolean())
    detail: Mapped[str] = mapped_column(Text())


class JudgeScore(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "judge_scores"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenario_runs.id", ondelete="CASCADE"), index=True
    )
    helpfulness: Mapped[int] = mapped_column(Integer())
    faithfulness: Mapped[int] = mapped_column(Integer())
    tone: Mapped[int] = mapped_column(Integer())
    model: Mapped[str] = mapped_column(String(80))


class Annotation(Base, Identified, Timestamped, OrganizationScoped):
    """A person's mark on one turn of a real call. Scores are optional; they
    are what the judge is calibrated against."""

    __tablename__ = "annotations"

    call_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer())
    mark: Mapped[Mark] = mapped_column(String(4))
    note: Mapped[str] = mapped_column(Text())
    person: Mapped[str] = mapped_column(String(200))
    helpfulness: Mapped[int | None] = mapped_column(Integer())
    faithfulness: Mapped[int | None] = mapped_column(Integer())
    tone: Mapped[int | None] = mapped_column(Integer())


class DailyMeasurement(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "daily_measurements"
    __table_args__ = (UniqueConstraint("organization_id", "day", name="uq_measurement_org_day"),)

    day: Mapped[str] = mapped_column(String(10), index=True)
    measurements: Mapped[dict[str, Any]] = mapped_column(JSONB)


class GapGroup(Base, Identified, Timestamped, OrganizationScoped):
    """A group of questions the agent could not answer, from one week."""

    __tablename__ = "gap_groups"

    week: Mapped[str] = mapped_column(String(10), index=True)
    size: Mapped[int] = mapped_column(Integer())
    customers: Mapped[int] = mapped_column(Integer())
    examples: Mapped[list[str]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="open", server_default="open")
    cohesion: Mapped[float] = mapped_column(Float())
