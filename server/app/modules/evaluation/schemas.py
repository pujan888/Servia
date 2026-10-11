from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import GateDecision, Mark


class AnnotationWrite(BaseModel):
    call_id: uuid.UUID
    turn_index: int = Field(ge=0)
    mark: Mark
    note: str = Field(min_length=1)
    person: str = Field(min_length=1, max_length=200)
    helpfulness: int | None = Field(default=None, ge=1, le=5)
    faithfulness: int | None = Field(default=None, ge=1, le=5)
    tone: int | None = Field(default=None, ge=1, le=5)


class AnnotationRead(AnnotationWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime


class ScenarioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    source_call_id: uuid.UUID | None
    caller_turns: list[str]
    responses: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    enabled: bool
    review_notes: list[str]


class ScenarioUpdate(BaseModel):
    """What a person changes after reading a recorded scenario."""

    assertions: list[dict[str, Any]]
    enabled: bool


class SuiteRun(BaseModel):
    release_id: uuid.UUID


class SuiteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    release_id: uuid.UUID
    baseline_id: uuid.UUID | None
    nightly: bool
    measurements: dict[str, Any]
    decision: GateDecision | None
    reasons: list[str]
    created_at: datetime


class DailyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    day: str
    measurements: dict[str, Any]


class GapRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    week: str
    size: int
    customers: int
    examples: list[str]
    cohesion: float
    status: str
