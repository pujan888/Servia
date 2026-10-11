from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import Escalation, FlagType, Relevance, ReviewStatus, Severity


class FlagDraft(BaseModel):
    """One detection result for one turn."""

    type: FlagType
    severity: Severity
    score: float
    evidence: str = ""
    model: str
    turn_index: int = 0


@dataclass(slots=True)
class TurnWindow:
    """What a classifier may read: the caller's new message, the last turns,
    and the active skill's escalation condition."""

    turn_index: int
    text: str
    recent: list[tuple[str, str]] = field(default_factory=list)
    skill_condition: str | None = None


class TurnAssessment(BaseModel):
    """The model's answer, for every type at once: one call, not five."""

    abuse: float = Field(ge=0, le=1)
    distress: float = Field(ge=0, le=1)
    manipulation: float = Field(ge=0, le=1)
    compliance: float = Field(ge=0, le=1)
    human_request: float = Field(ge=0, le=1)
    negativity: float = Field(ge=0, le=1)
    relevance: Literal["on_task", "small_talk", "off_topic", "nonsense"]
    skill_condition: bool = False
    evidence: str = ""


@dataclass(slots=True)
class Detection:
    flags: list[FlagDraft]
    relevance: Relevance
    negativity: float
    skill_condition: bool
    ms: int


class DetectionSettingWrite(BaseModel):
    flag_type: FlagType
    threshold: float = Field(ge=0, le=1)
    enabled: bool = True


class DetectionSettingRead(DetectionSettingWrite):
    model_config = ConfigDict(from_attributes=True)


class CallerBlockWrite(BaseModel):
    identifier: str = Field(min_length=3, max_length=320)
    reason: str = Field(min_length=3)
    expires_at: datetime | None = None


class CallerBlockRead(CallerBlockWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID


class ReviewItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    call_id: uuid.UUID
    turn_index: int | None
    reason: str
    status: ReviewStatus
    reviewer: str | None
    note: str | None
    created_at: datetime


class ReviewClose(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    note: str = Field(min_length=1)


class Action(StrEnum):
    CONTINUE = "continue"
    OFFER_TRANSFER = "offer_transfer"
    OFFER_END = "offer_end"
    ESCALATE = "escalate"
    END_CALL = "end_call"


@dataclass(slots=True, frozen=True)
class Decision:
    action: Action
    reason: str = ""


@dataclass(slots=True)
class CallSignals:
    """What the escalation policy reads. Kept on the call state, updated as each
    turn's detection arrives."""

    frustration: float = 0.0
    abuse_count: int = 0
    off_topic_streak: int = 0
    unanswered_turns: set[int] = field(default_factory=set)
    tool_failures: int = 0
    escalation: Escalation = Escalation.NONE
    escalation_reason: str | None = None
    offered: set[str] = field(default_factory=set)
    flag_types: set[FlagType] = field(default_factory=set)
    last_caller_text: str = ""
