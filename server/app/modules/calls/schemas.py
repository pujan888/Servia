from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.enums import CallOutcome, CallStatus, Optimisation, Speaker


class CallStart(BaseModel):
    """Data required to start a new call."""

    user_id: uuid.UUID
    service_id: uuid.UUID | None = None


class CallCredentials(BaseModel):
    """Connection details returned to join the call room."""

    call_id: uuid.UUID
    room_name: str
    server_url: str
    token: str
    identity: str


class CallRead(BaseModel):
    """API representation of a call and its current state."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    service_id: uuid.UUID | None
    room_name: str
    status: CallStatus
    identity_level: int
    worker_region: str | None
    started_at: datetime | None
    ended_at: datetime | None
    outcome: CallOutcome | None
    analysed_at: datetime | None


class TurnRecordRead(BaseModel):
    """API representation of one conversation turn and its timings."""

    model_config = ConfigDict(from_attributes=True)
    turn_index: int
    end_of_turn_ms: int | None
    transcription_ms: int | None
    prompt_ms: int | None
    llm_ttft_ms: int | None
    tts_ttfb_ms: int | None
    first_audio_ms: int | None
    prefix_hash: str | None
    prompt_tokens: int | None
    cached_tokens: int | None
    transcript_in: str | None
    text_spoken: str | None
    active_skill: str | None
    skill_loaded: str | None
    tool_set_changed: bool
    retrieval: list[dict[str, Any]]
    interrupted: bool
    created_at: datetime


class StageLatency(BaseModel):
    """Latency statistics for one processing stage."""

    stage: str
    samples: int
    p50: float | None
    p90: float | None
    p95: float | None


class LatencyReportRead(BaseModel):
    """API representation of latency statistics for an organization."""

    organization_id: uuid.UUID
    since: datetime
    stages: list[StageLatency]
    cache_hit_rate: float | None


class LatencyComparison(BaseModel):
    """Turns of calls with the optimisation on, against calls with it off."""

    optimisation: Optimisation
    on: LatencyReportRead
    off: LatencyReportRead


class CacheKindRead(BaseModel):
    kind: str
    memory: int
    store: int
    miss: int
    hit_rate: float | None


class CallContextRead(BaseModel):
    """API representation of the context pack compiled for the agent."""

    source: str
    pack: dict[str, Any]
    prefix: str
    prefix_hash: str
    sections: list[dict[str, Any]]
    truncations: list[str]


class JournalEntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    turn_index: int
    tool_key: str
    arguments: dict[str, Any]
    decision: str
    reason: str | None
    status: str
    latency_ms: int | None
    result_text: str | None
    from_cache: bool
    created_at: datetime


class ToolHealthRead(BaseModel):
    tool_key: str
    latency_class: str | None
    calls: int
    p95_ms: float | None
    refusal_rate: float
    failure_rate: float
    class_violation: bool


class TranscriptLineRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    turn_index: int
    speaker: Speaker
    text: str


# Simple mental model
# Schema	What it represents
# CallStart	Input: start a call
# CallCredentials	Output: how to connect to the call
# CallRead	Output: call information
# TurnRecordRead	Output: one conversation turn
# StageLatency	Output: latency for one stage
# LatencyReportRead	Output: complete latency report
# CallContextRead	Output: compiled context given to the agent
