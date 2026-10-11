from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.database import Database
from app.modules.calls.models import TurnRecord
from app.modules.calls.utils import Redactor
from app.modules.calls.writer import BufferedWriter


@dataclass(slots=True)
class TurnDraft:
    """Represents a draft of a turn record."""

    turn_index: int
    release_id: uuid.UUID | None = None
    end_of_turn_ms: int | None = None
    transcription_ms: int | None = None
    prompt_ms: int | None = None
    llm_ttft_ms: int | None = None
    tts_ttfb_ms: int | None = None
    first_audio_ms: int | None = None
    prefix_hash: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    transcript_in: str | None = None
    text_spoken: str | None = None
    tool_set_changed: bool = False
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    active_skill: str | None = None
    skill_loaded: str | None = None
    interrupted: bool = False
    flags: list[dict[str, Any]] = field(default_factory=list)
    relevance: str | None = None
    frustration: float | None = None
    guard_actions: list[str] = field(default_factory=list)
    cache_hits: dict[str, int] = field(default_factory=dict)
    optimisations: list[str] = field(default_factory=list)


class TurnRecorder(BufferedWriter[TurnDraft]):
    """Writes turn records off the turn's critical path. What was said is
    redacted here, before the first write."""

    def __init__(self, database: Database, organization_id: uuid.UUID, call_id: uuid.UUID) -> None:
        super().__init__(database)
        self._organization_id = organization_id
        self._call_id = call_id

    def to_row(self, draft: TurnDraft) -> TurnRecord:
        # Convert a temporary TurnDraft into a database TurnRecord.
        return TurnRecord(
            organization_id=self._organization_id,
            call_id=self._call_id,
            turn_index=draft.turn_index,
            release_id=draft.release_id,
            end_of_turn_ms=draft.end_of_turn_ms,
            transcription_ms=draft.transcription_ms,
            prompt_ms=draft.prompt_ms,
            llm_ttft_ms=draft.llm_ttft_ms,
            tts_ttfb_ms=draft.tts_ttfb_ms,
            first_audio_ms=draft.first_audio_ms,
            prefix_hash=draft.prefix_hash,
            prompt_tokens=draft.prompt_tokens or None,
            cached_tokens=draft.cached_tokens,
            completion_tokens=draft.completion_tokens or None,
            transcript_in=Redactor.redact(draft.transcript_in),
            text_spoken=Redactor.redact(draft.text_spoken),
            tool_set_changed=draft.tool_set_changed,
            retrieval=draft.retrieval,
            active_skill=draft.active_skill,
            skill_loaded=draft.skill_loaded,
            interrupted=draft.interrupted,
            flags=draft.flags,
            relevance=draft.relevance,
            frustration=draft.frustration,
            guard_actions=draft.guard_actions,
            cache_hits=draft.cache_hits,
            optimisations=draft.optimisations,
        )


# TurnDraft objects currently sitting in _buffer,
# convert them into database TurnRecord objects,
#  and save them to the database.
