from __future__ import annotations

import statistics
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AnalyticsSettings
from app.core.enums import CallChannel, CallOutcome, CallStatus
from app.modules.calls.models import Call, Handoff, TurnRecord


class OutcomeAnalytics:
    """The business measurements for one tenant over a window. Replays are
    never counted: they are tests, not calls."""

    COMPLETED = (CallOutcome.RESOLVED, CallOutcome.CALLBACK_PROMISED)

    def __init__(self, session: AsyncSession, settings: AnalyticsSettings) -> None:
        self._session = session
        self._settings = settings

    async def measure(
        self,
        organization_id: uuid.UUID,
        since: datetime,
        until: datetime,
        release_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        calls = await self._calls(organization_id, since, until, release_id)
        ids = [call.id for call in calls]
        turns = await self._turns(ids)
        handoffs = await self._handoffs(ids)
        outcomes = Counter(call.outcome for call in calls)
        count = len(calls)
        retrievals = [(t, r) for t in turns for r in t.retrieval]
        refusals = [(t, r) for t, r in retrievals if r.get("route") == "none"]
        audio = [t.first_audio_ms for t in turns if t.turn_index > 0 and t.first_audio_ms]
        return {
            "calls": count,
            "containment": self._rate(outcomes[CallOutcome.RESOLVED], count),
            "first_call_resolution": await self._fcr(calls, turns),
            "task_completion": self._rate(sum(outcomes[o] for o in self.COMPLETED), count),
            "escalation_rate": self._rate(outcomes[CallOutcome.ESCALATED], count),
            "escalations_by_reason": dict(Counter(h.reason for h in handoffs)),
            "refusal_rate": self._rate(len(refusals), len(retrievals)),
            "refusals_by_skill": dict(Counter(t.active_skill or "none" for t, _ in refusals)),
            "first_audio_p50_ms": self._percentile(audio, 0.5),
            "first_audio_p95_ms": self._percentile(audio, 0.95),
            "cost": self._cost(calls, turns),
        }

    async def _calls(self, organization_id, since, until, release_id) -> list[Call]:
        statement = select(Call).where(
            Call.organization_id == organization_id,
            Call.channel != CallChannel.REPLAY,
            Call.status == CallStatus.COMPLETED,
            Call.created_at >= since,
            Call.created_at < until,
        )
        if release_id is not None:
            statement = statement.where(Call.release_id == release_id)
        return list((await self._session.scalars(statement)).all())

    async def _turns(self, call_ids: list[uuid.UUID]) -> list[TurnRecord]:
        if not call_ids:
            return []
        return list(
            (
                await self._session.scalars(
                    select(TurnRecord).where(TurnRecord.call_id.in_(call_ids))
                )
            ).all()
        )

    async def _handoffs(self, call_ids: list[uuid.UUID]) -> list[Handoff]:
        if not call_ids:
            return []
        return list(
            (
                await self._session.scalars(select(Handoff).where(Handoff.call_id.in_(call_ids)))
            ).all()
        )

    async def _fcr(self, calls: list[Call], turns: list[TurnRecord]) -> float | None:
        """A resolved call counts if the same customer did not call again about
        the same task within the window."""
        skills: dict[uuid.UUID, set[str]] = {}
        for turn in turns:
            if turn.active_skill:
                skills.setdefault(turn.call_id, set()).add(turn.active_skill)
        resolved = [c for c in calls if c.outcome == CallOutcome.RESOLVED and c.ended_at]
        if not resolved:
            return None
        window = timedelta(days=self._settings.fcr_days)
        first_time = 0
        for call in resolved:
            later = (
                await self._session.scalars(
                    select(Call.id).where(
                        Call.user_id == call.user_id,
                        Call.channel != CallChannel.REPLAY,
                        Call.created_at > call.ended_at,
                        Call.created_at <= call.ended_at + window,
                    )
                )
            ).all()
            later_skills = set().union(*(await self._skills(later))) if later else set()
            if not (skills.get(call.id, set()) & later_skills):
                first_time += 1
        return self._rate(first_time, len(resolved))

    async def _skills(self, call_ids) -> list[set[str]]:
        rows = (
            await self._session.execute(
                select(TurnRecord.call_id, TurnRecord.active_skill).where(
                    TurnRecord.call_id.in_(call_ids), TurnRecord.active_skill.is_not(None)
                )
            )
        ).all()
        by_call: dict[uuid.UUID, set[str]] = {}
        for call_id, skill in rows:
            by_call.setdefault(call_id, set()).add(skill)
        return list(by_call.values())

    def _cost(self, calls: list[Call], turns: list[TurnRecord]) -> dict[str, float]:
        s = self._settings
        prompt = sum(t.prompt_tokens or 0 for t in turns)
        cached = sum(t.cached_tokens or 0 for t in turns)
        completion = sum(t.completion_tokens or 0 for t in turns)
        minutes = sum(
            (c.ended_at - c.started_at).total_seconds() / 60
            for c in calls
            if c.ended_at and c.started_at
        )
        characters = sum(len(t.text_spoken or "") for t in turns)
        total = (
            (prompt - cached) * s.price_prompt_mtok / 1e6
            + cached * s.price_cached_mtok / 1e6
            + completion * s.price_completion_mtok / 1e6
            + minutes * s.price_speech_minute
            + characters * s.price_synthesis_mchar / 1e6
        )
        return {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
            "speech_minutes": round(minutes, 2),
            "synthesis_characters": characters,
            "total": round(total, 4),
            "per_call": round(total / len(calls), 4) if calls else 0.0,
        }

    @staticmethod
    def _rate(part: int, whole: int) -> float | None:
        return round(part / whole, 4) if whole else None

    @staticmethod
    def _percentile(values: list[int], share: float) -> float | None:
        if len(values) < 2:
            return float(values[0]) if values else None
        return round(statistics.quantiles(values, n=100)[round(share * 100) - 1], 1)
