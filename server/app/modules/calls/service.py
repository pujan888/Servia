from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Integer, delete, func, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import (
    CallChannel,
    CallOutcome,
    CallStatus,
    IdentityLevel,
    LatencyClass,
    Optimisation,
    ToolStatus,
)
from app.core.errors import Conflict, NotFound
from app.modules.accounts.models import User
from app.modules.calls.models import (
    Call,
    CallContext,
    Handoff,
    ToolJournalEntry,
    TranscriptLine,
    TurnRecord,
)
from app.modules.calls.schemas import (
    CacheKindRead,
    LatencyComparison,
    LatencyReportRead,
    StageLatency,
    ToolHealthRead,
)
from app.modules.calls.utils import RoomName
from app.modules.catalog.service import CatalogService
from app.modules.jobs.service import JobQueue


class CallService:
    """Manages call creation and call status."""

    POST_CALL_JOB = "call.post_call"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start(
        self,
        user: User,
        service_id: uuid.UUID | None,
        channel: CallChannel = CallChannel.WEB,
    ) -> Call:
        """Create a new call."""
        if service_id is not None:
            service = await CatalogService(self._session).get(service_id)
            if service.organization_id != user.organization_id:
                raise Conflict("Service does not belong to the user's organization.")

        call = Call(
            organization_id=user.organization_id,
            user_id=user.id,
            service_id=service_id,
            channel=channel,
            room_name=RoomName.generate(),
        )
        self._session.add(call)
        await self._session.flush()
        return call

    async def get(self, call_id: uuid.UUID) -> Call:
        """Get an existing call."""
        call = await self._session.get(Call, call_id)
        if call is None:
            raise NotFound("Call not found", call_id=str(call_id))
        return call

    async def mark_active(
        self,
        call_id: uuid.UUID,
        identity_level: IdentityLevel,
        worker_region: str,
        release_id: uuid.UUID,
    ) -> Call:
        """Mark the call as active."""
        call = await self.get(call_id)
        call.status = CallStatus.ACTIVE
        call.identity_level = identity_level
        call.worker_region = worker_region
        call.release_id = release_id
        call.started_at = call.started_at or datetime.now(UTC)
        return call

    async def mark_ended(self, call_id: uuid.UUID) -> Call:
        """Mark the call as completed. The post-call job is queued in the same
        transaction, so no ended call is ever missed by the pipeline."""
        call = await self.get(call_id)
        call.status = CallStatus.COMPLETED
        call.ended_at = datetime.now(UTC)
        await JobQueue(self._session).enqueue(
            call.organization_id, self.POST_CALL_JOB, {"call_id": str(call.id)}
        )
        return call

    async def set_outcome(self, call_id: uuid.UUID, outcome: CallOutcome) -> None:
        call = await self.get(call_id)
        call.outcome = outcome

    async def mark_analysed(self, call_id: uuid.UUID) -> None:
        call = await self.get(call_id)
        call.analysed_at = datetime.now(UTC)


class CallContextService:
    """Stores and retrieves the context given to the agent."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, context: CallContext) -> None:
        """Save the context for a call."""
        await self._session.merge(context)

    async def get(self, organization_id: uuid.UUID, call_id: uuid.UUID) -> CallContext | None:
        """Get the context for a call."""
        result = await self._session.execute(
            select(CallContext).where(
                CallContext.organization_id == organization_id, CallContext.call_id == call_id
            )
        )
        return result.scalar_one_or_none()


class HandoffService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, handoff: Handoff) -> Handoff:
        self._session.add(handoff)
        await self._session.flush()
        call = await self._session.get(Call, handoff.call_id)
        call.escalation_reason, call.handoff_id = handoff.reason, handoff.id
        return handoff

    async def list_for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> Sequence[Handoff]:
        result = await self._session.execute(
            select(Handoff)
            .where(Handoff.organization_id == organization_id, Handoff.call_id == call_id)
            .order_by(Handoff.created_at)
        )
        return result.scalars().all()


class TranscriptService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace(self, call: Call, lines: list[TranscriptLine]) -> None:
        """Safe to run again: the call's earlier lines are removed first."""
        await self._session.execute(delete(TranscriptLine).where(TranscriptLine.call_id == call.id))
        self._session.add_all(lines)
        await self._session.flush()

    async def list_for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> Sequence[TranscriptLine]:
        result = await self._session.execute(
            select(TranscriptLine)
            .where(
                TranscriptLine.organization_id == organization_id,
                TranscriptLine.call_id == call_id,
            )
            .order_by(TranscriptLine.sequence)
        )
        return result.scalars().all()


class TurnRecordService:
    """Stores and retrieves conversation turn records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> Sequence[TurnRecord]:
        """List all turn records for a call."""
        result = await self._session.execute(
            select(TurnRecord)
            .where(TurnRecord.organization_id == organization_id, TurnRecord.call_id == call_id)
            .order_by(TurnRecord.turn_index)
        )
        return result.scalars().all()


class LatencyReportService:
    """Stores and retrieves conversation turn records."""

    STAGES = (
        ("end_of_turn", TurnRecord.end_of_turn_ms),
        ("transcription", TurnRecord.transcription_ms),
        ("prompt", TurnRecord.prompt_ms),
        ("llm_first_token", TurnRecord.llm_ttft_ms),
        ("tts_first_byte", TurnRecord.tts_ttfb_ms),
        ("first_audio", TurnRecord.first_audio_ms),
    )

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def report(
        self,
        organization_id: uuid.UUID,
        since: datetime,
        optimisation: Optimisation | None = None,
        enabled: bool = True,
    ) -> LatencyReportRead:
        """Generate a latency report for a given organization since a specific datetime.
        Turn 0 is the greeting: it has its own stage, and is left out of the
        turn stages. With an optimisation, only calls with it on, or off."""
        scope = [TurnRecord.organization_id == organization_id, TurnRecord.created_at >= since]
        if optimisation is not None:
            has = TurnRecord.optimisations.contains([optimisation.value])
            scope.append(has if enabled else ~has)
        turns = (*scope, TurnRecord.turn_index > 0)
        stages = [await self._stage(name, column, turns) for name, column in self.STAGES]
        greeting = (*scope, TurnRecord.turn_index == 0)
        stages.append(await self._stage("greeting", TurnRecord.first_audio_ms, greeting))
        return LatencyReportRead(
            organization_id=organization_id,
            since=since,
            stages=stages,
            cache_hit_rate=await self._cache_hit_rate(scope),
        )

    async def compare(
        self, organization_id: uuid.UUID, since: datetime, optimisation: Optimisation
    ) -> LatencyComparison:
        return LatencyComparison(
            optimisation=optimisation,
            on=await self.report(organization_id, since, optimisation, enabled=True),
            off=await self.report(organization_id, since, optimisation, enabled=False),
        )

    async def _stage(self, name: str, column, scope) -> StageLatency:
        result = await self._session.execute(
            select(
                func.count(column),
                func.percentile_cont(0.5).within_group(column),
                func.percentile_cont(0.9).within_group(column),
                func.percentile_cont(0.95).within_group(column),
            ).where(*scope, column.is_not(None))
        )
        samples, p50, p90, p95 = result.one()
        return StageLatency(stage=name, samples=samples, p50=p50, p90=p90, p95=p95)

    async def _cache_hit_rate(self, scope) -> float | None:
        result = await self._session.execute(
            select(
                func.sum(TurnRecord.cached_tokens),
                func.sum(TurnRecord.prompt_tokens),
            ).where(
                *scope,
                TurnRecord.turn_index > 1,
            )
        )

        cached, prompt = result.one()

        return round(cached / prompt, 3) if prompt else None


class CacheReportService:
    """Hits and misses of the shared cache, by key type, from the turn records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def report(self, organization_id: uuid.UUID, since: datetime) -> list[CacheKindRead]:
        pairs = func.jsonb_each_text(TurnRecord.cache_hits).table_valued("key", "value")
        result = await self._session.execute(
            select(pairs.c.key, func.sum(pairs.c.value.cast(Integer)))
            .select_from(TurnRecord)
            .join(pairs, true())
            .where(TurnRecord.organization_id == organization_id, TurnRecord.created_at >= since)
            .group_by(pairs.c.key)
        )
        counts: dict[str, dict[str, int]] = {}
        for key, total in result.all():
            kind, outcome = key.rsplit(".", 1)
            counts.setdefault(kind, {"memory": 0, "store": 0, "miss": 0})[outcome] = total
        return [self._row(kind, c) for kind, c in sorted(counts.items())]

    @staticmethod
    def _row(kind: str, counts: dict[str, int]) -> CacheKindRead:
        total = sum(counts.values())
        hits = counts["memory"] + counts["store"]
        return CacheKindRead(
            kind=kind, **counts, hit_rate=round(hits / total, 3) if total else None
        )


class ToolJournalService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> Sequence[ToolJournalEntry]:
        result = await self._session.execute(
            select(ToolJournalEntry)
            .where(
                ToolJournalEntry.organization_id == organization_id,
                ToolJournalEntry.call_id == call_id,
            )
            .order_by(ToolJournalEntry.created_at)
        )
        return result.scalars().all()


class ToolHealthService:
    """p95 latency, refusal rate and failure rate for each tool, and whether the
    p95 breaks the tool's latency class."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def report(
        self,
        organization_id: uuid.UUID,
        since: datetime,
        classes: dict[str, LatencyClass],
    ) -> list[ToolHealthRead]:
        entry = ToolJournalEntry
        executed = (entry.decision == "permitted") & entry.from_cache.is_(False)
        result = await self._session.execute(
            select(
                entry.tool_key,
                func.count(),
                func.percentile_cont(0.95).within_group(entry.latency_ms).filter(executed),
                func.count().filter(entry.decision == "refused"),
                func.count().filter(entry.status == ToolStatus.FAILED),
            )
            .where(entry.organization_id == organization_id, entry.created_at >= since)
            .group_by(entry.tool_key)
            .order_by(entry.tool_key)
        )
        return [self._row(*row, classes.get(row[0])) for row in result.all()]

    def _row(self, key, calls, p95, refused, failed, latency_class) -> ToolHealthRead:
        limit = latency_class.p95_limit_ms if latency_class else None
        return ToolHealthRead(
            tool_key=key,
            latency_class=latency_class,
            calls=calls,
            p95_ms=p95,
            refusal_rate=round(refused / calls, 3),
            failure_rate=round(failed / calls, 3),
            class_violation=bool(limit and p95 and p95 > limit),
        )
