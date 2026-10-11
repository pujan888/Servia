from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CallStatus
from app.modules.accounts.models import User
from app.modules.calls.models import Call
from app.modules.memory.models import CallSummary, Erasure, MemoryDigest, MemorySlot
from app.modules.memory.schemas import (
    Digest,
    LoadedMemory,
    SlotCandidate,
    SlotChange,
    SlotDecision,
    SummaryRead,
)
from app.modules.memory.utils import SlotPrecedence


class MemoryStore:
    """Reads and writes one tenant's customer memory. Every query carries the
    organization."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        call_id: uuid.UUID,
        pending_window: timedelta,
    ) -> LoadedMemory:
        """One round trip: the digest, and whether a call that ended recently is
        not in it yet."""
        pending = exists().where(
            Call.organization_id == organization_id,
            Call.user_id == user_id,
            Call.id != call_id,
            Call.status == CallStatus.COMPLETED,
            Call.analysed_at.is_(None),
            Call.ended_at >= datetime.now(UTC) - pending_window,
        )
        row = (
            await self._session.execute(
                select(self._digest_data(organization_id, user_id).label("data"), pending)
            )
        ).one()
        return LoadedMemory(self._digest(row[0]), row[1])

    async def digest(self, organization_id: uuid.UUID, user_id: uuid.UUID) -> Digest | None:
        return self._digest(
            await self._session.scalar(select(self._digest_data(organization_id, user_id)))
        )

    @staticmethod
    def _digest_data(organization_id: uuid.UUID, user_id: uuid.UUID):
        return (
            select(MemoryDigest.data)
            .where(MemoryDigest.organization_id == organization_id, MemoryDigest.user_id == user_id)
            .scalar_subquery()
        )

    @staticmethod
    def _digest(data: dict | None) -> Digest | None:
        return Digest.model_validate(data) if data is not None else None

    async def lock(self, user_id: uuid.UUID) -> None:
        """Memory writes and erasure take the customer's row lock, so they
        never interleave."""
        await self._session.execute(select(User.id).where(User.id == user_id).with_for_update())

    async def erased_since(
        self, organization_id: uuid.UUID, user_id: uuid.UUID, since: datetime
    ) -> bool:
        found = await self._session.scalar(
            select(Erasure.id)
            .where(
                Erasure.organization_id == organization_id,
                Erasure.user_id == user_id,
                Erasure.created_at >= since,
            )
            .limit(1)
        )
        return found is not None

    async def current_slots(
        self, organization_id: uuid.UUID, user_id: uuid.UUID
    ) -> Sequence[MemorySlot]:
        result = await self._session.execute(
            select(MemorySlot)
            .where(
                MemorySlot.organization_id == organization_id,
                MemorySlot.user_id == user_id,
                MemorySlot.valid_to.is_(None),
            )
            .order_by(MemorySlot.key)
        )
        return result.scalars().all()

    async def write_slots(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        call_id: uuid.UUID,
        candidates: Sequence[SlotCandidate],
    ) -> list[SlotChange]:
        current = {slot.key: slot for slot in await self.current_slots(organization_id, user_id)}
        changes = []
        for candidate in candidates:
            existing = current.get(candidate.key)
            decision = SlotPrecedence.decide(
                (existing.source, existing.value) if existing else None,
                candidate.source,
                candidate.value,
            )
            if decision == SlotDecision.CONFIRM:
                existing.confidence = max(existing.confidence, candidate.confidence)
                existing.source = SlotPrecedence.higher(existing.source, candidate.source)
                existing.evidence_call_id = call_id
                existing.evidence_turn = candidate.evidence_turn
            elif decision == SlotDecision.REPLACE:
                existing.valid_to = datetime.now(UTC)
                await self._session.flush()
            elif decision == SlotDecision.KEEP:
                logger.info(
                    "slot {} kept '{}' ({}) over '{}' ({})",
                    candidate.key,
                    existing.value,
                    existing.source,
                    candidate.value,
                    candidate.source,
                )
            if decision in (SlotDecision.INSERT, SlotDecision.REPLACE):
                current[candidate.key] = self._insert(organization_id, user_id, call_id, candidate)
            changes.append(
                SlotChange(
                    key=candidate.key,
                    value=candidate.value,
                    source=candidate.source,
                    decision=decision,
                    current=existing.value if existing else None,
                )
            )
        await self._session.flush()
        return changes

    def _insert(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        call_id: uuid.UUID,
        candidate: SlotCandidate,
    ) -> MemorySlot:
        slot = MemorySlot(
            organization_id=organization_id,
            user_id=user_id,
            key=candidate.key,
            value=candidate.value,
            source=candidate.source,
            confidence=candidate.confidence,
            evidence_call_id=call_id,
            evidence_turn=candidate.evidence_turn,
        )
        self._session.add(slot)
        return slot

    async def summaries(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int,
        exclude_call_id: uuid.UUID | None = None,
    ) -> list[SummaryRead]:
        statement = (
            select(CallSummary.call_id, CallSummary.text, Call.ended_at)
            .join(Call, Call.id == CallSummary.call_id)
            .where(CallSummary.organization_id == organization_id, CallSummary.user_id == user_id)
            .order_by(Call.ended_at.desc().nulls_last())
            .limit(limit)
        )
        if exclude_call_id is not None:
            statement = statement.where(CallSummary.call_id != exclude_call_id)
        result = await self._session.execute(statement)
        return [SummaryRead(call_id=c, text=t, call_at=at) for c, t, at in result.all()]

    async def save_summary(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        call_id: uuid.UUID,
        text: str,
        model: str,
    ) -> None:
        await self._session.execute(
            insert(CallSummary)
            .values(
                id=uuid.uuid4(),
                organization_id=organization_id,
                user_id=user_id,
                call_id=call_id,
                text=text,
                model=model,
            )
            .on_conflict_do_nothing(index_elements=["call_id"])
        )

    async def save_digest(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        digest: Digest,
        call_id: uuid.UUID,
    ) -> None:
        data = digest.model_dump(mode="json")
        await self._session.execute(
            insert(MemoryDigest)
            .values(
                organization_id=organization_id,
                user_id=user_id,
                data=data,
                source_call_id=call_id,
            )
            .on_conflict_do_update(
                index_elements=["user_id"],
                set_={"data": data, "source_call_id": call_id, "updated_at": func.now()},
            )
        )
