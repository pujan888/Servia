from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import case, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import MemorySettings
from app.core.database import Database
from app.core.enums import SlotSource
from app.modules.accounts.models import Organization
from app.modules.calls.models import Call, CallContext, ToolJournalEntry, TranscriptLine, TurnRecord
from app.modules.memory.models import CallSummary, Erasure, MemoryDigest, MemorySlot
from app.modules.memory.store import MemoryStore


class ErasureService:
    """Removes everything the agent remembers about a customer, in one
    transaction. Business records, such as their jobs, are not memory and stay."""

    MARKER = "[erased]"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def erase(
        self, organization_id: uuid.UUID, user_id: uuid.UUID, requested_by: str
    ) -> Erasure:
        await MemoryStore(self._session).lock(user_id)
        calls = select(Call.id).where(
            Call.organization_id == organization_id, Call.user_id == user_id
        )
        counts = {
            "slots": await self._delete(MemorySlot, organization_id, user_id),
            "summaries": await self._delete(CallSummary, organization_id, user_id),
            "digests": await self._delete(MemoryDigest, organization_id, user_id),
            "transcript_lines": await self._rows(
                update(TranscriptLine)
                .where(TranscriptLine.call_id.in_(calls))
                .values(text=self.MARKER)
            ),
            "turn_records": await self._rows(
                update(TurnRecord)
                .where(TurnRecord.call_id.in_(calls))
                .values(
                    transcript_in=self._marked(TurnRecord.transcript_in),
                    text_spoken=self._marked(TurnRecord.text_spoken),
                    retrieval=[],
                )
            ),
            "journal_entries": await self._rows(
                update(ToolJournalEntry)
                .where(ToolJournalEntry.call_id.in_(calls))
                .values(arguments={}, result_text=self.MARKER)
            ),
            "call_contexts": await self._rows(
                delete(CallContext).where(CallContext.call_id.in_(calls))
            ),
        }
        erasure = Erasure(
            organization_id=organization_id,
            user_id=user_id,
            requested_by=requested_by,
            counts=counts,
        )
        self._session.add(erasure)
        await self._session.flush()
        await self._session.refresh(erasure)
        logger.info("memory of {} erased by {}: {}", user_id, requested_by, counts)
        return erasure

    async def _delete(self, model, organization_id: uuid.UUID, user_id: uuid.UUID) -> int:
        return await self._rows(
            delete(model).where(model.organization_id == organization_id, model.user_id == user_id)
        )

    async def _rows(self, statement) -> int:
        result = await self._session.execute(statement.execution_options(synchronize_session=False))
        return result.rowcount

    def _marked(self, column):
        return case((column.is_(None), None), else_=self.MARKER)


class ForgetPolicy:
    """Removes old, uncertain guesses. What a caller said, or what a system of
    record gave, is never removed by age."""

    def __init__(self, database: Database, settings: MemorySettings) -> None:
        self._database = database
        self._settings = settings

    async def run(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=self._settings.forget_after_days)
        async with self._database.session() as session:
            result = await session.execute(
                delete(MemorySlot).where(
                    MemorySlot.source == SlotSource.INFERRED,
                    MemorySlot.confidence < self._settings.forget_below_confidence,
                    MemorySlot.valid_from < cutoff,
                )
            )
        if result.rowcount:
            logger.info("forgot {} old inferred slots", result.rowcount)


class TranscriptRetention:
    """Clears what was said in calls older than each tenant's retention setting."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def run(self) -> None:
        kept = func.make_interval(0, 0, 0, Organization.transcript_days)
        expired = (Call.organization_id == Organization.id) & (Call.ended_at < func.now() - kept)
        async with self._database.session() as session:
            lines = await session.execute(
                delete(TranscriptLine).where(TranscriptLine.call_id == Call.id, expired)
            )
            turns = await session.execute(
                update(TurnRecord)
                .where(
                    TurnRecord.call_id == Call.id,
                    expired,
                    TurnRecord.transcript_in.is_not(None) | TurnRecord.text_spoken.is_not(None),
                )
                .values(transcript_in=None, text_spoken=None, retrieval=[])
            )
        if lines.rowcount or turns.rowcount:
            logger.info(
                "transcript retention cleared {} lines and {} turns", lines.rowcount, turns.rowcount
            )
