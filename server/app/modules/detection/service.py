from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import TypeAdapter
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import CacheKeys, CacheLayer
from app.core.database import Database
from app.core.enums import ReviewStatus
from app.core.errors import NotFound
from app.modules.detection.models import CallerBlock, ReviewItem
from app.modules.detection.schemas import CallerBlockWrite, ReviewClose


class CallerBlockService:
    def __init__(self, session: AsyncSession, cache: CacheLayer) -> None:
        self._session = session
        self._cache = cache

    async def list(self, organization_id: uuid.UUID) -> Sequence[CallerBlock]:
        result = await self._session.execute(
            select(CallerBlock).where(CallerBlock.organization_id == organization_id)
        )
        return result.scalars().all()

    async def add(self, organization_id: uuid.UUID, data: CallerBlockWrite) -> CallerBlock:
        block = CallerBlock(organization_id=organization_id, **data.model_dump())
        self._session.add(block)
        await self._session.flush()
        await PreAnswerGate.invalidate(self._cache, organization_id)
        return block

    async def remove(self, organization_id: uuid.UUID, block_id: uuid.UUID) -> None:
        block = await self._session.get(CallerBlock, block_id)
        if block is None or block.organization_id != organization_id:
            raise NotFound("Block not found", block_id=str(block_id))
        await self._session.delete(block)
        await PreAnswerGate.invalidate(self._cache, organization_id)


class PreAnswerGate:
    """Layer 0: a blocked caller is refused before a room exists. The block list
    is in the shared cache, dropped on every change."""

    BLOCKS = TypeAdapter(frozenset[str])

    def __init__(self, database: Database, cache: CacheLayer) -> None:
        self._database = database
        self._cache = cache

    async def blocked(self, organization_id: uuid.UUID, identifiers: Sequence[str | None]) -> bool:
        key = CacheKeys.build(organization_id, "blocks")
        blocks = await self._cache.get(key, self.BLOCKS, lambda: self._load(organization_id))
        return any(value and value.lower() in blocks for value in identifiers)

    async def _load(self, organization_id: uuid.UUID) -> frozenset[str]:
        async with self._database.session() as session:
            result = await session.execute(
                select(CallerBlock.identifier).where(
                    CallerBlock.organization_id == organization_id,
                    or_(
                        CallerBlock.expires_at.is_(None),
                        CallerBlock.expires_at > datetime.now(UTC),
                    ),
                )
            )
            return frozenset(value.lower() for value in result.scalars())

    @staticmethod
    async def invalidate(cache: CacheLayer, organization_id: uuid.UUID) -> None:
        await cache.delete_prefix(CacheKeys.build(organization_id, "blocks"))


class ReviewService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self, organization_id: uuid.UUID, call_id: uuid.UUID, reason: str, turn_index: int | None
    ) -> None:
        """Safe to run again: the same reason for the same turn is stored once."""
        await self._session.execute(
            insert(ReviewItem)
            .values(
                id=uuid.uuid4(),
                organization_id=organization_id,
                call_id=call_id,
                reason=reason,
                turn_index=turn_index,
            )
            .on_conflict_do_nothing(constraint="uq_review_call_reason_turn")
        )

    async def list(
        self, organization_id: uuid.UUID, status: ReviewStatus | None
    ) -> Sequence[ReviewItem]:
        statement = select(ReviewItem).where(ReviewItem.organization_id == organization_id)
        if status is not None:
            statement = statement.where(ReviewItem.status == status)
        result = await self._session.execute(statement.order_by(ReviewItem.created_at.desc()))
        return result.scalars().all()

    async def close(self, item_id: uuid.UUID, data: ReviewClose) -> ReviewItem:
        item = await self._session.get(ReviewItem, item_id)
        if item is None:
            raise NotFound("Review item not found", item_id=str(item_id))
        item.status, item.reviewer, item.note = ReviewStatus.CLOSED, data.reviewer, data.note
        return item
