from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import Conflict, NotFound
from app.modules.jobs.service import JobQueue
from app.modules.knowledge.models import (
    AnswerCard,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgePublish,
    RetrievalTest,
)
from app.modules.knowledge.schemas import (
    CardWrite,
    CollectionWrite,
    DocumentWrite,
    RetrievalTestWrite,
)


class CollectionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[KnowledgeCollection]:
        result = await self._session.execute(
            select(KnowledgeCollection)
            .where(KnowledgeCollection.organization_id == organization_id)
            .order_by(KnowledgeCollection.key)
        )
        return result.scalars().all()

    async def get(self, organization_id: uuid.UUID, key: str) -> KnowledgeCollection:
        result = await self._session.execute(
            select(KnowledgeCollection).where(
                KnowledgeCollection.organization_id == organization_id,
                KnowledgeCollection.key == key,
            )
        )
        collection = result.scalar_one_or_none()
        if collection is None:
            raise NotFound("Collection not found", key=key)
        return collection

    async def save(self, organization_id: uuid.UUID, data: CollectionWrite) -> KnowledgeCollection:
        existing = {c.key: c for c in await self.list(organization_id)}
        collection = existing.get(data.key) or KnowledgeCollection(organization_id=organization_id)
        for field, value in data.model_dump().items():
            setattr(collection, field, value)
        self._session.add(collection)
        await self._session.flush()
        return collection


class _CollectionItems:
    model: type

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, collection: KnowledgeCollection) -> Sequence:
        result = await self._session.execute(
            select(self.model)
            .where(self.model.collection_id == collection.id)
            .order_by(self.model.created_at)
        )
        return result.scalars().all()

    async def _upsert(self, collection: KnowledgeCollection, key: str, values: dict):
        result = await self._session.execute(
            select(self.model).where(
                self.model.collection_id == collection.id, self.model.key == key
            )
        )
        row = result.scalar_one_or_none() or self.model(
            organization_id=collection.organization_id, collection_id=collection.id
        )
        for field, value in values.items():
            setattr(row, field, value)
        self._session.add(row)
        await self._session.flush()
        return row


class DocumentService(_CollectionItems):
    model = KnowledgeDocument

    async def save(self, collection: KnowledgeCollection, data: DocumentWrite) -> KnowledgeDocument:
        values = data.model_dump()
        values["metadata_"] = values.pop("metadata")
        return await self._upsert(collection, data.key, values)


class CardService(_CollectionItems):
    model = AnswerCard

    async def save(self, collection: KnowledgeCollection, data: CardWrite) -> AnswerCard:
        return await self._upsert(collection, data.key, data.model_dump())


class RetrievalTestService(_CollectionItems):
    model = RetrievalTest

    async def add(self, collection: KnowledgeCollection, data: RetrievalTestWrite) -> RetrievalTest:
        test = RetrievalTest(
            organization_id=collection.organization_id,
            collection_id=collection.id,
            **data.model_dump(),
        )
        self._session.add(test)
        await self._session.flush()
        return test


class PublishService:
    JOB_KIND = "knowledge.publish"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, collection: KnowledgeCollection) -> KnowledgePublish:
        """The publish record and its job are stored in one transaction."""
        running = await self._session.scalar(
            select(func.count())
            .select_from(KnowledgePublish)
            .where(
                KnowledgePublish.collection_id == collection.id,
                KnowledgePublish.status.in_(("pending", "running")),
            )
        )
        if running:
            raise Conflict(
                "A publish for this collection is already in progress", key=collection.key
            )
        last = await self._session.scalar(
            select(func.max(KnowledgePublish.version)).where(
                KnowledgePublish.collection_id == collection.id
            )
        )
        run = KnowledgePublish(
            organization_id=collection.organization_id,
            collection_id=collection.id,
            version=max(last or 0, collection.active_version or 0) + 1,
        )
        self._session.add(run)
        await self._session.flush()
        await JobQueue(self._session).enqueue(
            collection.organization_id, self.JOB_KIND, {"publish_id": str(run.id)}
        )
        return run

    async def get(self, organization_id: uuid.UUID, publish_id: uuid.UUID) -> KnowledgePublish:
        run = await self._session.get(KnowledgePublish, publish_id)
        if run is None or run.organization_id != organization_id:
            raise NotFound("Publish not found", publish_id=str(publish_id))
        return run
