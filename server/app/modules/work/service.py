from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import JobStatus, TaskStatus
from app.core.errors import Conflict, NotFound
from app.modules.work.models import ActionOutbox, Task, TaskEvidence
from app.modules.work.schemas import TaskCreate
from app.modules.work.utils import TaskReference


class TaskService:
    REFERENCE_ATTEMPTS = 5

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, organization_id: uuid.UUID, data: TaskCreate, reference: str | None = None
    ) -> Task:
        task = Task(
            organization_id=organization_id,
            reference=reference or await self.unique_reference(organization_id),
            resolved_at=datetime.now(UTC) if data.status == TaskStatus.DONE else None,
            **data.model_dump(),
        )
        self._session.add(task)
        await self._session.flush()
        return task

    async def list_for_user(
        self, user_id: uuid.UUID, *, open_only: bool = False, limit: int | None = None
    ) -> Sequence[Task]:
        statement = select(Task).where(Task.user_id == user_id)
        if open_only:
            statement = statement.where(
                Task.status.in_([status for status in TaskStatus if status.is_open])
            )
        statement = statement.order_by(Task.updated_at.desc())
        if limit is not None:
            statement = statement.limit(limit)
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def get_by_reference(self, user_id: uuid.UUID, reference: str) -> Task:
        """A reference given for a request that was merged into an existing work
        item finds that item."""
        normalised = TaskReference.normalise(reference)
        merged = select(TaskEvidence.task_id).where(TaskEvidence.reference == normalised)
        result = await self._session.execute(
            select(Task).where(
                Task.user_id == user_id,
                (Task.reference == normalised) | Task.id.in_(merged),
            )
        )
        task = result.scalars().first()
        if task is None:
            raise NotFound("Task not found", reference=reference)
        return task

    async def open_work_items(
        self, user_id: uuid.UUID, kind: str, since: datetime
    ) -> Sequence[Task]:
        result = await self._session.execute(
            select(Task).where(
                Task.user_id == user_id,
                Task.kind == kind,
                Task.status.in_([status for status in TaskStatus if status.is_open]),
                Task.created_at >= since,
            )
        )
        return result.scalars().all()

    async def list_work_items(
        self, organization_id: uuid.UUID, queue: str | None, open_only: bool
    ) -> Sequence[Task]:
        statement = select(Task).where(
            Task.organization_id == organization_id, Task.kind.is_not(None)
        )
        if queue is not None:
            statement = statement.where(Task.queue == queue)
        if open_only:
            statement = statement.where(
                Task.status.in_([status for status in TaskStatus if status.is_open])
            )
        result = await self._session.execute(statement.order_by(Task.created_at.desc()))
        return result.scalars().all()

    async def unique_reference(self, organization_id: uuid.UUID) -> str:
        for _ in range(self.REFERENCE_ATTEMPTS):
            candidate = TaskReference.generate()
            result = await self._session.execute(
                select(Task.id).where(
                    Task.organization_id == organization_id, Task.reference == candidate
                )
            )
            if result.first() is None:
                return candidate
        raise Conflict("Could not allocate a task reference")


class ReceiptService:
    """The writes a call asked for. A write that failed for good is not a receipt."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> Sequence[ActionOutbox]:
        result = await self._session.execute(
            select(ActionOutbox)
            .where(
                ActionOutbox.organization_id == organization_id,
                ActionOutbox.call_id == call_id,
                ActionOutbox.status != JobStatus.FAILED,
            )
            .order_by(ActionOutbox.created_at)
        )
        return result.scalars().all()
