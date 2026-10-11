from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import JobStatus
from app.modules.jobs.models import Job


class JobQueue:
    """A durable queue in Postgres. `FOR UPDATE SKIP LOCKED` lets many workers
    take jobs at the same time without taking the same job twice."""

    BACKOFF_BASE_S = 2

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, organization_id: uuid.UUID, kind: str, payload: dict[str, Any]) -> Job:
        job = Job(
            organization_id=organization_id,
            kind=kind,
            payload=payload,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def claim(self) -> Job | None:
        result = await self._session.execute(
            select(Job)
            .where(Job.status == JobStatus.PENDING, Job.run_after <= datetime.now(UTC))
            .order_by(Job.run_after)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is not None:
            job.status = JobStatus.RUNNING
            job.attempts += 1
        return job

    async def complete(self, job_id: uuid.UUID) -> None:
        job = await self._session.get(Job, job_id)
        job.status = JobStatus.DONE
        job.last_error = None

    async def fail(self, job_id: uuid.UUID, error: str, max_attempts: int) -> bool:
        """Returns True if the job will not be tried again."""
        job = await self._session.get(Job, job_id)
        job.last_error = error[:2000]
        if job.attempts >= max_attempts:
            job.status = JobStatus.FAILED
            return True
        job.status = JobStatus.PENDING
        job.run_after = datetime.now(UTC) + timedelta(seconds=self.BACKOFF_BASE_S**job.attempts)
        return False
