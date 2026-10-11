from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.agent.tools.contracts import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.database import Database
from app.core.enums import IdentityLevel, JobStatus
from app.modules.jobs.models import Job
from app.modules.jobs.runner import JobHandler
from app.modules.jobs.service import JobQueue
from app.modules.work.models import ActionOutbox
from app.modules.work.service import TaskService


@dataclass(slots=True, frozen=True)
class Receipt:
    reference: str
    eta: str
    existing: bool


class OutboxService:
    JOB_KIND = "outbox.execute"

    def __init__(self, database: Database) -> None:
        self._database = database

    async def enqueue(
        self,
        ctx: ToolContext,
        tool_key: str,
        implementation: str,
        arguments: dict[str, Any],
        idempotency_key: str,
        eta: str,
    ) -> Receipt:
        """Stores the write and its job in one transaction. A second request with
        the same key returns the first receipt and stores nothing."""
        async with self._database.session() as session:
            reference = await TaskService(session).unique_reference(ctx.organization_id)
            inserted = await session.execute(
                insert(ActionOutbox)
                .values(
                    id=uuid.uuid4(),
                    organization_id=ctx.organization_id,
                    idempotency_key=idempotency_key,
                    reference=reference,
                    call_id=ctx.call_id,
                    caller_id=ctx.caller_id,
                    identity_level=int(ctx.identity_level),
                    timezone=ctx.timezone,
                    tool_key=tool_key,
                    implementation=implementation,
                    arguments=arguments,
                    context=ctx.signals,
                )
                .on_conflict_do_nothing(index_elements=["idempotency_key"])
                .returning(ActionOutbox.id)
            )
            outbox_id = inserted.scalar_one_or_none()
            if outbox_id is None:
                existing = await session.execute(
                    select(ActionOutbox.reference).where(
                        ActionOutbox.idempotency_key == idempotency_key
                    )
                )
                return Receipt(existing.scalar_one(), eta, existing=True)
            await JobQueue(session).enqueue(
                ctx.organization_id, self.JOB_KIND, {"outbox_id": str(outbox_id)}
            )
        return Receipt(reference, eta, existing=False)


class OutboxJobHandler(JobHandler):
    """Runs a stored write. It rebuilds the tool context from the outbox row,
    because the call that asked for the write may have ended."""

    kind = OutboxService.JOB_KIND

    def __init__(self, database: Database, registry: ToolRegistry) -> None:
        self._database = database
        self._registry = registry

    async def handle(self, job: Job) -> None:
        row = await self._row(job)
        if row.status == JobStatus.DONE:
            return
        implementation = self._registry.get(row.implementation)
        if implementation is None:
            raise LookupError(f"implementation '{row.implementation}' is not registered")

        ctx = ToolContext(
            organization_id=row.organization_id,
            call_id=row.call_id,
            caller_id=row.caller_id,
            identity_level=IdentityLevel(row.identity_level),
            turn_index=0,
            timezone=row.timezone,
            signals=row.context,
            receipt_reference=row.reference,
        )
        outcome = await implementation.execute(ctx, row.arguments)
        async with self._database.session() as session:
            stored = await session.get(ActionOutbox, row.id)
            stored.status = JobStatus.DONE
            stored.result_text = outcome.speech

    async def on_failed(self, job: Job, error: str) -> None:
        row = await self._row(job)
        async with self._database.session() as session:
            stored = await session.get(ActionOutbox, row.id)
            stored.status = JobStatus.FAILED
            stored.result_text = error[:2000]
        logger.error("write {} ({}) failed for good: {}", row.reference, row.tool_key, error)

    async def _row(self, job: Job) -> ActionOutbox:
        async with self._database.session() as session:
            row = await session.get(ActionOutbox, uuid.UUID(job.payload["outbox_id"]))
        if row is None:
            raise LookupError(f"outbox row {job.payload['outbox_id']} not found")
        return row
