from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from loguru import logger

from app.core.database import Database
from app.modules.jobs.models import Job
from app.modules.jobs.service import JobQueue


class JobHandler(ABC):
    kind: str
    max_attempts: int = 5

    @abstractmethod
    async def handle(self, job: Job) -> None: ...

    async def on_failed(self, job: Job, error: str) -> None:
        """Called when a job has failed and will not be retried."""
        logger.error("job {} ({}) failed for good: {}", job.id, job.kind, error)


@dataclass
class PeriodicTask:
    name: str
    interval_s: float
    run: Callable[[], Awaitable[None]]
    last_run: float = 0.0


class JobRunner:
    def __init__(
        self,
        database: Database,
        handlers: Sequence[JobHandler],
        periodic: Sequence[PeriodicTask] = (),
        poll_s: float = 0.5,
    ) -> None:
        self._database = database
        self._handlers = {h.kind: h for h in handlers}
        self._periodic = periodic
        self._poll_s = poll_s

    async def run_once(self) -> bool:
        """Claim and process one pending job."""
        async with self._database.session() as session:
            job = await JobQueue(session).claim()
        if job is None:
            return False
        handler = self._handlers.get(job.kind)
        try:
            if handler is None:
                raise ValueError(f"No handler for job kind {job.kind}")
            await handler.handle(job)
        except Exception as e:
            logger.warning("job {} ({}) failed: {}", job.id, job.kind, e)
            max_attempts = handler.max_attempts if handler else 1
            async with self._database.session() as session:
                final = await JobQueue(session).fail(job.id, repr(e), max_attempts)
            if final and handler is not None:
                await handler.on_failed(job, repr(e))
            return True
        async with self._database.session() as session:
            await JobQueue(session).complete(job.id)
        return True

    async def run_forever(self, stop: asyncio.Event) -> None:
        # Continuously run periodic tasks and process jobs until shutdown is requested.
        while not stop.is_set():
            await self.run_periodic()
            if not await self.run_once():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_s)
                except TimeoutError:
                    pass

    async def run_periodic(self) -> None:
        """Run periodic tasks whose configured intervals have elapsed."""
        now = time.monotonic()
        for task in self._periodic:
            if now - task.last_run < task.interval_s:
                continue
            task.last_run = now
            try:
                await task.run()
            except Exception:
                logger.exception("periodic task {} failed", task.name)


# Keep checking the job table. When a job is available, take it, give it to the correct handler,
# retry it if it fails, mark it done if it succeeds,
# and keep doing this until the application is stopped
