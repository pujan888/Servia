"""Background job process: uv run python -m app.jobs_worker"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import select

from app.agent.tools.outbox import OutboxJobHandler
from app.analysis.outcomes.schedule import DailyMeasurements, EvaluationRetention, WeeklyGaps
from app.analysis.postcall.pipeline import PostCallFactory, PostCallJobHandler
from app.core.config import LatencySettings
from app.core.container import Container
from app.core.logging import LogConfig
from app.eval.suite import NightlyRun, SuiteJobHandler, SuiteRunner
from app.modules.accounts.models import Organization
from app.modules.calls.service import LatencyReportService, ToolHealthService
from app.modules.config.service import AgentConfigService
from app.modules.jobs.runner import JobRunner, PeriodicTask
from app.modules.knowledge.ingestion import PublisherFactory, PublishJobHandler
from app.modules.memory.service import ForgetPolicy, TranscriptRetention
from app.modules.releases.canary import CanaryMonitor


class ToolClassMonitor:
    """Warns when a tool's p95 no longer fits its latency class."""

    INTERVAL_S = 300
    WINDOW = timedelta(hours=1)

    def __init__(self, container: Container) -> None:
        self._database = container.database

    async def run(self) -> None:
        since = datetime.now(UTC) - self.WINDOW
        async with self._database.session() as session:
            organizations = (await session.execute(select(Organization.id))).scalars().all()
            for organization_id in organizations:
                config = await AgentConfigService(session).load(organization_id)
                classes = {b.key: b.latency_class for b in config.tools}
                for row in await ToolHealthService(session).report(organization_id, since, classes):
                    if row.class_violation:
                        logger.warning(
                            "tool {} for {} breaks class {}: p95 {} ms",
                            row.tool_key,
                            organization_id,
                            row.latency_class,
                            round(row.p95_ms),
                        )


class LatencyBudgetMonitor:
    """Warns when a tenant's first audio is over the budget, once there are
    enough turns in the window to say so."""

    def __init__(self, container: Container, settings: LatencySettings) -> None:
        self._database = container.database
        self._settings = settings

    async def run(self) -> None:
        since = datetime.now(UTC) - timedelta(hours=self._settings.monitor_window_h)
        async with self._database.session() as session:
            organizations = (await session.execute(select(Organization.id))).scalars().all()
            for organization_id in organizations:
                report = await LatencyReportService(session).report(organization_id, since)
                first = next(stage for stage in report.stages if stage.stage == "first_audio")
                if first.samples < self._settings.monitor_min_samples:
                    continue
                if first.p50 > self._settings.first_audio_p50_ms or (
                    first.p95 > self._settings.first_audio_p95_ms
                ):
                    logger.warning(
                        "first audio for {} is over budget: p50 {} ms, p95 {} ms, {} turns",
                        organization_id,
                        round(first.p50),
                        round(first.p95),
                        first.samples,
                    )


class JobsWorker:
    def __init__(self) -> None:
        self._container = Container()
        self._container.settings.livekit.export_env()
        database, memory = self._container.database, self._container.settings.memory
        monitor = ToolClassMonitor(self._container)
        c, settings = self._container, self._container.settings
        latency = settings.latency
        budget = LatencyBudgetMonitor(self._container, latency)
        self._runner = JobRunner(
            database,
            handlers=[
                OutboxJobHandler(database, self._container.registry),
                PublishJobHandler(database, PublisherFactory.build(self._container)),
                PostCallJobHandler(PostCallFactory.build(self._container)),
                SuiteJobHandler(SuiteRunner(self._container)),
            ],
            periodic=[
                PeriodicTask("tool_class_monitor", monitor.INTERVAL_S, monitor.run),
                PeriodicTask("latency_budget", latency.monitor_interval_s, budget.run),
                PeriodicTask(
                    "canary_monitor", settings.releases.monitor_interval_s, CanaryMonitor(c).run
                ),
                PeriodicTask(
                    "nightly_suite",
                    settings.evaluation.nightly_interval_s,
                    NightlyRun(c).run,
                    last_run=time.monotonic(),
                ),
                PeriodicTask(
                    "daily_measurements",
                    settings.analytics.daily_interval_s,
                    DailyMeasurements(c).run,
                ),
                PeriodicTask(
                    "weekly_gaps", settings.analytics.weekly_interval_s, WeeklyGaps(c).run
                ),
                PeriodicTask(
                    "evaluation_retention",
                    settings.analytics.daily_interval_s,
                    EvaluationRetention(c).run,
                ),
                PeriodicTask(
                    "forget_policy",
                    memory.retention_interval_s,
                    ForgetPolicy(database, memory).run,
                ),
                PeriodicTask(
                    "transcript_retention",
                    memory.retention_interval_s,
                    TranscriptRetention(database).run,
                ),
            ],
        )

    async def run(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        logger.info("jobs worker started")
        try:
            await self._runner.run_forever(stop)
        finally:
            await self._container.aclose()
            logger.info("jobs worker stopped")


if __name__ == "__main__":
    LogConfig.install()
    asyncio.run(JobsWorker().run())
