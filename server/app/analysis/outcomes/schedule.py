from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from app.analysis.outcomes.analytics import OutcomeAnalytics
from app.analysis.outcomes.gaps import GapMiner
from app.core.container import Container
from app.core.enums import CallChannel
from app.modules.accounts.models import Organization
from app.modules.calls.models import Call
from app.modules.evaluation.models import DailyMeasurement, GapGroup, ScenarioRun


class DailyMeasurements:
    """Yesterday's measurements for each tenant, written once and kept."""

    def __init__(self, container: Container) -> None:
        self._container = container

    async def run(self) -> None:
        day = datetime.now(UTC).date() - timedelta(days=1)
        since = datetime.combine(day, time.min, tzinfo=UTC)
        async with self._container.database.session() as session:
            for organization_id in (await session.scalars(select(Organization.id))).all():
                measured = await OutcomeAnalytics(
                    session, self._container.settings.analytics
                ).measure(organization_id, since, since + timedelta(days=1))
                await session.execute(
                    insert(DailyMeasurement)
                    .values(
                        organization_id=organization_id, day=day.isoformat(), measurements=measured
                    )
                    .on_conflict_do_update(
                        constraint="uq_measurement_org_day", set_={"measurements": measured}
                    )
                )


class WeeklyGaps:
    def __init__(self, container: Container) -> None:
        self._container = container

    async def run(self) -> None:
        settings = self._container.settings.analytics
        until = datetime.now(UTC)
        since = until - timedelta(days=settings.gap_days)
        week = since.date().isoformat()
        async with self._container.database.session() as session:
            for organization_id in (await session.scalars(select(Organization.id))).all():
                gaps = await GapMiner(session, self._container.embedder, settings).mine(
                    organization_id, since, until
                )
                await session.execute(
                    delete(GapGroup).where(
                        GapGroup.organization_id == organization_id, GapGroup.week == week
                    )
                )
                session.add_all(
                    GapGroup(
                        organization_id=organization_id,
                        week=week,
                        size=gap.size,
                        customers=gap.customers,
                        examples=gap.examples,
                        cohesion=gap.cohesion,
                    )
                    for gap in gaps
                )
                if gaps:
                    logger.info("{} gap groups for {} this week", len(gaps), organization_id)


class EvaluationRetention:
    def __init__(self, container: Container) -> None:
        self._container = container

    async def run(self) -> None:
        now = datetime.now(UTC)
        evaluation, analytics = (
            self._container.settings.evaluation,
            self._container.settings.analytics,
        )
        async with self._container.database.session() as session:
            runs_before = now - timedelta(days=evaluation.run_days)
            await session.execute(delete(ScenarioRun).where(ScenarioRun.created_at < runs_before))
            await session.execute(
                delete(Call).where(
                    Call.channel == CallChannel.REPLAY, Call.created_at < runs_before
                )
            )
            measured_before = (now - timedelta(days=analytics.measurement_days)).date().isoformat()
            await session.execute(
                delete(DailyMeasurement).where(DailyMeasurement.day < measured_before)
            )
            gaps_before = (now - timedelta(days=analytics.gap_group_days)).date().isoformat()
            await session.execute(delete(GapGroup).where(GapGroup.week < gaps_before))
