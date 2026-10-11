from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select

from app.analysis.outcomes.analytics import OutcomeAnalytics
from app.core.container import Container
from app.core.enums import ReleaseStatus
from app.modules.releases.models import AgentRelease
from app.modules.releases.service import ReleaseCatalog, ReleasePromoter


class CanaryMonitor:
    """After enough calls or enough time, a canary goes to every call if no
    measurement is worse than the stable release by more than its tolerance.
    Otherwise it is rolled back, and an alert is logged."""

    def __init__(self, container: Container) -> None:
        self._container = container
        self._settings = container.settings.releases
        self._catalog = ReleaseCatalog(container.database, container.cache, container.settings)

    async def run(self) -> None:
        async with self._container.database.session() as session:
            canaries = (
                await session.scalars(
                    select(AgentRelease).where(
                        AgentRelease.status == ReleaseStatus.LIVE, AgentRelease.canary_percent < 100
                    )
                )
            ).all()
        for canary in canaries:
            await self._judge(canary)

    async def _judge(self, canary: AgentRelease) -> None:
        now = datetime.now(UTC)
        settings, container = self._settings, self._container
        async with container.database.session() as session:
            stable = await session.scalar(
                select(AgentRelease).where(
                    AgentRelease.organization_id == canary.organization_id,
                    AgentRelease.status == ReleaseStatus.LIVE,
                    AgentRelease.canary_percent == 100,
                )
            )
            analytics = OutcomeAnalytics(session, container.settings.analytics)
            new = await analytics.measure(canary.organization_id, canary.live_at, now, canary.id)
            old = await analytics.measure(canary.organization_id, canary.live_at, now, stable.id)
            aged = now - canary.live_at >= timedelta(hours=settings.canary_hours)
            if new["calls"] < settings.canary_calls and not aged:
                return
            worse = self.worse(new, old)
            release = await session.get(AgentRelease, canary.id)
            promoter = ReleasePromoter(session, container.registry, container.settings)
            if worse:
                await promoter.rollback(release)
            else:
                await promoter.complete(release)
        await self._catalog.resolver.invalidate(canary.organization_id)
        if worse:
            logger.warning("canary v{} rolled back: {}", canary.version, worse)
        else:
            logger.info("canary v{} now serves every call", canary.version)

    def worse(self, new: dict[str, Any], old: dict[str, Any]) -> list[str]:
        s, findings = self._settings, []
        for key, tolerance, higher_is_worse in (
            ("escalation_rate", s.tolerance_escalation, True),
            ("refusal_rate", s.tolerance_refusal, True),
            ("task_completion", s.tolerance_completion, False),
        ):
            if new[key] is None or old[key] is None:
                continue
            change = new[key] - old[key] if higher_is_worse else old[key] - new[key]
            if change > tolerance:
                findings.append(f"{key}: {old[key]} → {new[key]}")
        if new["first_audio_p95_ms"] and old["first_audio_p95_ms"]:
            rise = new["first_audio_p95_ms"] / old["first_audio_p95_ms"] - 1
            if rise > s.tolerance_p95_ms:
                findings.append(
                    f"first audio p95: {old['first_audio_p95_ms']} → {new['first_audio_p95_ms']} ms"
                )
        return findings
