from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.enums import Optimisation
from app.modules.calls.models import TurnRecord
from app.modules.calls.service import CacheReportService, LatencyReportService


def turn(call, index: int, first_audio: int, optimisations: list[str], hits=None) -> TurnRecord:
    return TurnRecord(
        organization_id=call.organization_id,
        call_id=call.id,
        turn_index=index,
        first_audio_ms=first_audio,
        optimisations=optimisations,
        cache_hits=hits or {},
    )


class TestLatencyReport:
    async def test_calls_with_a_switch_are_compared_with_calls_without_it(self, database, tenants):
        on = await tenants.create("A", "Leak")
        greeting = [Optimisation.PREPARED_GREETING.value]
        async with database.session() as session:
            session.add_all(
                [
                    turn(on, 0, 150, greeting),
                    turn(on, 1, 900, greeting),
                    turn(on, 2, 1100, greeting),
                ]
            )
            session.add_all([turn(on, 3, 1000, [])])
            await session.flush()
        since = datetime.now(UTC) - timedelta(hours=1)
        async with database.session() as session:
            comparison = await LatencyReportService(session).compare(
                on.organization_id, since, Optimisation.PREPARED_GREETING
            )
        stages = {s.stage: s for s in comparison.on.stages}
        assert (stages["greeting"].samples, stages["greeting"].p50) == (1, 150)
        assert (stages["first_audio"].samples, stages["first_audio"].p50) == (2, 1000)
        off_stages = {s.stage: s for s in comparison.off.stages}
        assert (off_stages["first_audio"].samples, off_stages["greeting"].samples) == (1, 0)


class TestCacheReport:
    async def test_hit_rates_by_key_type_come_from_the_turn_records(self, database, tenants):
        call = await tenants.create("A", "Leak")
        async with database.session() as session:
            session.add_all(
                [
                    turn(call, 1, 900, [], {"kb.store": 2, "kb.miss": 2, "config.memory": 1}),
                    turn(call, 2, 900, [], {"kb.memory": 4}),
                ]
            )
        since = datetime.now(UTC) - timedelta(hours=1)
        async with database.session() as session:
            rows = await CacheReportService(session).report(call.organization_id, since)
        by_kind = {row.kind: row for row in rows}
        assert (by_kind["kb"].memory, by_kind["kb"].store, by_kind["kb"].miss) == (4, 2, 2)
        assert by_kind["kb"].hit_rate == 0.75 and by_kind["config"].hit_rate == 1.0
