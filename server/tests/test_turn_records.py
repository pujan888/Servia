from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.modules.calls.recorder import TurnDraft, TurnRecorder
from app.modules.calls.service import LatencyReportService, TurnRecordService


class TestTurnRecords:
    async def test_recorder_writes_each_turn(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        recorder = TurnRecorder(database, call.organization_id, call.id)
        for index in (1, 2, 3):
            recorder.record(TurnDraft(turn_index=index, first_audio_ms=700 + index * 10))
        await recorder.aclose()

        async with database.session() as session:
            rows = await TurnRecordService(session).list_for_call(call.organization_id, call.id)
        assert [row.turn_index for row in rows] == [1, 2, 3]

    async def test_latency_report_gives_percentiles_per_stage(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        recorder = TurnRecorder(database, call.organization_id, call.id)
        for index in range(1, 21):
            recorder.record(
                TurnDraft(
                    turn_index=index,
                    first_audio_ms=index * 100,
                    prompt_tokens=1000,
                    cached_tokens=800,
                )
            )
        await recorder.aclose()

        async with database.session() as session:
            report = await LatencyReportService(session).report(
                call.organization_id, datetime.now(UTC) - timedelta(hours=1)
            )
        first_audio = next(stage for stage in report.stages if stage.stage == "first_audio")
        assert first_audio.samples == 20
        assert first_audio.p50 == 1050
        assert report.cache_hit_rate == 0.8
