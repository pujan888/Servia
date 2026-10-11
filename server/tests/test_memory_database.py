from __future__ import annotations

import statistics
import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.agent.tools.contracts import ToolContext
from app.agent.tools.implementations.memory import GetCallHistory
from app.analysis.outcomes.classifier import OutcomeClassifier
from app.analysis.postcall.digest import DigestBuilder
from app.analysis.postcall.pipeline import PostCallPipeline
from app.analysis.postcall.slots import SlotExtractor
from app.analysis.postcall.summariser import Summariser
from app.analysis.postcall.transcript import TranscriptBuilder
from app.core.config import ContextSettings, MemorySettings, SettingsFactory
from app.core.enums import IdentityLevel, SlotSource, ToolStatus
from app.core.tokens import TokenCounter
from app.modules.accounts.models import Organization
from app.modules.calls.models import Call, ToolJournalEntry, TranscriptLine, TurnRecord
from app.modules.calls.service import CallService
from app.modules.calls.utils import RoomName
from app.modules.config.schemas import SlotKeyWrite
from app.modules.config.service import AgentConfigService, SlotKeyService
from app.modules.context.assembler import ContextAssembler
from app.modules.jobs.models import Job
from app.modules.memory.models import CallSummary, MemorySlot
from app.modules.memory.schemas import SlotCandidate
from app.modules.memory.service import ErasureService, ForgetPolicy, TranscriptRetention
from app.modules.memory.store import MemoryStore
from app.modules.releases.service import ReleaseCatalog
from app.modules.work.models import ActionOutbox
from tests.fakes import FakeModel

SUMMARY = "Sam asked for a visit on Thursday morning, reference RT4VWX. A colleague will confirm."
SLOTS = {
    "slots": [
        {
            "key": "times_to_avoid",
            "value": "afternoons",
            "source": "stated",
            "confidence": 0.9,
            "line": 2,
        }
    ]
}


class MemoryBuilder:
    """A tenant with slot keys, and finished calls ready for the pipeline."""

    def __init__(self, database, tenants, registry, cache) -> None:
        self.database = database
        self.tenants = tenants
        self.registry = registry
        self.cache = cache

    async def tenant(self) -> Call:
        call = await self.tenants.create(f"memory-{uuid.uuid4().hex[:6]}", "Fridge running warm")
        async with self.database.session() as session:
            await SlotKeyService(session).save(
                call.organization_id,
                SlotKeyWrite(
                    key="times_to_avoid",
                    description="Times that do not suit the caller, as a rule.",
                    example="afternoons",
                ),
            )
        return call

    async def finished_call(self, first: Call) -> Call:
        """A call like the scenario's first call: a booking, and a preference."""
        async with self.database.session() as session:
            call = Call(
                organization_id=first.organization_id,
                user_id=first.user_id,
                room_name=RoomName.generate(),
                started_at=datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
            )
            session.add(call)
            await session.flush()
            session.add_all(
                [
                    self._turn(call, 0, None, "Hello Sam."),
                    self._turn(
                        call,
                        1,
                        "Can someone come Thursday morning? Never afternoons.",
                        "Logged, reference R T 4 V W X.",
                    ),
                    self._turn(call, 2, "Thanks, bye.", "Goodbye."),
                    ToolJournalEntry(
                        organization_id=call.organization_id,
                        call_id=call.id,
                        turn_index=1,
                        tool_key="request_callback",
                        arguments={"preferred_time": "Thursday morning"},
                        decision="permitted",
                        status="accepted",
                        idempotency_key=f"idem-{call.id}",
                    ),
                    ActionOutbox(
                        organization_id=call.organization_id,
                        idempotency_key=f"idem-{call.id}",
                        reference="RT4VWX",
                        call_id=call.id,
                        caller_id=call.user_id,
                        identity_level=3,
                        timezone="Europe/London",
                        tool_key="request_callback",
                        implementation="request_callback",
                        arguments={},
                    ),
                ]
            )
            await session.flush()
            await CallService(session).mark_ended(call.id)
        return call

    async def job(self, call: Call) -> Job:
        async with self.database.session() as session:
            return await session.scalar(
                select(Job).where(Job.payload["call_id"].astext == str(call.id))
            )

    def pipeline(self, summaries=(SUMMARY,), slots=(SLOTS,), outcomes=()) -> PostCallPipeline:
        self.summary_model = FakeModel(texts=summaries)
        return PostCallPipeline(
            self.database,
            ReleaseCatalog(self.database, self.cache, SettingsFactory.load()),
            TranscriptBuilder(),
            Summariser(self.summary_model, 3),
            SlotExtractor(FakeModel(structured=slots), 0.6),
            OutcomeClassifier(FakeModel(structured=outcomes), self.registry),
            DigestBuilder(MemorySettings(), TokenCounter()),
        )

    async def run(self, call: Call, pipeline: PostCallPipeline | None = None) -> dict:
        return await (pipeline or self.pipeline()).run(await self.job(call))

    async def next_call_pack(self, first: Call):
        async with self.database.session() as session:
            call = Call(
                organization_id=first.organization_id,
                user_id=first.user_id,
                room_name=RoomName.generate(),
            )
            session.add(call)
            await session.flush()
        assembler = ContextAssembler(self.database, ContextSettings(deadline_ms=2000))
        return await assembler.assemble(call.id, IdentityLevel.VERIFIED)

    async def count(self, model, call: Call) -> int:
        async with self.database.session() as session:
            return await session.scalar(
                select(func.count()).select_from(model).where(model.user_id == call.user_id)
            )

    @staticmethod
    def _turn(call: Call, index: int, heard: str | None, said: str) -> TurnRecord:
        return TurnRecord(
            organization_id=call.organization_id,
            call_id=call.id,
            turn_index=index,
            transcript_in=heard,
            text_spoken=said,
            active_skill="book_visit",
        )


@pytest.fixture
def memory(database, tenants, registry, cache):
    return MemoryBuilder(database, tenants, registry, cache)


class TestPostCallPipeline:
    async def test_the_pipeline_prepares_memory_for_the_next_call(self, memory, database):
        first = await memory.tenant()
        call = await memory.finished_call(first)
        progress = await memory.run(call)

        assert set(progress) == {"transcript", "summary", "slots", "outcome", "review", "digest"}
        assert progress["transcript"]["lines"] == 6
        async with database.session() as session:
            stored = await session.get(Call, call.id)
            store = MemoryStore(session)
            [slot] = await store.current_slots(call.organization_id, call.user_id)
        assert (stored.outcome, stored.analysed_at is not None) == ("callback_promised", True)
        assert (slot.value, slot.source, slot.evidence_turn) == ("afternoons", "stated", 1)

        pack = await memory.next_call_pack(first)
        assert pack.digest.summary == (
            "Sam asked for a visit on Thursday 1 October morning, reference RT4VWX. "
            "A colleague will confirm."
        )
        assert [(s.label, s.value) for s in pack.digest.slots] == [("Times to avoid", "afternoons")]
        assert "memory" not in pack.degraded

    async def test_a_failed_step_runs_again_without_repeating_finished_steps(self, memory):
        call = await memory.finished_call(await memory.tenant())
        pipeline = memory.pipeline(slots=(RuntimeError("provider down"), SLOTS))
        with pytest.raises(RuntimeError):
            await memory.run(call, pipeline)
        progress = (await memory.job(call)).progress
        assert set(progress) == {"transcript", "summary", "outcome", "review"}

        await memory.run(call, pipeline)
        assert len(memory.summary_model.prompts) == 1
        assert await memory.count(CallSummary, call) == 1
        assert await memory.count(MemorySlot, call) == 1

    async def test_a_call_not_yet_in_memory_marks_memory_degraded(self, memory):
        first = await memory.tenant()
        await memory.finished_call(first)
        pack = await memory.next_call_pack(first)
        assert "memory" in pack.degraded


class TestMemoryStore:
    async def test_precedence_keeps_one_current_value_and_the_history(self, memory, database):
        call = await memory.tenant()
        writes = [
            ("stated", "mornings"),
            ("inferred", "evenings"),
            ("stated", "Mornings"),
            ("stated", "evenings"),
        ]
        decisions = []
        for source, value in writes:
            async with database.session() as session:
                [change] = await MemoryStore(session).write_slots(
                    call.organization_id,
                    call.user_id,
                    call.id,
                    [SlotCandidate(key="preferred", value=value, source=source, confidence=0.8)],
                )
                decisions.append(change.decision)
        assert decisions == ["insert", "keep", "confirm", "replace"]
        async with database.session() as session:
            [current] = await MemoryStore(session).current_slots(call.organization_id, call.user_id)
        assert (current.value, current.source) == ("evenings", SlotSource.STATED)
        assert await memory.count(MemorySlot, call) == 2

    async def test_memory_of_one_tenant_is_not_read_by_another(self, memory, database):
        first = await memory.tenant()
        await memory.run(await memory.finished_call(first))
        other = await memory.tenant()
        async with database.session() as session:
            store = MemoryStore(session)
            assert await store.digest(other.organization_id, first.user_id) is None
            assert await store.current_slots(other.organization_id, first.user_id) == []

    async def test_the_digest_read_at_call_setup_takes_5_ms_or_less(self, memory, database):
        """Measured as the assembler reads it: a session of its own, one statement.
        The first reads prepare the statement and are not counted. The median is
        checked: on a busy machine the slowest reads are set by the operating
        system, not by the query."""
        first = await memory.tenant()
        await memory.run(await memory.finished_call(first))

        async def read() -> float:
            started = time.perf_counter()
            async with database.session() as session:
                await MemoryStore(session).load(
                    first.organization_id, first.user_id, first.id, timedelta(minutes=10)
                )
            return (time.perf_counter() - started) * 1000

        for _ in range(10):
            await read()
        timings = [await read() for _ in range(200)]
        assert statistics.median(timings) <= 5

    async def test_the_history_tool_gives_earlier_calls_only(self, memory, database):
        first = await memory.tenant()
        await memory.run(await memory.finished_call(first))
        tool = GetCallHistory(database, limit=3)
        context = ToolContext(
            organization_id=first.organization_id,
            call_id=first.id,
            caller_id=first.user_id,
            identity_level=IdentityLevel.VERIFIED,
            turn_index=1,
            timezone="Europe/London",
        )
        outcome = await tool.execute(context, {})
        assert outcome.status == ToolStatus.OK and "RT4VWX" in outcome.speech


class TestErasure:
    async def test_erasure_removes_all_memory_of_the_customer(self, memory, database):
        first = await memory.tenant()
        call = await memory.finished_call(first)
        await memory.run(call)
        async with database.session() as session:
            erasure = await ErasureService(session).erase(
                first.organization_id, first.user_id, "privacy@northwind.test"
            )
        assert (erasure.counts["slots"], erasure.counts["summaries"]) == (1, 1)

        async with database.session() as session:
            store = MemoryStore(session)
            assert await store.digest(first.organization_id, first.user_id) is None
            assert await store.current_slots(first.organization_id, first.user_id) == []
            assert await store.summaries(first.organization_id, first.user_id, 10) == []
            texts = await session.scalars(
                select(TranscriptLine.text).where(TranscriptLine.call_id == call.id)
            )
            turns = await session.scalars(
                select(TurnRecord.transcript_in).where(
                    TurnRecord.call_id == call.id, TurnRecord.transcript_in.is_not(None)
                )
            )
            assert set(texts) == {"[erased]"} and set(turns) == {"[erased]"}
        pack = await memory.next_call_pack(first)
        assert pack.digest is None and "memory" not in pack.degraded

    async def test_a_call_from_before_the_erasure_writes_no_memory(self, memory, database):
        first = await memory.tenant()
        call = await memory.finished_call(first)
        async with database.session() as session:
            await ErasureService(session).erase(first.organization_id, first.user_id, "test")
        progress = await memory.run(call)
        assert progress["summary"]["skipped"] == "erased"
        assert await memory.count(CallSummary, call) == 0
        assert await memory.count(MemorySlot, call) == 0


class TestRetention:
    async def test_the_forget_rule_removes_only_old_uncertain_guesses(self, memory, database):
        call = await memory.tenant()
        old = datetime.now(UTC) - timedelta(days=200)
        rows = {
            "old_weak_guess": ("inferred", 0.65, old),
            "old_strong_guess": ("inferred", 0.8, old),
            "old_statement": ("stated", 0.6, old),
            "new_weak_guess": ("inferred", 0.65, datetime.now(UTC)),
        }
        async with database.session() as session:
            session.add_all(
                MemorySlot(
                    organization_id=call.organization_id,
                    user_id=call.user_id,
                    key=key,
                    value="x",
                    source=source,
                    confidence=confidence,
                    valid_from=valid_from,
                )
                for key, (source, confidence, valid_from) in rows.items()
            )
        await ForgetPolicy(database, MemorySettings()).run()
        async with database.session() as session:
            kept = await MemoryStore(session).current_slots(call.organization_id, call.user_id)
        assert [slot.key for slot in kept] == [
            "new_weak_guess",
            "old_statement",
            "old_strong_guess",
        ]

    async def test_transcripts_are_cleared_after_the_tenant_retention(self, memory, database):
        first = await memory.tenant()
        call = await memory.finished_call(first)
        await memory.run(call)
        async with database.session() as session:
            (await session.get(Organization, first.organization_id)).transcript_days = 30
            (await session.get(Call, call.id)).ended_at = datetime.now(UTC) - timedelta(days=31)
        await TranscriptRetention(database).run()
        async with database.session() as session:
            lines = await session.scalar(
                select(func.count()).where(TranscriptLine.call_id == call.id)
            )
            spoken = await session.scalar(
                select(func.count()).where(
                    TurnRecord.call_id == call.id, TurnRecord.text_spoken.is_not(None)
                )
            )
        assert (lines, spoken) == (0, 0)


class TestSlotKeys:
    async def test_a_saved_slot_key_reaches_the_agent_configuration(self, memory, database):
        call = await memory.tenant()
        async with database.session() as session:
            config = await AgentConfigService(session).load(call.organization_id)
        assert [key.key for key in config.slot_keys] == ["times_to_avoid"]
        assert config.slot_key("times_to_avoid").label == "Times to avoid"
