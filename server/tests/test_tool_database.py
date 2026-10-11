from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.agent.tools.contracts import ToolContext
from app.agent.tools.implementations.work import AddWorkNote, GetWorkItem
from app.agent.tools.outbox import OutboxJobHandler, OutboxService
from app.core.enums import IdentityLevel, JobStatus, LatencyClass, ToolStatus
from app.core.errors import Invalid
from app.modules.accounts.models import User
from app.modules.calls.journal import JournalDraft, ToolJournalWriter
from app.modules.calls.service import ToolHealthService
from app.modules.config.schemas import BindingWrite
from app.modules.config.service import ConfigVersionService, ToolBindingService
from app.modules.jobs.models import Job
from app.modules.jobs.runner import JobHandler, JobRunner
from app.modules.work.models import ActionOutbox, Task


async def context_for(database, call) -> ToolContext:
    async with database.session() as session:
        user = await session.get(User, call.user_id)
    return ToolContext(
        organization_id=call.organization_id,
        call_id=call.id,
        caller_id=user.id,
        identity_level=IdentityLevel.VERIFIED,
        turn_index=1,
        timezone="Europe/London",
    )


class TestImplementations:
    async def test_work_item_is_found_only_for_its_own_tenant(self, database, tenants) -> None:
        call_a = await tenants.create("Tenant A", "Boiler for A")
        await tenants.create("Tenant B", "Fridge for B")
        outcome = await GetWorkItem(database).execute(
            await context_for(database, call_a), {"reference": "KQ7M3A"}
        )
        assert "Boiler for A" in outcome.speech
        assert "Fridge" not in outcome.speech

    async def test_a_reference_from_a_lookup_meets_the_note_precondition(
        self, database, tenants
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        ctx = await context_for(database, call)
        note = AddWorkNote(database)
        assert note.missing_precondition(ctx, {"reference": "KQ7M3A"}) is not None
        await GetWorkItem(database).execute(ctx, {"reference": "KQ7M3A"})
        assert note.missing_precondition(ctx, {"reference": "kq7m3a"}) is None

    async def test_a_reference_from_the_context_pack_is_known(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        ctx = await context_for(database, call)
        ctx.known_references = frozenset({"KQ7M3A"})
        assert AddWorkNote(database).missing_precondition(ctx, {"reference": "KQ7M3A"}) is None
        assert AddWorkNote(database).missing_precondition(ctx, {"reference": "AAAAAA"}) is not None

    async def test_adding_a_note_twice_adds_it_once(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        ctx = await context_for(database, call)
        ctx.receipt_reference = "NOTE01"
        arguments = {"reference": "KQ7M3A", "note": "The leak is worse"}
        await AddWorkNote(database).execute(ctx, arguments)
        await AddWorkNote(database).execute(ctx, arguments)
        async with database.session() as session:
            task = await session.scalar(
                select(Task).where(Task.organization_id == call.organization_id)
            )
        assert task.details.count("[note NOTE01]") == 1


class TestOutboxAndJobs:
    async def test_the_same_write_is_stored_once(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        ctx = await context_for(database, call)
        outbox = OutboxService(database)
        args = {"topic": "boiler", "reason": "It is leaking"}
        first = await outbox.enqueue(
            ctx, "request_callback", "request_callback", args, "key-1", "soon"
        )
        second = await outbox.enqueue(
            ctx, "request_callback", "request_callback", args, "key-1", "soon"
        )
        assert first.reference == second.reference and second.existing
        async with database.session() as session:
            rows = await session.scalar(
                select(func.count())
                .select_from(ActionOutbox)
                .where(ActionOutbox.idempotency_key == "key-1")
            )
            jobs = await session.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.organization_id == call.organization_id)
            )
        assert (rows, jobs) == (1, 1)

    async def test_the_job_runner_does_the_write(self, database, tenants, registry) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        ctx = await context_for(database, call)
        receipt = await OutboxService(database).enqueue(
            ctx,
            "request_callback",
            "request_callback",
            {"topic": "boiler", "reason": "It is leaking"},
            "key-2",
            "soon",
        )
        runner = JobRunner(database, [OutboxJobHandler(database, registry)])
        while await runner.run_once():
            pass
        async with database.session() as session:
            task = await session.scalar(select(Task).where(Task.reference == receipt.reference))
            row = await session.scalar(
                select(ActionOutbox).where(ActionOutbox.idempotency_key == "key-2")
            )
        assert (task.title, task.kind) == ("Callback about the boiler", "callback")
        assert row.status == JobStatus.DONE

    async def test_a_failing_job_is_retried_then_given_up(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")

        class Broken(JobHandler):
            kind = "test.broken"
            max_attempts = 2
            failed_for_good = []

            async def handle(self, job):
                raise RuntimeError("down")

            async def on_failed(self, job, error):
                self.failed_for_good.append(job.id)

        async with database.session() as session:
            from app.modules.jobs.service import JobQueue

            job = await JobQueue(session).enqueue(call.organization_id, "test.broken", {})
        runner = JobRunner(database, [Broken()])
        await runner.run_once()
        async with database.session() as session:
            stored = await session.get(Job, job.id)
            assert (stored.status, stored.attempts) == (JobStatus.PENDING, 1)
            stored.run_after = datetime.now(UTC) - timedelta(seconds=1)
        await runner.run_once()
        async with database.session() as session:
            assert (await session.get(Job, job.id)).status == JobStatus.FAILED
        assert Broken.failed_for_good == [job.id]


class TestBindingsAndHealth:
    def binding(self, **overrides) -> BindingWrite:
        values = dict(
            key="get_open_work",
            implementation="get_open_work",
            description="List the caller's open jobs when they ask about their work.",
            latency_class=LatencyClass.A,
            fallback_text="Not available now. Offer a callback.",
        )
        values.update(overrides)
        return BindingWrite(**values)

    async def test_a_schema_that_asks_for_identity_is_rejected(
        self, database, tenants, registry
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        bad = self.binding(
            parameters={"type": "object", "properties": {"customer_id": {"type": "string"}}}
        )
        with pytest.raises(Invalid):
            async with database.session() as session:
                await ToolBindingService(session, registry).save(call.organization_id, bad)

    async def test_a_write_tool_must_be_class_c(self, database, tenants, registry) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        bad = self.binding(key="request_callback", implementation="request_callback")
        with pytest.raises(Invalid):
            async with database.session() as session:
                await ToolBindingService(session, registry).save(call.organization_id, bad)

    async def test_saving_a_binding_bumps_the_config_version(
        self, database, tenants, registry
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        async with database.session() as session:
            before = await ConfigVersionService(session).current(call.organization_id)
            await ToolBindingService(session, registry).save(call.organization_id, self.binding())
            assert await ConfigVersionService(session).current(call.organization_id) == before + 1

    async def test_a_slow_class_a_tool_is_flagged(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        journal = ToolJournalWriter(database)
        for latency in (100, 120, 400, 450, 500):
            journal.record(
                JournalDraft(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    turn_index=1,
                    tool_key="get_open_work",
                    decision="permitted",
                    status=ToolStatus.OK,
                    latency_ms=latency,
                )
            )
        await journal.aclose()
        async with database.session() as session:
            [row] = await ToolHealthService(session).report(
                call.organization_id,
                datetime.now(UTC) - timedelta(hours=1),
                {"get_open_work": LatencyClass.A},
            )
        assert row.calls == 5 and row.class_violation
