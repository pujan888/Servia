from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.core.config import DetectionSettings, WorkSettings
from app.core.enums import FlagType, Priority
from app.modules.calls.models import Call, Handoff, TurnRecord
from app.modules.calls.service import HandoffService
from app.modules.config.schemas import RouteWrite
from app.modules.config.service import AgentConfigService, DetectionSettingService, RouteService
from app.modules.detection.models import ReviewItem
from app.modules.detection.schemas import CallerBlockWrite
from app.modules.detection.service import CallerBlockService, PreAnswerGate
from app.modules.work.factory import TableRouter, WorkItemDraft, WorkItemFactory
from app.modules.work.models import Task, TaskEvidence
from app.modules.work.service import TaskService
from tests.fakes import FakeModel
from tests.test_memory_database import MemoryBuilder

LEAK = (
    "Washing machine still leaking at the door seal after the last visit; "
    "customer wants a return visit."
)
LEAK_AGAIN = "The door seal on the washing machine is leaking again. Please send an engineer back."
FRIDGE = "Customer wants someone to look at the fridge, which is running warm."


async def routes(database, organization_id) -> None:
    async with database.session() as session:
        await RouteService(session).replace(
            organization_id,
            [
                RouteWrite(work_item_type="repair", description="Engineer repairs.", queue="field"),
                RouteWrite(
                    work_item_type="complaint",
                    description="Service complaints.",
                    queue="care",
                    priority=Priority.HIGH,
                ),
            ],
        )


def factory(database, embedder, *summaries: str) -> WorkItemFactory:
    answers = [{"title": s[:40], "summary": s, "confidence": 0.9} for s in summaries]
    return WorkItemFactory(
        database,
        embedder,
        FakeModel(structured=answers),
        TableRouter(database),
        WorkSettings(),
        DetectionSettings(),
    )


def draft(call: Call, reference: str, kind: str = "repair", **signals) -> WorkItemDraft:
    return WorkItemDraft(
        organization_id=call.organization_id,
        user_id=call.user_id,
        call_id=call.id,
        reference=reference,
        kind=kind,
        description="from the call",
        signals={"evidence": f"said in {reference}", **signals},
    )


class TestWorkItems:
    async def test_two_calls_about_one_problem_make_one_item_with_two_evidence_rows(
        self, database, tenants, embedder
    ):
        call = await tenants.create("A", "Leak")
        await routes(database, call.organization_id)
        items = factory(database, embedder, LEAK, LEAK_AGAIN)
        first, created = await items.make_or_update(draft(call, "MC7KHP"))
        second, merged = await items.make_or_update(draft(call, "RT4VWX"))
        assert (created, merged) == (True, False) and second.id == first.id
        async with database.session() as session:
            evidence = await session.scalar(
                select(func.count()).where(TaskEvidence.task_id == first.id)
            )
            stored = await session.get(Task, first.id)
            found = await TaskService(session).get_by_reference(call.user_id, "RT4VWX")
        assert (evidence, stored.update_count, found.id) == (2, 1, first.id)
        assert (stored.kind, stored.queue) == ("repair", "field")

    async def test_a_different_problem_makes_a_new_item(self, database, tenants, embedder):
        call = await tenants.create("A", "Leak")
        await routes(database, call.organization_id)
        items = factory(database, embedder, LEAK, FRIDGE)
        first, _ = await items.make_or_update(draft(call, "MC7KHP"))
        second, created = await items.make_or_update(draft(call, "RT4VWX"))
        assert created and second.id != first.id

    async def test_a_rerun_of_the_same_request_changes_nothing(self, database, tenants, embedder):
        call = await tenants.create("A", "Leak")
        await routes(database, call.organization_id)
        items = factory(database, embedder, LEAK)
        first, _ = await items.make_or_update(draft(call, "MC7KHP"))
        again, created = await items.make_or_update(draft(call, "MC7KHP"))
        assert (again.id, created) == (first.id, False)

    async def test_priority_comes_from_the_call_and_the_route_never_the_model(
        self, database, tenants, embedder
    ):
        call = await tenants.create("A", "Leak")
        await routes(database, call.organization_id)
        items = factory(database, embedder, LEAK, "The engineer was rude on the last visit.")
        urgent, _ = await items.make_or_update(draft(call, "MC7KHP", flags=[FlagType.DISTRESS]))
        routed, _ = await items.make_or_update(draft(call, "RT4VWX", kind="complaint"))
        assert (urgent.priority, routed.priority, routed.queue) == (
            Priority.URGENT,
            Priority.HIGH,
            "care",
        )


class TestPreAnswerGate:
    async def test_a_blocked_caller_is_refused_and_an_expired_block_is_not(
        self, database, tenants, cache
    ):
        call = await tenants.create("A", "Leak")
        gate = PreAnswerGate(database, cache)
        assert not await gate.blocked(call.organization_id, ["+447700900999"])
        async with database.session() as session:
            service = CallerBlockService(session, cache)
            await service.add(
                call.organization_id, CallerBlockWrite(identifier="+447700900999", reason="abuse")
            )
            await service.add(
                call.organization_id,
                CallerBlockWrite(
                    identifier="+447700900111",
                    reason="old",
                    expires_at=datetime.now(UTC) - timedelta(days=1),
                ),
            )
        assert await gate.blocked(call.organization_id, [None, "+447700900999"])
        assert not await gate.blocked(call.organization_id, ["+447700900111"])


class TestReviewAndHandoff:
    async def test_danger_flags_guard_actions_and_failed_handoffs_are_reviewed_once(
        self, database, tenants, registry, cache
    ):
        builder = MemoryBuilder(database, tenants, registry, cache)
        first = await builder.tenant()
        call = await builder.finished_call(first)
        async with database.session() as session:
            turn = await session.scalar(
                select(TurnRecord).where(TurnRecord.call_id == call.id, TurnRecord.turn_index == 1)
            )
            turn.flags = [
                {"type": "distress", "severity": "now", "score": 0.9, "evidence": "", "model": "t"},
                {"type": "guard", "severity": "info", "score": 1.0, "evidence": "", "model": "g"},
                {
                    "type": "off_topic",
                    "severity": "next_turn",
                    "score": 1,
                    "evidence": "",
                    "model": "t",
                },
            ]
            await HandoffService(session).record(
                Handoff(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    reason="human_request",
                    briefing="b",
                    level=3,
                    result="failed",
                )
            )
        await builder.run(call)
        await builder.run(call, builder.pipeline())
        async with database.session() as session:
            reasons = sorted(
                await session.scalars(
                    select(ReviewItem.reason).where(ReviewItem.call_id == call.id)
                )
            )
            stored = await session.get(Call, call.id)
        assert reasons == ["distress", "guard", "handoff_failed"]
        assert stored.escalation_reason == "human_request" and stored.outcome == "callback_promised"

    async def test_a_completed_transfer_makes_the_outcome_escalated(
        self, database, tenants, registry, cache
    ):
        builder = MemoryBuilder(database, tenants, registry, cache)
        call = await builder.finished_call(await builder.tenant())
        async with database.session() as session:
            await HandoffService(session).record(
                Handoff(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    reason="human_request",
                    briefing="b",
                    level=0,
                    result="transferred",
                )
            )
        await builder.run(call)
        async with database.session() as session:
            assert (await session.get(Call, call.id)).outcome == "escalated"


class TestTenantSettings:
    async def test_routes_and_thresholds_reach_the_agent_configuration(self, database, tenants):
        call = await tenants.create("A", "Leak")
        await routes(database, call.organization_id)
        async with database.session() as session:
            await RouteService(session).replace(
                call.organization_id,
                [
                    RouteWrite(
                        work_item_type="handoff", description="Live transfers.", queue="desk"
                    ),
                    RouteWrite(
                        work_item_type="repair", description="Engineer repairs.", queue="field"
                    ),
                ],
            )
            await DetectionSettingService(session).save(
                call.organization_id, FlagType.ABUSE, 0.9, enabled=True
            )
            await DetectionSettingService(session).save(
                call.organization_id, FlagType.OFF_TOPIC, 1.0, enabled=False
            )
            config = await AgentConfigService(session).load(call.organization_id)
        assert config.work_item_types == frozenset({"repair"})
        assert config.route("handoff").queue == "desk"
        assert config.detection.thresholds[FlagType.ABUSE] == 0.9
        assert not config.detection.fires(FlagType.OFF_TOPIC, 1.0)
        assert config.detection.fires(FlagType.DISTRESS, 0.7)
