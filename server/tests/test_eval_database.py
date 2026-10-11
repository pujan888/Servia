from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
from sqlalchemy import func, select

from app.analysis.outcomes.analytics import OutcomeAnalytics
from app.analysis.outcomes.gaps import GapMiner
from app.core.config import AnalyticsSettings, SettingsFactory
from app.core.enums import (
    CallChannel,
    CallOutcome,
    CallStatus,
    GateDecision,
    LatencyClass,
    ReleaseStatus,
)
from app.eval.assertions import Observation, ToolCall, TurnSeen
from app.eval.recorder import ScenarioRecorder
from app.eval.suite import SuiteRunner
from app.modules.calls.models import Call, CallContext, Handoff, ToolJournalEntry, TurnRecord
from app.modules.calls.utils import RoomName
from app.modules.config.models import ToolBinding
from app.modules.config.schemas import SkillWrite
from app.modules.config.service import SkillService
from app.modules.evaluation.models import AssertionResult, Scenario, SuiteResult
from app.modules.releases.service import ReleaseBuilder, ReleaseCatalog
from app.modules.work.models import ActionOutbox

SETTINGS = SettingsFactory.load()


async def bind(database, organization_id, key: str, latency=LatencyClass.A) -> None:
    async with database.session() as session:
        session.add(
            ToolBinding(
                organization_id=organization_id,
                key=key,
                implementation=key,
                description=f"The {key} tool, for tests.",
                parameters={"type": "object", "properties": {}},
                latency_class=latency,
                fallback_text="Not available now.",
            )
        )


class TestScenarioRecorder:
    async def test_a_real_call_becomes_a_disabled_scenario_with_its_answers(
        self, database, tenants, registry, cache
    ):
        call = await tenants.create("A", "Leak")
        await bind(database, call.organization_id, "request_callback", LatencyClass.C)
        await bind(database, call.organization_id, "search_knowledge")
        async with database.session() as session:
            session.add(
                CallContext(
                    call_id=call.id,
                    organization_id=call.organization_id,
                    pack={"caller": {"id": str(call.user_id), "full_name": "Sam Taylor"}},
                    prefix="p",
                    prefix_hash="h",
                    sections=[],
                )
            )
            rows = [
                TurnRecord(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    turn_index=1,
                    transcript_in="Do you sell TVs?",
                    skill_loaded=None,
                ),
                TurnRecord(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    turn_index=2,
                    transcript_in="Please call me back",
                    skill_loaded="general_inquiry",
                ),
                ToolJournalEntry(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    turn_index=1,
                    tool_key="search_knowledge",
                    arguments={"query": "tvs"},
                    decision="permitted",
                    status="empty",
                ),
                ToolJournalEntry(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    turn_index=2,
                    tool_key="request_callback",
                    arguments={"reason": "tv"},
                    decision="permitted",
                    status="accepted",
                    result_text="logged",
                    idempotency_key=f"k-{call.id}",
                ),
                ActionOutbox(
                    organization_id=call.organization_id,
                    idempotency_key=f"k-{call.id}",
                    reference="RT4VWX",
                    call_id=call.id,
                    caller_id=call.user_id,
                    identity_level=3,
                    timezone="UTC",
                    tool_key="request_callback",
                    implementation="request_callback",
                    arguments={},
                ),
                Handoff(
                    organization_id=call.organization_id,
                    call_id=call.id,
                    reason="human_request",
                    briefing="b",
                    level=1,
                    result="callback",
                ),
            ]
            session.add_all(rows)
        catalog = ReleaseCatalog(database, cache, SETTINGS)
        async with database.session() as session:
            scenario = await ScenarioRecorder(session, registry, catalog).record(call.id)
        assert scenario.enabled is False
        assert scenario.caller_turns == ["Do you sell TVs?", "Please call me back"]
        assert scenario.responses == [
            {
                "tool_key": "request_callback",
                "arguments": {"reason": "tv"},
                "status": "accepted",
                "text": "logged",
                "reference": "RT4VWX",
            }
        ]
        kinds = [a["type"] for a in scenario.assertions]
        assert kinds == [
            "no_invented_price",
            "refused",
            "tool_called",
            "reference_said",
            "skill_loaded",
            "escalated",
        ]


class ScriptedReplay:
    """Stands in for the replay: each run returns the next observation."""

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)

    async def run(self, scenario, config, release_id) -> Observation:
        return self.observations.pop(0)


def observation(called: bool) -> Observation:
    tools = [ToolCall(1, "request_callback", {}, "accepted", "logged", "RT4VWX")] if called else []
    return Observation(
        turns=[
            TurnSeen(1, "call me back", "Your reference is R, T, four, V, W, X.", None, 900, False)
        ],
        tools=tools,
        escalation=None,
        unexpected=[],
        known_text="",
    )


class TestSuite:
    async def suite(self, database, cache, registry, organization_id, release_id, observations):
        container = SimpleNamespace(
            database=database, cache=cache, settings=SETTINGS, registry=registry, embedder=None
        )
        runner = SuiteRunner(container)
        runner._replay = ScriptedReplay(observations)
        async with database.session() as session:
            suite = SuiteResult(organization_id=organization_id, release_id=release_id)
            session.add(suite)
        return await runner.run(suite.id)

    async def test_two_of_three_runs_pass_and_a_regression_is_blocked(
        self, database, cache, registry, tenants
    ):
        call = await tenants.create("A", "Leak")
        async with database.session() as session:
            release = await ReleaseBuilder(session, SETTINGS).build(call.organization_id, "test")
            session.add(
                Scenario(
                    organization_id=call.organization_id,
                    name="callback",
                    pack={"caller": {"id": str(call.user_id)}},
                    caller_turns=["call me back"],
                    responses=[],
                    assertions=[{"type": "tool_called", "tool": "request_callback"}],
                    enabled=True,
                )
            )
        good = await self.suite(
            database,
            cache,
            registry,
            call.organization_id,
            release.id,
            [observation(True), observation(False), observation(True)],
        )
        assert good.measurements["pass_rate"] == 1.0 and good.decision == GateDecision.PASS
        async with database.session() as session:
            failed = await session.scalar(
                select(func.count()).where(
                    AssertionResult.organization_id == call.organization_id,
                    AssertionResult.passed.is_(False),
                )
            )
            assert failed == 1
            live = await session.get(type(release), release.id)
            live.status, live.canary_percent = ReleaseStatus.LIVE, 100
        async with database.session() as session:
            await SkillService(session).create(
                call.organization_id,
                SkillWrite(
                    key="broken",
                    name="Broken",
                    trigger="When anything at all happens on any call",
                    goal="g",
                    steps="1.",
                ),
            )
            broken = await ReleaseBuilder(session, SETTINGS).build(call.organization_id, "test")
        bad = await self.suite(
            database,
            cache,
            registry,
            call.organization_id,
            broken.id,
            [observation(False), observation(False), observation(True)],
        )
        assert bad.decision == GateDecision.BLOCK and bad.baseline_id == good.id


class TestAnalytics:
    async def test_the_measurements_match_a_hand_count(self, database, tenants):
        call = await tenants.create("A", "Leak")
        now = datetime.now(UTC)
        async with database.session() as session:
            rows = []
            for outcome in (
                CallOutcome.RESOLVED,
                CallOutcome.RESOLVED,
                CallOutcome.ESCALATED,
                CallOutcome.CALLBACK_PROMISED,
            ):
                rows.append(
                    Call(
                        organization_id=call.organization_id,
                        user_id=call.user_id,
                        room_name=RoomName.generate(),
                        status=CallStatus.COMPLETED,
                        outcome=outcome,
                        started_at=now - timedelta(minutes=5),
                        ended_at=now,
                    )
                )
            rows.append(
                Call(
                    organization_id=call.organization_id,
                    user_id=call.user_id,
                    room_name=RoomName.generate(),
                    status=CallStatus.COMPLETED,
                    outcome=CallOutcome.UNRESOLVED,
                    channel=CallChannel.REPLAY,
                )
            )
            session.add_all(rows)
            await session.flush()
            session.add(
                TurnRecord(
                    organization_id=call.organization_id,
                    call_id=rows[0].id,
                    turn_index=1,
                    first_audio_ms=900,
                    prompt_tokens=1000,
                    cached_tokens=800,
                    retrieval=[{"route": "none"}, {"route": "card"}],
                    active_skill="general_inquiry",
                    text_spoken="abcd",
                )
            )
            session.add(
                Handoff(
                    organization_id=call.organization_id,
                    call_id=rows[2].id,
                    reason="human_request",
                    briefing="b",
                    level=0,
                    result="transferred",
                )
            )
        async with database.session() as session:
            measured = await OutcomeAnalytics(session, AnalyticsSettings()).measure(
                call.organization_id, now - timedelta(hours=1), now + timedelta(hours=1)
            )
        assert measured["calls"] == 4
        assert (
            measured["containment"],
            measured["task_completion"],
            measured["escalation_rate"],
        ) == (0.5, 0.75, 0.25)
        assert measured["escalations_by_reason"] == {"human_request": 1}
        assert (measured["refusal_rate"], measured["refusals_by_skill"]) == (
            0.5,
            {"general_inquiry": 1},
        )
        assert (
            measured["cost"]["synthesis_characters"] == 4
            and measured["cost"]["speech_minutes"] == 20.0
        )


class TestGapMining:
    async def test_ten_plain_paraphrases_make_one_group(self, embedder):
        same = [
            "Do you sell televisions?",
            "Can I buy a TV from you?",
            "Are TVs something you stock?",
            "Do you have any televisions for sale?",
            "I'm looking to buy a new television, do you sell them?",
            "Is it possible to get a TV from your shop?",
            "Do you guys do TVs?",
            "Can you sell me a smart TV?",
            "Have you got televisions?",
            "Where can I buy a television from you?",
        ]
        other = [
            "Do you install solar panels?",
            "Can you fix my car?",
            "Do you offer student discounts?",
        ]
        vectors = np.asarray(await embedder.passages(same + other), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        groups = GapMiner.cluster(vectors @ vectors.T, 1 - AnalyticsSettings().gap_distance)
        tv = [g for g in groups if any(i < 10 for i in g)]
        assert len(tv) == 1 and sorted(tv[0]) == list(range(10))
