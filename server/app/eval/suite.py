from __future__ import annotations

import asyncio
import statistics
import uuid
from typing import Any

from loguru import logger
from sqlalchemy import select

from app.agent.runtime import AgentRuntime
from app.analysis.model import AnalysisModel
from app.core.config import EvaluationSettings
from app.core.container import Container
from app.core.enums import GateDecision, ReleaseStatus
from app.eval.assertions import Assertions, Checked, NoInventedPrice, Observation
from app.eval.judge import LLMJudge
from app.eval.replay import ReplayRunner
from app.modules.evaluation.models import (
    AssertionResult,
    JudgeScore,
    Scenario,
    ScenarioRun,
    SuiteResult,
)
from app.modules.jobs.models import Job
from app.modules.jobs.runner import JobHandler
from app.modules.jobs.service import JobQueue
from app.modules.releases.models import AgentRelease
from app.modules.releases.service import ReleaseCatalog
from app.voice.fillers import FillerLibrary


class RegressionGate:
    """Compares a suite result with its baseline. The worst finding decides."""

    RANK = (GateDecision.PASS, GateDecision.WARN, GateDecision.BLOCK)

    def __init__(self, settings: EvaluationSettings) -> None:
        self._s = settings

    def decide(
        self, current: dict[str, Any], baseline: dict[str, Any] | None
    ) -> tuple[GateDecision, list[str]]:
        if not current["scenarios"]:
            return GateDecision.PASS, ["The suite has no scenarios."]
        findings: list[tuple[GateDecision, str]] = []
        if current["invented_prices"]:
            findings.append(
                (
                    GateDecision.BLOCK,
                    f"An invented price in {current['invented_prices']} scenarios.",
                )
            )
        if not baseline or not baseline.get("scenarios"):
            findings.append((GateDecision.PASS, "No baseline to compare with."))
            return self._worst(findings)
        s = self._s
        findings += [
            self._drop(
                "assertion pass rate",
                current,
                baseline,
                "pass_rate",
                s.warn_pass_rate,
                s.block_pass_rate,
            ),
            self._drop(
                "task completion",
                current,
                baseline,
                "completion",
                s.warn_completion,
                s.block_completion,
            ),
        ]
        if current.get("faithfulness") is not None and baseline.get("faithfulness") is not None:
            findings.append(
                self._drop(
                    "faithfulness",
                    current,
                    baseline,
                    "faithfulness",
                    s.warn_faithfulness,
                    s.block_faithfulness,
                )
            )
        if baseline.get("turn_p95_ms"):
            rise = (current["turn_p95_ms"] - baseline["turn_p95_ms"]) / baseline["turn_p95_ms"]
            level = (
                GateDecision.BLOCK
                if rise >= s.block_turn_p95
                else GateDecision.WARN
                if rise >= s.warn_turn_p95
                else GateDecision.PASS
            )
            findings.append((level, f"replay turn p95 changed by {rise:+.0%}."))
        return self._worst(findings)

    @staticmethod
    def _drop(name, current, baseline, key, warn, block) -> tuple[GateDecision, str]:
        drop = baseline[key] - current[key]
        level = (
            GateDecision.BLOCK
            if drop >= block
            else GateDecision.WARN
            if drop >= warn
            else GateDecision.PASS
        )
        return level, f"{name}: {baseline[key]:.3f} → {current[key]:.3f}."

    def _worst(self, findings) -> tuple[GateDecision, list[str]]:
        decision = max((level for level, _ in findings), key=self.RANK.index)
        return decision, [f"{level}: {text}" for level, text in findings]


class SuiteRunner:
    """Runs every enabled scenario of a tenant on one release, each several
    times. An assertion passes if it passed in enough of the runs; the model
    does not answer the same way twice."""

    def __init__(self, container: Container) -> None:
        self._container = container
        self._settings = container.settings.evaluation
        self._replay = ReplayRunner(container, AgentRuntime(container.settings, FillerLibrary()))
        self._catalog = ReleaseCatalog(container.database, container.cache, container.settings)
        self._gate = RegressionGate(self._settings)
        judge_model = AnalysisModel(self._settings.judge_model, 30)
        self._judge = LLMJudge(judge_model) if self._settings.judge_enabled else None

    async def run(self, suite_id: uuid.UUID) -> SuiteResult:
        database = self._container.database
        async with database.session() as session:
            suite = await session.get(SuiteResult, suite_id)
            scenarios = (
                await session.scalars(
                    select(Scenario).where(
                        Scenario.organization_id == suite.organization_id,
                        Scenario.enabled.is_(True),
                    )
                )
            ).all()
        config = await self._catalog.config(suite.organization_id, suite.release_id)
        limit = asyncio.Semaphore(self._settings.concurrency)

        async def one(scenario: Scenario) -> dict[str, Any]:
            async with limit:
                return await self._scenario(suite, scenario, config)

        tallies = list(await asyncio.gather(*(one(scenario) for scenario in scenarios)))
        measurements = self._measure(tallies)
        async with database.session() as session:
            baseline = await self._baseline(session, suite)
            decision, reasons = self._gate.decide(
                measurements, baseline.measurements if baseline else None
            )
            stored = await session.get(SuiteResult, suite_id)
            stored.measurements, stored.decision, stored.reasons = measurements, decision, reasons
            stored.baseline_id = baseline.id if baseline else None
        if stored.nightly and decision != GateDecision.PASS:
            logger.warning("nightly suite for release {} is worse: {}", suite.release_id, reasons)
        logger.info("suite {}: {} {}", suite_id, decision, measurements)
        return stored

    async def _scenario(self, suite: SuiteResult, scenario: Scenario, config) -> dict[str, Any]:
        per_assertion = [0] * len(scenario.assertions)
        timings, unexpected, faithfulness = [], 0, []
        for number in range(1, self._settings.runs + 1):
            seen = await self._replay.run(scenario, config, suite.release_id)
            checks = [Assertions.build(spec).check(seen) for spec in scenario.assertions]
            for index, check in enumerate(checks):
                per_assertion[index] += check.passed
            timings += [turn.ms for turn in seen.turns]
            unexpected += len(seen.unexpected)
            faithfulness += await self._store_run(suite, scenario, number, seen, checks)
        needed = self._settings.runs_to_pass
        passed = [count >= needed for count in per_assertion]
        price = [
            ok
            for spec, ok in zip(scenario.assertions, passed, strict=True)
            if spec["type"] == NoInventedPrice.kind
        ]
        return {
            "assertions": len(passed),
            "passed": sum(passed),
            "complete": all(passed),
            "invented_price": not all(price),
            "timings": timings,
            "unexpected": unexpected,
            "faithfulness": faithfulness,
        }

    async def _store_run(
        self,
        suite: SuiteResult,
        scenario: Scenario,
        number: int,
        seen: Observation,
        checks: list[Checked],
    ) -> list[int]:
        scores = await self._judge_turns(seen) if self._judge else []
        async with self._container.database.session() as session:
            run = ScenarioRun(
                organization_id=suite.organization_id,
                suite_id=suite.id,
                scenario_id=scenario.id,
                run_number=number,
                passed=all(check.passed for check in checks),
                transcript=[
                    {"turn": t.index, "caller": t.caller, "agent": t.agent} for t in seen.turns
                ],
                turn_ms=[turn.ms for turn in seen.turns],
            )
            session.add(run)
            await session.flush()
            session.add_all(
                AssertionResult(
                    organization_id=suite.organization_id,
                    run_id=run.id,
                    position=index,
                    assertion=spec,
                    passed=check.passed,
                    detail=check.detail,
                )
                for index, (spec, check) in enumerate(zip(scenario.assertions, checks, strict=True))
            )
            session.add_all(
                JudgeScore(
                    organization_id=suite.organization_id,
                    run_id=run.id,
                    model=self._judge.name,
                    **verdict,
                )
                for verdict in scores
            )
        return [verdict["faithfulness"] for verdict in scores]

    async def _judge_turns(self, seen: Observation) -> list[dict[str, int]]:
        """Only turns that used the knowledge are judged: faithfulness needs facts."""
        verdicts, conversation = [], []
        for turn in seen.turns:
            conversation.append(f"caller: {turn.caller}")
            if turn.index in seen.passages:
                verdict = await self._judge.score(
                    conversation, turn.agent, seen.passages[turn.index]
                )
                verdicts.append(verdict.model_dump())
            conversation.append(f"agent: {turn.agent}")
        return verdicts

    def _measure(self, tallies: list[dict[str, Any]]) -> dict[str, Any]:
        timings = [ms for t in tallies for ms in t["timings"]]
        faithfulness = [score for t in tallies for score in t["faithfulness"]]
        assertions = sum(t["assertions"] for t in tallies)
        return {
            "scenarios": len(tallies),
            "runs": len(tallies) * self._settings.runs,
            "pass_rate": round(sum(t["passed"] for t in tallies) / assertions, 4)
            if assertions
            else 1.0,
            "completion": round(sum(t["complete"] for t in tallies) / len(tallies), 4)
            if tallies
            else 1.0,
            "turn_p95_ms": self._p95(timings),
            "faithfulness": round(statistics.mean(faithfulness), 3) if faithfulness else None,
            "invented_prices": sum(t["invented_price"] for t in tallies),
            "unexpected_calls": sum(t["unexpected"] for t in tallies),
        }

    @staticmethod
    def _p95(values: list[int]) -> int:
        if len(values) < 2:
            return values[0] if values else 0
        return round(statistics.quantiles(values, n=20)[-1])

    async def _baseline(self, session, suite: SuiteResult) -> SuiteResult | None:
        """A nightly run is compared with the previous night on the same release;
        a new release with the latest suite of the stable live release."""
        statement = select(SuiteResult).where(
            SuiteResult.organization_id == suite.organization_id,
            SuiteResult.id != suite.id,
            SuiteResult.decision.is_not(None),
        )
        if suite.nightly:
            statement = statement.where(
                SuiteResult.release_id == suite.release_id, SuiteResult.nightly.is_(True)
            )
        else:
            stable = select(AgentRelease.id).where(
                AgentRelease.organization_id == suite.organization_id,
                AgentRelease.status == ReleaseStatus.LIVE,
                AgentRelease.canary_percent == 100,
                AgentRelease.id != suite.release_id,
            )
            statement = statement.where(SuiteResult.release_id.in_(stable))
        return await session.scalar(statement.order_by(SuiteResult.created_at.desc()).limit(1))


class SuiteJobs:
    KIND = "eval.suite"

    @classmethod
    async def enqueue(
        cls, session, organization_id: uuid.UUID, release_id: uuid.UUID, nightly: bool
    ) -> SuiteResult:
        suite = SuiteResult(organization_id=organization_id, release_id=release_id, nightly=nightly)
        session.add(suite)
        await session.flush()
        await JobQueue(session).enqueue(organization_id, cls.KIND, {"suite_id": str(suite.id)})
        return suite


class SuiteJobHandler(JobHandler):
    kind = SuiteJobs.KIND
    max_attempts = 2

    def __init__(self, runner: SuiteRunner) -> None:
        self._runner = runner

    async def handle(self, job: Job) -> None:
        await self._runner.run(uuid.UUID(job.payload["suite_id"]))


class NightlyRun:
    """Runs the suite on every live release each night. A supplier can change a
    model behind the same name; this is how that is noticed."""

    def __init__(self, container: Container) -> None:
        self._database = container.database

    async def run(self) -> None:
        async with self._database.session() as session:
            live = (
                await session.scalars(
                    select(AgentRelease).where(AgentRelease.status == ReleaseStatus.LIVE)
                )
            ).all()
            for release in live:
                await SuiteJobs.enqueue(session, release.organization_id, release.id, nightly=True)
        if live:
            logger.info("nightly suites queued for {} live releases", len(live))
