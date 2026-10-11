from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.modules.config.schemas import CollectionSpec
from app.modules.knowledge.models import RetrievalTest
from app.modules.knowledge.retrieval import KnowledgeRouter


@dataclass
class GateReport:
    passed: bool
    pass_rate: float
    tests: int
    failures: list[dict] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "tests": self.tests,
            "failures": self.failures,
            "reason": self.reason,
        }


class VerificationGate:
    """Runs retrieval only, with no model, on the new version. If the right
    answer is not first, no prompt change can fix the agent's reply, so a failing
    version is never made live."""

    def __init__(self, router: KnowledgeRouter, min_tests: int, pass_rate: float) -> None:
        self._router = router
        self._min_tests = min_tests
        self._pass_rate = pass_rate

    async def run(
        self, organization_id: uuid.UUID, candidate: CollectionSpec, tests: Sequence[RetrievalTest]
    ) -> GateReport:
        if len(tests) < self._min_tests:
            return GateReport(
                False, 0.0, len(tests), reason=f"needs at least {self._min_tests} tests"
            )
        failures = []
        for test in tests:
            result = await self._router.search(
                organization_id,
                test.question,
                [candidate],
                caller_values=test.filters,
                use_cache=False,
            )
            if not self._passes(test, result):
                failures.append(
                    {
                        "question": test.question,
                        "expected": test.expected_card_key or test.expected_heading or "refusal",
                        "got": result.card_key or result.top_heading or result.route,
                        "similarity": result.best_similarity,
                    }
                )
        rate = round(1 - len(failures) / len(tests), 3)
        return GateReport(rate >= self._pass_rate, rate, len(tests), failures)

    @staticmethod
    def _passes(test: RetrievalTest, result) -> bool:
        if test.expect_refusal:
            return result.route == "none"
        if test.expected_card_key:
            return result.card_key == test.expected_card_key
        return result.top_heading == test.expected_heading
