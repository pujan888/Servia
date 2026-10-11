from __future__ import annotations

import numpy as np
import pytest

from app.analysis.outcomes.gaps import GapMiner
from app.core.config import EvaluationSettings, ReleaseSettings
from app.core.enums import GateDecision
from app.eval.assertions import Assertions, Observation, ToolCall, TurnSeen
from app.eval.judge import JudgeVerdict, LLMJudge
from app.eval.mocks import MockToolLayer
from app.eval.suite import RegressionGate
from app.modules.releases.canary import CanaryMonitor


def seen(**overrides) -> Observation:
    values = {
        "turns": [
            TurnSeen(1, "I need an engineer", "Which day suits you?", "book_visit", 900, False),
            TurnSeen(
                2,
                "Thursday",
                "Logged. Your reference is K, Q, seven, M, three, A. A visit costs £49.",
                "book_visit",
                1200,
                False,
            ),
            TurnSeen(
                3,
                "Do you sell TVs?",
                "I do not have that information. Would you like a callback?",
                "general_inquiry",
                800,
                True,
            ),
        ],
        "tools": [
            ToolCall(
                2,
                "request_callback",
                {"preferred_time": "Thursday morning"},
                "accepted",
                "logged",
                "KQ7M3A",
            )
        ],
        "escalation": None,
        "unexpected": [],
        "known_text": "Visits cost £49.",
    }
    return Observation(**{**values, **overrides})


class TestAssertions:
    @pytest.mark.parametrize(
        ("spec", "passes"),
        [
            (
                {
                    "type": "tool_called",
                    "tool": "request_callback",
                    "arguments": {"preferred_time": "thursday"},
                },
                True,
            ),
            (
                {
                    "type": "tool_called",
                    "tool": "request_callback",
                    "arguments": {"preferred_time": "friday"},
                },
                False,
            ),
            ({"type": "tool_not_called", "tool": "create_work_item"}, True),
            ({"type": "tool_not_called", "tool": "request_callback"}, False),
            ({"type": "reference_said"}, True),
            ({"type": "refused"}, True),
            ({"type": "escalated"}, False),
            ({"type": "skill_loaded", "skill": "book_visit", "turn": 2}, True),
            ({"type": "skill_loaded", "skill": "check_order"}, False),
            ({"type": "no_invented_price"}, True),
            ({"type": "turn_count", "max": 2, "tool": "request_callback"}, True),
            ({"type": "turn_count", "max": 1, "tool": "request_callback"}, False),
        ],
    )
    def test_each_type(self, spec, passes) -> None:
        assert Assertions.build(spec).check(seen()).passed is passes

    def test_an_invented_price_fails(self) -> None:
        check = Assertions.build({"type": "no_invented_price"}).check(seen(known_text=""))
        assert not check.passed and "49" in check.detail

    def test_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Assertions.build({"type": "sounds_nice"})


class TestMockToolLayer:
    RESPONSES = [
        {
            "tool_key": "get_work_item",
            "arguments": {"reference": "AAA"},
            "status": "ok",
            "text": "first",
        },
        {
            "tool_key": "get_work_item",
            "arguments": {"reference": "KQ7M3A"},
            "status": "ok",
            "text": "second",
        },
        {
            "tool_key": "request_callback",
            "arguments": {"reason": "leak"},
            "status": "accepted",
            "text": "logged",
            "reference": "RT4VWX",
        },
    ]

    def test_the_same_arguments_are_matched_first(self) -> None:
        mocks = MockToolLayer(self.RESPONSES)
        assert mocks.answer("get_work_item", {"reference": "kq7m3a "}).text == "second"
        assert mocks.answer("get_work_item", {"reference": "ZZZ"}).text == "first"

    def test_free_text_worded_differently_still_gets_its_answer(self) -> None:
        result = MockToolLayer(self.RESPONSES).answer(
            "request_callback", {"reason": "water everywhere"}
        )
        assert (result.status, result.reference) == ("accepted", "RT4VWX")

    def test_a_call_with_no_recorded_answer_fails_and_is_reported(self) -> None:
        mocks = MockToolLayer(self.RESPONSES)
        assert mocks.answer("create_work_item", {}).status == "failed"
        assert mocks.unexpected == ["create_work_item"]


class TestRegressionGate:
    gate = RegressionGate(EvaluationSettings())
    BASE = {
        "scenarios": 20,
        "pass_rate": 0.95,
        "completion": 0.9,
        "turn_p95_ms": 2000,
        "faithfulness": None,
        "invented_prices": 0,
    }

    def decide(self, **changes):
        return self.gate.decide({**self.BASE, **changes}, self.BASE)

    def test_the_same_results_pass(self) -> None:
        assert self.decide()[0] == GateDecision.PASS

    def test_a_deliberate_regression_is_blocked(self) -> None:
        decision, reasons = self.decide(pass_rate=0.90, completion=0.8)
        assert decision == GateDecision.BLOCK and any("pass rate" in r for r in reasons)

    def test_a_small_drop_warns(self) -> None:
        assert self.decide(pass_rate=0.935)[0] == GateDecision.WARN

    def test_slower_replays_warn_then_block(self) -> None:
        assert self.decide(turn_p95_ms=2150)[0] == GateDecision.WARN
        assert self.decide(turn_p95_ms=2300)[0] == GateDecision.BLOCK

    def test_any_invented_price_blocks(self) -> None:
        assert self.decide(invented_prices=1)[0] == GateDecision.BLOCK

    def test_without_a_baseline_it_passes_and_says_so(self) -> None:
        decision, reasons = self.gate.decide({**self.BASE}, None)
        assert decision == GateDecision.PASS and "No baseline" in reasons[-1]


class TestJudgeCalibration:
    def test_agreement_is_within_one_point_on_all_three(self) -> None:
        verdict = JudgeVerdict(helpfulness=4, faithfulness=5, tone=4)
        close = JudgeVerdict(helpfulness=5, faithfulness=4, tone=4)
        far = JudgeVerdict(helpfulness=2, faithfulness=5, tone=4)
        assert LLMJudge.agreement([(verdict, close), (verdict, far)]) == 0.5


class TestCanaryTolerances:
    monitor = CanaryMonitor.__new__(CanaryMonitor)
    monitor._settings = ReleaseSettings()
    OLD = {
        "escalation_rate": 0.1,
        "refusal_rate": 0.1,
        "task_completion": 0.8,
        "first_audio_p95_ms": 1500,
    }

    def test_inside_every_tolerance_is_not_worse(self) -> None:
        assert self.monitor.worse({**self.OLD, "escalation_rate": 0.14}, self.OLD) == []

    def test_one_measurement_outside_its_tolerance_is_worse(self) -> None:
        worse = self.monitor.worse({**self.OLD, "task_completion": 0.7}, self.OLD)
        assert worse == ["task_completion: 0.8 → 0.7"]


class TestClustering:
    def test_groups_form_only_above_the_similarity(self) -> None:
        similarity = np.array(
            [[1, 0.9, 0.85, 0.1], [0.9, 1, 0.8, 0.2], [0.85, 0.8, 1, 0.1], [0.1, 0.2, 0.1, 1]]
        )
        groups = sorted(sorted(g) for g in GapMiner.cluster(similarity, 0.75))
        assert groups == [[0, 1, 2], [3]]
