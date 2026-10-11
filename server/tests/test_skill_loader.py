from __future__ import annotations

from app.agent.skills.loader import SkillLoader
from tests.factories import AgentRuntimeFactory, ConfigFactory, PackFactory


class TestSkillLoader:
    def setup_method(self) -> None:
        config = ConfigFactory.build(
            ConfigFactory.skill("book_visit"), ConfigFactory.skill("check_order")
        )
        self.state = AgentRuntimeFactory.state(PackFactory.build(), config)
        self.loader = SkillLoader()

    def test_unknown_key_returns_the_error_and_changes_nothing(self) -> None:
        assert self.loader.load(self.state, "refund_everything") == SkillLoader.UNKNOWN
        assert self.state.active_skill == "general_inquiry"

    def test_valid_key_sets_the_active_skill_and_returns_the_procedure(self) -> None:
        text = self.loader.load(self.state, "check_order")
        assert text.startswith(SkillLoader.PREFACE)
        assert "Procedure: Check Order (check_order)" in text
        assert self.state.active_skill == "check_order"

    def test_loading_the_active_skill_again_is_a_no_op(self) -> None:
        self.loader.load(self.state, "check_order")
        assert self.loader.load(self.state, "check_order") == SkillLoader.ALREADY_ACTIVE
        assert len(self.state.skill_history) == 1

    def test_three_jobs_in_one_call_are_recorded_in_order(self) -> None:
        for turn, key in enumerate(("book_visit", "check_order", "general_inquiry"), start=1):
            self.state.turn_index = turn
            self.loader.load(self.state, key)
        assert self.state.skill_history == [
            ("book_visit", 1),
            ("check_order", 2),
            ("general_inquiry", 3),
        ]

    def test_a_skill_from_another_tenant_cannot_be_loaded(self) -> None:
        assert self.loader.load(self.state, "other_tenant_skill") == SkillLoader.UNKNOWN
