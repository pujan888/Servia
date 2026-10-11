from __future__ import annotations

from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.section import PromptInputs
from app.core.config import PromptSettings
from app.core.tokens import TokenCounter
from tests.factories import PackFactory


class TestPromptCompiler:
    def setup_method(self) -> None:
        self.compiler = PromptCompiler(PromptSettings())
        self.counter = TokenCounter(PromptSettings().token_margin)

    def test_same_inputs_give_same_prefix_and_hash(self) -> None:
        first = self.compiler.compile(PromptInputs(PackFactory.build()))
        second = self.compiler.compile(PromptInputs(PackFactory.build()))
        assert first.prefix == second.prefix
        assert first.prefix_hash == second.prefix_hash

    def test_local_time_is_in_the_suffix_not_the_prefix(self) -> None:
        pack = PackFactory.build()
        compiled = self.compiler.compile(PromptInputs(pack))
        suffix = self.compiler.turn_suffix(PromptInputs(pack))
        assert "local time" not in compiled.prefix
        assert "local time" in suffix.text

    def test_sections_are_in_tier_order(self) -> None:
        tiers = [
            report.tier
            for report in self.compiler.compile(PromptInputs(PackFactory.build())).sections
        ]
        order = ["STATIC", "TENANT", "CALL"]
        assert tiers == sorted(tiers, key=order.index)

    def test_forty_open_items_stay_inside_the_context_budget(self) -> None:
        compiled = self.compiler.compile(
            PromptInputs(PackFactory.build(tasks=40, profile_fields=10))
        )
        context = next(r for r in compiled.sections if r.name == "context")
        assert context.tokens <= PromptSettings().context_tokens
        assert compiled.truncations

    def test_truncation_follows_the_fixed_sequence(self) -> None:
        compiled = self.compiler.compile(
            PromptInputs(PackFactory.build(tasks=40, profile_fields=10))
        )
        assert compiled.truncations[0].startswith("profile_fields")
        assert compiled.truncations[1].startswith("tasks")

    def test_small_pack_is_not_truncated(self) -> None:
        assert self.compiler.compile(PromptInputs(PackFactory.build())).truncations == []

    def test_prefix_does_not_change_between_turns(self) -> None:
        pack = PackFactory.build()
        compiled = self.compiler.compile(PromptInputs(pack))
        self.compiler.turn_suffix(PromptInputs(pack))
        self.compiler.turn_suffix(PromptInputs(pack))
        assert self.compiler.compile(PromptInputs(pack)).prefix_hash == compiled.prefix_hash
