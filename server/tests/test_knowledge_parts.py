from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from app.agent.prompt.section import PromptInputs
from app.agent.prompt.sections import KnowledgeSection
from app.agent.tools.contracts import ToolContext
from app.agent.tools.exposure import ToolExposure
from app.agent.tools.implementations.knowledge import KnowledgeState, SearchKnowledge
from app.core.enums import IdentityLevel, ToolStatus
from app.core.tokens import TokenCounter
from app.modules.config.schemas import CollectionSpec
from app.modules.knowledge.catalog import CardIndex
from app.modules.knowledge.connectors import SourceError, UrlGuard
from app.modules.knowledge.enricher import ContextualEnricher
from app.modules.knowledge.retrieval import HybridSearcher
from app.modules.knowledge.schemas import Passage, RetrievalResult
from app.modules.knowledge.utils import (
    ChunkPlan,
    ContentLinter,
    HeadingSplitter,
    MarkdownNormaliser,
)
from tests.factories import ConfigFactory, PackFactory, ToolFactory


def collection(key: str, injected: str | None = None, default: bool = False) -> CollectionSpec:
    return CollectionSpec(
        id=uuid.uuid4(),
        key=key,
        name=key.title(),
        is_default=default,
        active_version=1,
        filter_keys=(),
        injected_text=injected,
    )


class TestHeadingSplitter:
    def splitter(self, max_tokens: int = 512) -> HeadingSplitter:
        return HeadingSplitter(TokenCounter(), max_tokens, 0.12)

    def test_each_h2_becomes_a_chunk_with_its_headings(self) -> None:
        plans = self.splitter().split("# Guide\n\n## One\nFirst.\n\n## Two\nSecond.\n", "x")
        assert [(p.title, p.heading) for p in plans] == [("Guide", "One"), ("Guide", "Two")]
        assert plans[0].text.startswith("# Guide\n## One")

    def test_a_numbered_list_is_never_split(self) -> None:
        steps = "\n".join(
            f"{i}. Step number {i} of the procedure, with some words." for i in range(1, 30)
        )
        markdown = f"# Guide\n\n## Steps\nIntro sentence.\n\n{steps}\n\nClosing paragraph here.\n"
        plans = self.splitter(max_tokens=120).split(markdown, "x")
        holding = [p for p in plans if "1. Step number 1 " in p.body]
        assert len(holding) == 1 and "29. Step number 29" in holding[0].body

    def test_a_long_section_is_split_with_overlap(self) -> None:
        paragraph = " ".join(f"Sentence number {i} says something useful." for i in range(40))
        markdown = f"# Guide\n\n## Long\n{paragraph}\n\n{paragraph}\n"
        plans = self.splitter(max_tokens=200).split(markdown, "x")
        assert len(plans) > 1
        assert all(p.heading == "Long" for p in plans)
        assert plans[1].body.split("\n\n")[0] in plans[0].body

    def test_text_before_the_first_h2_keeps_the_title(self) -> None:
        plans = self.splitter().split("# Guide\nIntro text.\n\n## One\nFirst.\n", "x")
        assert plans[0].heading == "Guide"


class TestNormaliserAndLinter:
    def test_html_becomes_markdown_headings(self) -> None:
        markdown = MarkdownNormaliser.normalise(
            "<h1>Title</h1><h2>Part</h2><p>Text</p><script>x()</script>", "html"
        )
        assert "# Title" in markdown and "## Part" in markdown
        assert "x()" not in markdown and "Menu" not in markdown

    def test_facts_that_change_are_flagged(self) -> None:
        text = (
            "Engineers visit from nine to five.\n"
            "A visit costs £49.\n"
            "Open at 9am.\n"
            "The cover lasts 2 years."
        )
        kinds = [(w.line, w.kind) for w in ContentLinter.lint(text)]
        assert kinds == [(1, "time"), (2, "price"), (3, "time")]


class TestUrlGuard:
    @pytest.mark.parametrize(
        "url",
        ["http://localhost/x", "http://127.0.0.1/x", "http://10.0.0.5/x", "file:///etc/passwd"],
    )
    def test_non_public_addresses_are_refused(self, url) -> None:
        with pytest.raises(SourceError):
            asyncio.run(UrlGuard.check(url))


class TestEnricher:
    plan = ChunkPlan(
        0, "Returns", "Warranty cover", "The warranty lasts 2 years. Code E24 is not covered."
    )

    def test_without_llm_the_prefix_is_added(self) -> None:
        [text] = asyncio.run(ContextualEnricher(None, 2).enrich([self.plan]))
        assert text == "Returns, section 'Warranty cover': " + self.plan.body

    def test_a_rewrite_that_loses_a_code_is_rejected(self) -> None:
        assert not ContextualEnricher.acceptable(self.plan, "The warranty lasts two years.")

    def test_a_padded_rewrite_is_rejected(self) -> None:
        padded = self.plan.body + " This is explained in detail in the text above." * 4
        assert not ContextualEnricher.acceptable(self.plan, padded)

    def test_a_rewrite_that_repeats_the_section_name_is_rejected(self) -> None:
        assert not ContextualEnricher.acceptable(
            self.plan, "As stated in Warranty cover, " + self.plan.body
        )

    def test_a_clean_rewrite_is_accepted(self) -> None:
        rewritten = "The appliance warranty lasts 2 years. Error code E24 is not covered."
        assert ContextualEnricher.acceptable(self.plan, rewritten)


class TestPromptAndExposure:
    def test_searchable_knowledge_tells_the_model_to_search(self) -> None:
        inputs = PromptInputs(PackFactory.build(), collections=(collection("general"),))
        assert "search_knowledge" in KnowledgeSection().render(inputs, 800).text

    def test_small_collections_are_in_the_prompt_in_full(self) -> None:
        cards = "Answers from Help:\n- Q: When are you open?\n  A: Nine to five."
        inputs = PromptInputs(
            PackFactory.build(), collections=(collection("help", injected=cards),)
        )
        text = KnowledgeSection().render(inputs, 800).text
        assert cards in text and "search_knowledge" not in text

    def test_search_is_hidden_when_everything_is_in_the_prompt(self) -> None:
        tools = [ToolFactory.binding("search_knowledge"), ToolFactory.binding("end_call")]
        config = ConfigFactory.build(tools=tools).model_copy(
            update={"collections": (collection("help", injected="Answers"),)}
        )
        assert ToolExposure().keys(config, []) == ["end_call"]


class TestSmallRetrievalParts:
    def test_codes_are_tokens_with_a_digit(self) -> None:
        assert HybridSearcher.codes("My washer shows E24 and part SX-300, what now?") == [
            "e24",
            "300",
        ]

    def test_card_index_returns_the_best_variant(self) -> None:
        index = CardIndex(
            ("a", "b"), ("A", "B"), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        assert index.best(np.array([0.2, 0.9], dtype=np.float32))[:2] == ("b", "B")


class FakeRouter:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls = []

    async def search(self, organization_id, query, scope, others, caller_values):
        self.calls.append((query, scope, others, caller_values))
        return self.result


class TestSearchKnowledgeTool:
    def ctx(self) -> ToolContext:
        return ToolContext(
            organization_id=uuid.uuid4(),
            call_id=uuid.uuid4(),
            caller_id=uuid.uuid4(),
            identity_level=IdentityLevel.UNKNOWN,
            turn_index=1,
            timezone="UTC",
            knowledge_scope=(collection("general"),),
            caller_values={"plan": "legacy"},
        )

    def run(self, result: RetrievalResult):
        ctx = self.ctx()
        router = FakeRouter(result)
        outcome = asyncio.run(SearchKnowledge(router).execute(ctx, {"query": "cancel?"}))
        return outcome, ctx, router

    def test_passages_come_with_the_answer_only_rule(self) -> None:
        passage = Passage(
            title="Plan",
            heading="Cancel",
            text="Cancel any time.",
            similarity=0.8,
            chunk_id=uuid.uuid4(),
        )
        outcome, ctx, router = self.run(
            RetrievalResult(route="scoped", passages=[passage], query="cancel?")
        )
        assert outcome.status == ToolStatus.OK
        assert outcome.speech.startswith(SearchKnowledge.ONLY_THESE)
        assert "[1] Plan — Cancel" in outcome.speech
        assert router.calls[0][3] == {"plan": "legacy"}
        assert ctx.tool_state[KnowledgeState.KEY][0]["route"] == "scoped"

    def test_a_card_is_returned_as_an_answer(self) -> None:
        outcome, _, _ = self.run(
            RetrievalResult(route="card", card_key="hours", card_answer="Nine to five.")
        )
        assert "Nine to five." in outcome.speech

    def test_nothing_found_is_empty_not_failed(self) -> None:
        outcome, _, _ = self.run(RetrievalResult(route="none"))
        assert outcome.status == ToolStatus.EMPTY
        assert "Do not guess" in outcome.speech
