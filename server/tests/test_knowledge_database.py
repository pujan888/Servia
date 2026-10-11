from __future__ import annotations

import time

import pytest

from app.core.config import SettingsFactory
from app.modules.config.service import AgentConfigService
from app.modules.knowledge.enricher import ContextualEnricher
from app.modules.knowledge.ingestion import Publisher
from app.modules.knowledge.models import KnowledgePublish
from app.modules.knowledge.schemas import (
    CardWrite,
    CollectionWrite,
    DocumentWrite,
    RetrievalTestWrite,
)
from app.modules.knowledge.service import (
    CardService,
    CollectionService,
    DocumentService,
    RetrievalTestService,
)

GUIDE = """# Care guide

## Cancel the plan (legacy plan)
The legacy plan can only be cancelled in the first 14 days.

## Cancel the plan (current plan)
The current plan can be cancelled at any time with 30 days notice.

## Error code E24
E24 means the drain pump is blocked. Clean the pump filter.

## Delivery
Deliveries arrive between Monday and Saturday. The driver phones one hour before.
"""


class KnowledgeBuilder:
    """Builds and publishes a small collection for a test tenant."""

    def __init__(self, database, embedder, router) -> None:
        self.database = database
        settings = SettingsFactory.load().knowledge.model_copy(update={"min_tests": 2})
        self.publisher = Publisher(
            database, embedder, ContextualEnricher(None, 2), router, settings
        )

    async def collection(
        self, organization_id, key="guide", default=True, content=GUIDE, tests=None, cards=()
    ):
        async with self.database.session() as session:
            collection = await CollectionService(session).save(
                organization_id,
                CollectionWrite(
                    key=key, name=key.title(), is_default=default, filter_keys=["plan"]
                ),
            )
            if content:
                for name, text, plan in self._documents(content):
                    await DocumentService(session).save(
                        collection,
                        DocumentWrite(
                            key=name,
                            title="Care guide",
                            source_type="upload",
                            content=text,
                            metadata={"plan": plan} if plan else {},
                        ),
                    )
            for card in cards:
                await CardService(session).save(collection, card)
            for test in tests or [
                RetrievalTestWrite(
                    question="How do I clean the pump for E24?", expected_heading="Error code E24"
                ),
                RetrievalTestWrite(question="What is the capital of France?", expect_refusal=True),
            ]:
                await RetrievalTestService(session).add(collection, test)
        return await self.publish(organization_id, key)

    async def publish(self, organization_id, key):
        async with self.database.session() as session:
            collection = await CollectionService(session).get(organization_id, key)
            run = KnowledgePublish(
                organization_id=organization_id,
                collection_id=collection.id,
                version=(collection.active_version or 0) + 1,
            )
            session.add(run)
            await session.flush()
        return await self.publisher.publish(run.id)

    @staticmethod
    def _documents(content):
        sections = content.split("\n## ")
        legacy = [s for s in sections if "(legacy plan)" in s]
        current = [s for s in sections if "(current plan)" in s]
        common = [s for s in sections if s not in legacy + current]
        yield "guide_common", "\n## ".join(common), None
        if legacy:
            yield "guide_legacy", "# Care guide\n\n## " + "\n## ".join(legacy), "legacy"
        if current:
            yield "guide_current", "# Care guide\n\n## " + "\n## ".join(current), "current"


@pytest.fixture
def builder(database, embedder, knowledge_router):
    return KnowledgeBuilder(database, embedder, knowledge_router)


async def config_for(database, organization_id):
    async with database.session() as session:
        return await AgentConfigService(session).load(organization_id)


class TestPublishing:
    async def test_a_passing_version_becomes_live(self, database, tenants, builder) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        report = await builder.collection(call.organization_id)
        config = await config_for(database, call.organization_id)
        assert report["passed"] and report["gate"]["pass_rate"] == 1.0
        assert config.collection("guide").active_version == 1

    async def test_a_failing_version_does_not_replace_the_live_one(
        self, database, tenants, builder
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        async with database.session() as session:
            collection = await CollectionService(session).get(call.organization_id, "guide")
            await RetrievalTestService(session).add(
                collection,
                RetrievalTestWrite(question="Tell me about delivery", expected_heading="Returns"),
            )
            await RetrievalTestService(session).add(
                collection,
                RetrievalTestWrite(
                    question="When does the driver phone?", expected_heading="Refunds"
                ),
            )
        report = await builder.publish(call.organization_id, "guide")
        config = await config_for(database, call.organization_id)
        assert not report["passed"]
        assert config.collection("guide").active_version == 1

    async def test_unchanged_documents_are_reused(self, database, tenants, builder) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        report = await builder.publish(call.organization_id, "guide")
        assert all(document["reused"] for document in report["documents"])


class TestRetrieval:
    async def search(self, router, database, organization_id, query, **values):
        config = await config_for(database, organization_id)
        scope = list(config.default_collections)
        others = [c for c in config.collections if c not in scope]
        return await router.search(organization_id, query, scope, others, values, use_cache=False)

    async def test_the_callers_plan_picks_the_right_section(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        legacy = await self.search(
            knowledge_router,
            database,
            call.organization_id,
            "How do I cancel the plan?",
            plan="legacy",
        )
        current = await self.search(
            knowledge_router,
            database,
            call.organization_id,
            "How do I cancel the plan?",
            plan="current",
        )
        assert legacy.top_heading == "Cancel the plan (legacy plan)"
        assert current.top_heading == "Cancel the plan (current plan)"

    async def test_an_exact_code_is_found(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        result = await self.search(knowledge_router, database, call.organization_id, "it says E24")
        assert result.top_heading == "Error code E24"

    async def test_an_unrelated_question_finds_nothing(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        result = await self.search(
            knowledge_router, database, call.organization_id, "What is the capital of France?"
        )
        assert result.route == "none"

    async def test_other_collections_are_searched_when_the_scope_has_nothing(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        await builder.collection(
            call.organization_id,
            key="visits",
            default=False,
            content="# Visits\n\n## Visit windows\nVisits are booked in two-hour windows.\n",
            tests=[
                RetrievalTestWrite(
                    question="How long is an engineer visit window?",
                    expected_heading="Visit windows",
                ),
                RetrievalTestWrite(question="What is the capital of France?", expect_refusal=True),
            ],
        )
        result = await self.search(
            knowledge_router,
            database,
            call.organization_id,
            "How long is the engineer visit window?",
        )
        assert (result.route, result.top_heading) == ("tenant", "Visit windows")

    async def test_a_card_answers_before_any_search(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        card = CardWrite(
            key="hours",
            questions=[
                "What are your opening hours?",
                "When are you open?",
                "What time do you close?",
            ],
            answer="We are open from nine to five on weekdays.",
        )
        await builder.collection(call.organization_id, cards=[card])
        result = await self.search(
            knowledge_router, database, call.organization_id, "what time are you open"
        )
        assert (result.route, result.card_key) == ("card", "hours")

    async def test_one_tenant_never_finds_another_tenants_content(
        self, database, tenants, builder, knowledge_router
    ) -> None:
        a = await tenants.create("Tenant A", "Boiler")
        b = await tenants.create("Tenant B", "Fridge")
        await builder.collection(a.organization_id)
        await builder.collection(
            b.organization_id,
            content="# Other\n\n## Gardening\nWater the tomatoes every evening in summer.\n",
            tests=[
                RetrievalTestWrite(
                    question="How often should I water tomatoes?", expected_heading="Gardening"
                ),
                RetrievalTestWrite(question="What is the capital of France?", expect_refusal=True),
            ],
        )
        result = await self.search(
            knowledge_router, database, b.organization_id, "E24 drain pump blocked"
        )
        assert result.route == "none"

    async def test_search_is_fast(self, database, tenants, builder, knowledge_router) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await builder.collection(call.organization_id)
        questions = ["cancel plan", "error E24", "delivery days", "driver phone", "pump filter"] * 6
        timings = []
        for question in questions:
            started = time.perf_counter()
            await self.search(
                knowledge_router, database, call.organization_id, question + " please"
            )
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        assert timings[int(len(timings) * 0.95) - 1] < 100


class TestInjection:
    async def test_a_small_card_only_collection_goes_into_the_prompt(
        self, database, tenants, builder
    ) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        card = CardWrite(
            key="hours",
            questions=[
                "What are your opening hours?",
                "When are you open?",
                "What time do you close?",
            ],
            answer="We are open from nine to five on weekdays.",
        )
        await builder.collection(
            call.organization_id,
            content=None,
            cards=[card],
            tests=[
                RetrievalTestWrite(question="When are you open?", expected_card_key="hours"),
                RetrievalTestWrite(question="What is the capital of France?", expect_refusal=True),
            ],
        )
        [spec] = (await config_for(database, call.organization_id)).collections
        assert spec.injectable and "nine to five" in spec.injected_text
