from __future__ import annotations

import uuid

import pytest

from app.core.config import SettingsFactory
from app.core.enums import GateDecision, ReleaseStatus
from app.core.errors import Conflict
from app.modules.config.schemas import SkillWrite
from app.modules.config.service import SkillService
from app.modules.evaluation.models import SuiteResult
from app.modules.releases.service import (
    ReleaseBuilder,
    ReleaseCatalog,
    ReleasePromoter,
    ReleaseResolver,
    ReleaseStore,
)

SETTINGS = SettingsFactory.load()


def skill(key: str, trigger: str) -> SkillWrite:
    return SkillWrite(key=key, name=key, trigger=trigger, goal="Done.", steps="1. Do it.")


class Releases:
    def __init__(self, database, cache, registry, organization_id) -> None:
        self.database = database
        self.cache = cache
        self.registry = registry
        self.org = organization_id
        self.catalog = ReleaseCatalog(database, cache, SETTINGS)

    async def build(self):
        async with self.database.session() as session:
            return await ReleaseBuilder(session, SETTINGS).build(self.org, "test")

    async def add_skill(self, key: str) -> None:
        async with self.database.session() as session:
            await SkillService(session).create(
                self.org, skill(key, f"When the caller asks about {key} and nothing else at all")
            )

    async def live(self, release, decision=GateDecision.PASS):
        """Stages it, records a suite result, and sends it live."""
        async with self.database.session() as session:
            release = await ReleaseStore(session).get(self.org, release.id)
            promoter = ReleasePromoter(session, self.registry, SETTINGS)
            await promoter.stage(release)
            session.add(
                SuiteResult(organization_id=self.org, release_id=release.id, decision=decision)
            )
            await session.flush()
            release = await promoter.go_live(release)
        await self.catalog.resolver.invalidate(self.org)
        return release

    async def act(self, release, action: str):
        async with self.database.session() as session:
            release = await ReleaseStore(session).get(self.org, release.id)
            release = await getattr(ReleasePromoter(session, self.registry, SETTINGS), action)(
                release
            )
        await self.catalog.resolver.invalidate(self.org)
        return release


@pytest.fixture
async def releases(database, cache, registry, tenants):
    call = await tenants.create("A", "Leak")
    return Releases(database, cache, registry, call.organization_id)


class TestBuilding:
    async def test_no_change_gives_the_same_release(self, releases):
        first = await releases.build()
        again = await releases.build()
        assert again.id == first.id and again.version == 1

    async def test_one_changed_skill_is_one_different_item(self, releases, database):
        first = await releases.build()
        await releases.add_skill("check_order")
        second = await releases.build()
        async with database.session() as session:
            changes = await ReleaseStore(session).diff(first, second)
        assert second.version == 2 and second.hash != first.hash
        assert [(c.item_type, c.item_key, c.change) for c in changes] == [
            ("skills", "check_order", "added")
        ]

    async def test_a_release_gives_back_the_configuration_it_froze(self, releases):
        await releases.add_skill("check_order")
        release = await releases.build()
        config = await releases.catalog.config(releases.org, release.id)
        assert [s.key for s in config.skills] == ["check_order"] and config.version == 1


class TestPromotion:
    async def test_a_release_whose_suite_blocked_cannot_go_live(self, releases):
        release = await releases.build()
        with pytest.raises(Conflict):
            await releases.live(release, decision=GateDecision.BLOCK)

    async def test_a_call_keeps_its_release_after_a_new_one_goes_live(self, releases):
        first = await releases.live(await releases.build())
        call = uuid.uuid4()
        before = await releases.catalog.config(releases.org, first.id)
        await releases.add_skill("check_order")
        second = await releases.live(await releases.build())
        assert (first.canary_percent, second.canary_percent) == (100, 10)
        still = await releases.catalog.config(releases.org, first.id)
        assert still == before and [s.key for s in still.skills] == []
        assert await releases.catalog.resolver.resolve(releases.org, call) in (first.id, second.id)

    async def test_the_canary_gets_about_one_call_in_ten(self, releases):
        first = await releases.live(await releases.build())
        await releases.add_skill("check_order")
        second = await releases.live(await releases.build())
        resolver = releases.catalog.resolver
        chosen = [await resolver.resolve(releases.org, uuid.uuid4()) for _ in range(1000)]
        share = chosen.count(second.id) / len(chosen)
        assert 0.07 < share < 0.13 and chosen.count(first.id) + chosen.count(second.id) == 1000

    async def test_a_finished_canary_serves_every_call(self, releases):
        first = await releases.live(await releases.build())
        await releases.add_skill("check_order")
        second = await releases.act(await releases.live(await releases.build()), "complete")
        live = await releases.catalog.resolver.live(releases.org)
        assert (live.stable, live.canary, second.canary_percent) == (second.id, None, 100)
        async with releases.database.session() as session:
            assert (await ReleaseStore(session).get(releases.org, first.id)).status == (
                ReleaseStatus.RETIRED
            )

    async def test_a_rollback_reaches_new_calls_at_once(self, releases):
        first = await releases.live(await releases.build())
        await releases.add_skill("check_order")
        second = await releases.act(await releases.live(await releases.build()), "complete")
        await releases.act(second, "rollback")
        resolver: ReleaseResolver = releases.catalog.resolver
        assert await resolver.resolve(releases.org, uuid.uuid4()) == first.id

    async def test_a_canary_rollback_leaves_the_stable_release(self, releases):
        first = await releases.live(await releases.build())
        await releases.add_skill("check_order")
        canary = await releases.act(await releases.live(await releases.build()), "rollback")
        live = await releases.catalog.resolver.live(releases.org)
        assert canary.status == ReleaseStatus.RETIRED and (live.stable, live.canary) == (
            first.id,
            None,
        )

    async def test_a_tool_with_no_code_cannot_be_staged(self, releases, database):
        from app.core.enums import LatencyClass
        from app.modules.config.models import ToolBinding

        async with database.session() as session:
            session.add(
                ToolBinding(
                    organization_id=releases.org,
                    key="ghost",
                    implementation="does_not_exist",
                    description="A tool whose code was removed.",
                    parameters={"type": "object", "properties": {}},
                    latency_class=LatencyClass.A,
                    fallback_text="Not available now.",
                )
            )
        release = await releases.build()
        with pytest.raises(Conflict):
            await releases.live(release)
