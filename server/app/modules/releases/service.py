from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from loguru import logger
from pydantic import TypeAdapter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.section import PromptInputs
from app.agent.tools.registry import ToolRegistry
from app.core.cache import CacheKeys, CacheLayer
from app.core.config import Settings
from app.core.database import Database
from app.core.enums import GateDecision, IdentityLevel, ReleaseStatus
from app.core.errors import Conflict, NotFound
from app.modules.accounts.models import Organization
from app.modules.config.schemas import AgentConfig
from app.modules.config.service import AgentConfigService
from app.modules.context.schemas import CallerContext, ContextPack, OrganizationContext
from app.modules.evaluation.models import SuiteResult
from app.modules.knowledge.models import KnowledgePublish
from app.modules.releases.items import ReleaseItem, ReleaseItems
from app.modules.releases.models import AgentRelease, ReleaseContent
from app.modules.releases.schemas import ItemDiff, LiveSet


class ReleaseStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, organization_id: uuid.UUID, release_id: uuid.UUID) -> AgentRelease:
        release = await self._session.get(AgentRelease, release_id)
        if release is None or release.organization_id != organization_id:
            raise NotFound("Release not found", release_id=str(release_id))
        return release

    async def list(self, organization_id: uuid.UUID) -> Sequence[AgentRelease]:
        result = await self._session.execute(
            select(AgentRelease)
            .where(AgentRelease.organization_id == organization_id)
            .order_by(AgentRelease.version.desc())
        )
        return result.scalars().all()

    async def items(self, release: AgentRelease) -> list[ReleaseItem]:
        result = await self._session.execute(
            select(ReleaseContent).where(ReleaseContent.release_id == release.id)
        )
        return [
            ReleaseItem(r.item_type, r.item_key, r.position, r.content, r.hash)
            for r in result.scalars()
        ]

    async def live(self, organization_id: uuid.UUID) -> Sequence[AgentRelease]:
        result = await self._session.execute(
            select(AgentRelease).where(
                AgentRelease.organization_id == organization_id,
                AgentRelease.status == ReleaseStatus.LIVE,
            )
        )
        return result.scalars().all()

    async def diff(self, first: AgentRelease, second: AgentRelease) -> list[ItemDiff]:
        before = {(i.item_type, i.item_key): i for i in await self.items(first)}
        after = {(i.item_type, i.item_key): i for i in await self.items(second)}
        changes = []
        for key in sorted(before.keys() | after.keys()):
            old, new = before.get(key), after.get(key)
            if old and new and old.hash == new.hash:
                continue
            change = "added" if old is None else "removed" if new is None else "changed"
            changes.append(
                ItemDiff(
                    item_type=key[0],
                    item_key=key[1],
                    change=change,
                    before=old.content if old else None,
                    after=new.content if new else None,
                )
            )
        return changes


class ReleaseBuilder:
    """Freezes the tenant's current configuration. If nothing differs from the
    newest release, that release is returned and no new one is made."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def build(self, organization_id: uuid.UUID, created_by: str) -> AgentRelease:
        settings = self._settings
        config = await AgentConfigService(
            self._session, settings.knowledge.injection_tokens, settings.detection, settings.latency
        ).load(organization_id)
        fingerprint = PromptCompiler(settings.prompt).fingerprint()
        items = ReleaseItems.from_config(config, fingerprint)
        digest = ReleaseItems.release_hash(items)
        newest = (await ReleaseStore(self._session).list(organization_id))[:1]
        if newest and newest[0].hash == digest:
            return newest[0]
        version = (
            await self._session.scalar(
                select(func.coalesce(func.max(AgentRelease.version), 0)).where(
                    AgentRelease.organization_id == organization_id
                )
            )
        ) + 1
        release = AgentRelease(
            organization_id=organization_id, version=version, hash=digest, created_by=created_by
        )
        self._session.add(release)
        await self._session.flush()
        self._session.add_all(
            ReleaseContent(
                organization_id=organization_id,
                release_id=release.id,
                item_type=item.item_type,
                item_key=item.item_key,
                position=item.position,
                content=item.content,
                hash=item.hash,
            )
            for item in items
        )
        await self._session.flush()
        logger.info("release v{} built for {} ({} items)", version, organization_id, len(items))
        return release


class ReleaseResolver:
    """Gives each new call its release. The same call always gets the same one.
    The live set is cached for a minute and dropped on every change, so a
    rollback reaches new calls at once, and in a minute at worst."""

    LIVE = TypeAdapter(LiveSet)

    def __init__(self, database: Database, cache: CacheLayer) -> None:
        self._database = database
        self._cache = cache

    async def resolve(self, organization_id: uuid.UUID, call_id: uuid.UUID) -> uuid.UUID:
        live = await self.live(organization_id)
        if live.canary and self.bucket(call_id) < live.canary_percent:
            return live.canary
        if live.stable is None:
            raise NotFound("The tenant has no live release", organization_id=str(organization_id))
        return live.stable

    async def live(self, organization_id: uuid.UUID) -> LiveSet:
        key = CacheKeys.build(organization_id, "live")
        return await self._cache.get(key, self.LIVE, lambda: self._load(organization_id))

    async def invalidate(self, organization_id: uuid.UUID) -> None:
        await self._cache.delete_prefix(CacheKeys.build(organization_id, "live"))

    @staticmethod
    def bucket(call_id: uuid.UUID) -> int:
        return int(hashlib.sha256(f"release:{call_id}".encode()).hexdigest(), 16) % 100

    async def _load(self, organization_id: uuid.UUID) -> LiveSet:
        async with self._database.session() as session:
            live = await ReleaseStore(session).live(organization_id)
        stable = next((r for r in live if r.canary_percent == 100), None)
        canary = next((r for r in live if r.canary_percent < 100), None)
        return LiveSet(
            stable=stable.id if stable else None,
            canary=canary.id if canary else None,
            canary_percent=canary.canary_percent if canary else 0,
        )


class ReleaseCatalog:
    """A release's configuration, from the shared cache. A release never
    changes, so its entry never goes stale."""

    CONFIG = TypeAdapter(AgentConfig)

    def __init__(self, database: Database, cache: CacheLayer, settings: Settings) -> None:
        self._database = database
        self._cache = cache
        self._settings = settings
        self.resolver = ReleaseResolver(database, cache)

    async def for_call(
        self, organization_id: uuid.UUID, call_id: uuid.UUID
    ) -> tuple[uuid.UUID, AgentConfig]:
        release_id = await self.resolver.resolve(organization_id, call_id)
        return release_id, await self.config(organization_id, release_id)

    async def config(self, organization_id: uuid.UUID, release_id: uuid.UUID | None) -> AgentConfig:
        """A call made before releases existed has none; it gets the current rows."""
        if release_id is None:
            return await self._current(organization_id)
        key = CacheKeys.build(organization_id, "release", release_id)
        return await self._cache.get(
            key, self.CONFIG, lambda: self._load(organization_id, release_id)
        )

    async def _load(self, organization_id: uuid.UUID, release_id: uuid.UUID) -> AgentConfig:
        async with self._database.session() as session:
            store = ReleaseStore(session)
            release = await store.get(organization_id, release_id)
            items = await store.items(release)
        return ReleaseItems.to_config(organization_id, release.version, items)

    async def _current(self, organization_id: uuid.UUID) -> AgentConfig:
        settings = self._settings
        async with self._database.session() as session:
            return await AgentConfigService(
                session, settings.knowledge.injection_tokens, settings.detection, settings.latency
            ).load(organization_id)


class ReleasePromoter:
    """draft → staged: the prompt compiles in its budget, every tool resolves,
    every collection version passed its gate. staged → live: the suite did not
    block it. A release goes live as a canary if another is stable."""

    def __init__(self, session: AsyncSession, registry: ToolRegistry, settings: Settings) -> None:
        self._session = session
        self._registry = registry
        self._settings = settings

    async def stage(self, release: AgentRelease) -> AgentRelease:
        self._expect(release, ReleaseStatus.DRAFT)
        items = await ReleaseStore(self._session).items(release)
        config = ReleaseItems.to_config(release.organization_id, release.version, items)
        errors = await self._staging_errors(config)
        if errors:
            raise Conflict("The release cannot be staged", errors=errors)
        release.status = ReleaseStatus.STAGED
        return release

    async def go_live(self, release: AgentRelease) -> AgentRelease:
        self._expect(release, ReleaseStatus.STAGED)
        suite = await self._session.scalar(
            select(SuiteResult)
            .where(SuiteResult.release_id == release.id, SuiteResult.decision.is_not(None))
            .order_by(SuiteResult.created_at.desc())
            .limit(1)
        )
        if suite is None or suite.decision == GateDecision.BLOCK:
            raise Conflict(
                "The release has no suite result that allows it",
                decision=suite.decision if suite else None,
            )
        live = await ReleaseStore(self._session).live(release.organization_id)
        if any(r.canary_percent < 100 for r in live):
            raise Conflict("Another canary is live; finish or roll it back first")
        release.status, release.live_at = ReleaseStatus.LIVE, datetime.now(UTC)
        release.canary_percent = self._settings.releases.canary_percent if live else 100
        release.notes = [*release.notes, *suite.reasons]
        await self._changed(release)
        return release

    async def complete(self, release: AgentRelease) -> AgentRelease:
        """The canary becomes stable, for every call."""
        self._expect(release, ReleaseStatus.LIVE)
        for other in await ReleaseStore(self._session).live(release.organization_id):
            if other.id != release.id:
                self._retire(other)
        release.canary_percent = 100
        await self._changed(release)
        return release

    async def rollback(self, release: AgentRelease) -> AgentRelease:
        """A canary is simply retired. A stable release gives way to the one that
        was stable before it."""
        self._expect(release, ReleaseStatus.LIVE)
        was_canary = release.canary_percent < 100
        self._retire(release)
        if not was_canary:
            previous = await self._session.scalar(
                select(AgentRelease)
                .where(
                    AgentRelease.organization_id == release.organization_id,
                    AgentRelease.id != release.id,
                    AgentRelease.live_at.is_not(None),
                    AgentRelease.status == ReleaseStatus.RETIRED,
                )
                .order_by(AgentRelease.live_at.desc())
                .limit(1)
            )
            if previous is None:
                raise Conflict("There is no earlier release to go back to")
            previous.status, previous.canary_percent, previous.retired_at = (
                ReleaseStatus.LIVE,
                100,
                None,
            )
        await self._changed(release)
        return release

    async def _staging_errors(self, config: AgentConfig) -> list[str]:
        errors = []
        for binding in config.tools:
            if self._registry.get(binding.implementation) is None:
                errors.append(
                    f"Tool '{binding.key}' has no implementation '{binding.implementation}'."
                )
        for collection in config.collections:
            passed = await self._session.scalar(
                select(KnowledgePublish.id).where(
                    KnowledgePublish.collection_id == collection.id,
                    KnowledgePublish.version == collection.active_version,
                    KnowledgePublish.status == "published",
                )
            )
            if passed is None:
                errors.append(
                    f"Collection '{collection.key}' v{collection.active_version} "
                    "did not pass its gate."
                )
        compiled = PromptCompiler(self._settings.prompt).compile(
            PromptInputs(pack=await self._sample_pack(config), collections=config.collections)
        )
        errors += [
            f"Prompt section '{s.name}' is over its budget: {s.tokens} of {s.budget} tokens."
            for s in compiled.sections
            if s.over_budget
        ]
        return errors

    async def _sample_pack(self, config: AgentConfig) -> ContextPack:
        organization = await self._session.get(Organization, config.organization_id)
        return ContextPack(
            call_id=uuid.uuid4(),
            identity_level=IdentityLevel.VERIFIED,
            organization=OrganizationContext(
                id=organization.id,
                name=organization.name,
                timezone=organization.timezone,
                persona_name=organization.persona_name,
                persona_instructions=organization.persona_instructions,
            ),
            caller=CallerContext(
                id=uuid.uuid4(),
                full_name="Sample Caller",
                locale="en-GB",
                timezone=organization.timezone,
            ),
        )

    async def _changed(self, release: AgentRelease) -> None:
        """The caller drops the cached live set after the commit, never before:
        a call in between would cache the old set for a minute."""
        await self._session.flush()
        logger.info(
            "release v{} for {} is {} at {}%",
            release.version,
            release.organization_id,
            release.status,
            release.canary_percent,
        )

    @staticmethod
    def _retire(release: AgentRelease) -> None:
        release.status, release.canary_percent = ReleaseStatus.RETIRED, 0
        release.retired_at = datetime.now(UTC)

    @staticmethod
    def _expect(release: AgentRelease, status: ReleaseStatus) -> None:
        if release.status != status:
            raise Conflict(f"The release is {release.status}, not {status}")
