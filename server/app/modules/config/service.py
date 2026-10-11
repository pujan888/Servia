from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import DetectionSettings, LatencySettings
from app.core.enums import FlagType, LatencyClass, Optimisation, SideEffect
from app.core.errors import Conflict, Invalid, NotFound
from app.core.tokens import TokenCounter
from app.modules.accounts.models import Organization
from app.modules.catalog.models import Service
from app.modules.config.models import (
    DetectionSetting,
    EscalationRoute,
    MemorySlotKey,
    ModelProfile,
    OptimisationFlag,
    Skill,
    ToolBinding,
)
from app.modules.config.schemas import (
    AgentConfig,
    BindingSpec,
    BindingWrite,
    CollectionSpec,
    DetectionSpec,
    ModelProfileWrite,
    OptimisationWrite,
    ProfileSpec,
    RouteSpec,
    RouteWrite,
    SkillSpec,
    SkillWrite,
    SlotKeySpec,
    SlotKeyWrite,
    ValidationReport,
)
from app.modules.config.utils import SchemaGuard
from app.modules.knowledge.models import CardVariant, KnowledgeChunk, KnowledgeCollection

if TYPE_CHECKING:
    from app.agent.tools.registry import ToolRegistry


class ConfigVersionService:
    """One counter per tenant to track the version of the config. Incremented on each change."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bump(self, organization_id: uuid.UUID) -> None:
        # increases the config version for the current organization
        await self._session.execute(
            update(Organization)
            .where(Organization.id == organization_id)
            .values(config_version=Organization.config_version + 1)
        )

    async def current(self, organization_id: uuid.UUID) -> int:
        # reads the current config version for the current organization
        result = await self._session.execute(
            select(Organization.config_version).where(Organization.id == organization_id)
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise NotFound("Organization not found", organization_id=str(organization_id))
        return version


class SkillValidator:
    # The model chooses by trigger alone; it must say when, and when not, to use the skill.
    MIN_TRIGGER_WORDS = 6

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def validate(self, organization_id: uuid.UUID, data: SkillWrite) -> ValidationReport:
        """Validates a skill spec for an organization. Returns a ValidationReport with errors."""
        report = ValidationReport()
        if len(data.trigger.split()) < self.MIN_TRIGGER_WORDS:
            report.errors.append(
                "The trigger is too short. Say when to use this skill and when not to use it."
            )
        if not data.steps.strip():
            report.errors.append("The steps are empty.")
        if data.service_id is not None and not await self._service_belongs(
            organization_id, data.service_id
        ):
            report.errors.append("The service belongs to another organization.")
        return report

    async def _service_belongs(self, organization_id: uuid.UUID, service_id: uuid.UUID) -> bool:
        # checks whether a Service belongs to the same organization as the Skill being created.
        service = await self._session.get(Service, service_id)
        return service is not None and service.organization_id == organization_id


class SkillService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[Skill]:
        # get all skill for the organization
        result = await self._session.execute(
            select(Skill)
            .where(Skill.organization_id == organization_id)
            .order_by(Skill.position, Skill.key)
        )
        return result.scalars().all()

    async def get(self, organization_id: uuid.UUID, key: str) -> Skill:
        """get one skill by key for the organization"""
        result = await self._session.execute(
            select(Skill).where(Skill.organization_id == organization_id, Skill.key == key)
        )
        skill = result.scalar_one_or_none()
        if skill is None:
            raise NotFound("Skill not found", key=key)
        return skill

    async def create(
        self, organization_id: uuid.UUID, data: SkillWrite, system: bool = False
    ) -> Skill:
        # create a new skill for the organization
        await self._check(organization_id, data)
        skill = Skill(organization_id=organization_id, system=system, **data.model_dump())
        self._session.add(skill)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return skill

    async def update(self, organization_id: uuid.UUID, key: str, data: SkillWrite) -> Skill:
        # update and existing skill for the organization
        skill = await self.get(organization_id, key)
        if skill.system and (data.key != skill.key or not data.enabled):
            raise Conflict("A system skill cannot be renamed or disabled", key=key)
        await self._check(organization_id, data)
        for field, value in data.model_dump().items():
            setattr(skill, field, value)
        await ConfigVersionService(self._session).bump(organization_id)
        return skill

    async def _check(self, organization_id: uuid.UUID, data: SkillWrite) -> None:
        # check if the skill is valid and belongs to the organization
        report = await SkillValidator(self._session).validate(organization_id, data)
        if not report.ok:
            raise Invalid("The skill is not valid", errors=report.errors)


class ModelProfileService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, organization_id: uuid.UUID) -> ModelProfile | None:
        return await self._session.get(ModelProfile, organization_id)

    async def save(self, organization_id: uuid.UUID, data: ModelProfileWrite) -> ModelProfile:
        profile = await self.get(organization_id) or ModelProfile(organization_id=organization_id)
        for field, value in data.model_dump().items():
            setattr(profile, field, value)
        self._session.add(profile)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return profile


class ToolBindingService:
    def __init__(self, session: AsyncSession, registry: ToolRegistry) -> None:
        self._session = session
        self._registry = registry

    async def list(self, organization_id: uuid.UUID) -> Sequence[ToolBinding]:
        result = await self._session.execute(
            select(ToolBinding)
            .where(ToolBinding.organization_id == organization_id)
            .order_by(ToolBinding.key)
        )
        return result.scalars().all()

    async def get(self, organization_id: uuid.UUID, key: str) -> ToolBinding:
        result = await self._session.execute(
            select(ToolBinding).where(
                ToolBinding.organization_id == organization_id, ToolBinding.key == key
            )
        )
        binding = result.scalar_one_or_none()
        if binding is None:
            raise NotFound("Tool binding not found", key=key)
        return binding

    async def save(
        self, organization_id: uuid.UUID, data: BindingWrite, key: str | None = None
    ) -> tuple[ToolBinding, ValidationReport]:
        report = await self.validate(organization_id, data)
        if not report.ok:
            raise Invalid("The tool binding is not valid", errors=report.errors)
        binding = (
            await self.get(organization_id, key)
            if key
            else ToolBinding(organization_id=organization_id)
        )
        for field, value in data.model_dump().items():
            setattr(binding, field, value)
        self._session.add(binding)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return binding, report

    async def validate(self, organization_id: uuid.UUID, data: BindingWrite) -> ValidationReport:
        report = ValidationReport()
        implementation = self._registry.get(data.implementation)
        if implementation is None:
            report.errors.append(
                f"No implementation is called '{data.implementation}'. "
                f"Known: {sorted(self._registry.names)}."
            )
        elif (
            implementation.side_effect == SideEffect.WRITE and data.latency_class != LatencyClass.C
        ):
            report.errors.append("A tool that writes must be latency class C.")
        report.errors += SchemaGuard.schema_errors(data.parameters)
        if data.latency_class == LatencyClass.B and not data.filler_category:
            report.warnings.append("A class B tool has no filler category; the default is used.")
        if data.cache_ttl_s and implementation and implementation.side_effect == SideEffect.WRITE:
            report.errors.append("A tool that writes cannot be cached.")
        known = {binding.key for binding in await self.list(organization_id)} | {data.key}
        unknown = sorted(set(data.invalidates) - known)
        if unknown:
            report.warnings.append(f"'invalidates' names tools that do not exist: {unknown}.")
        return report


class SlotKeyService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[MemorySlotKey]:
        result = await self._session.execute(
            select(MemorySlotKey)
            .where(MemorySlotKey.organization_id == organization_id)
            .order_by(MemorySlotKey.key)
        )
        return result.scalars().all()

    async def save(self, organization_id: uuid.UUID, data: SlotKeyWrite) -> MemorySlotKey:
        """Creates the key, or updates the key of the same name."""
        result = await self._session.execute(
            select(MemorySlotKey).where(
                MemorySlotKey.organization_id == organization_id, MemorySlotKey.key == data.key
            )
        )
        slot_key = result.scalar_one_or_none() or MemorySlotKey(organization_id=organization_id)
        for field, value in data.model_dump().items():
            setattr(slot_key, field, value)
        self._session.add(slot_key)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return slot_key


class RouteService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[EscalationRoute]:
        result = await self._session.execute(
            select(EscalationRoute)
            .where(EscalationRoute.organization_id == organization_id)
            .order_by(EscalationRoute.work_item_type)
        )
        return result.scalars().all()

    async def replace(
        self, organization_id: uuid.UUID, routes: list[RouteWrite]
    ) -> Sequence[EscalationRoute]:
        """The whole table at once: types left out are removed."""
        existing = {route.work_item_type: route for route in await self.list(organization_id)}
        for data in routes:
            route = existing.pop(data.work_item_type, None) or EscalationRoute(
                organization_id=organization_id
            )
            for field, value in data.model_dump().items():
                setattr(route, field, value)
            self._session.add(route)
        for route in existing.values():
            await self._session.delete(route)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return await self.list(organization_id)


class DetectionSettingService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[DetectionSetting]:
        result = await self._session.execute(
            select(DetectionSetting)
            .where(DetectionSetting.organization_id == organization_id)
            .order_by(DetectionSetting.flag_type)
        )
        return result.scalars().all()

    async def save(
        self, organization_id: uuid.UUID, flag_type: FlagType, threshold: float, enabled: bool
    ) -> DetectionSetting:
        rows = {row.flag_type: row for row in await self.list(organization_id)}
        row = rows.get(flag_type) or DetectionSetting(
            organization_id=organization_id, flag_type=flag_type
        )
        row.threshold, row.enabled = threshold, enabled
        self._session.add(row)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return row


class OptimisationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[OptimisationFlag]:
        result = await self._session.execute(
            select(OptimisationFlag)
            .where(OptimisationFlag.organization_id == organization_id)
            .order_by(OptimisationFlag.name)
        )
        return result.scalars().all()

    async def save(self, organization_id: uuid.UUID, data: OptimisationWrite) -> OptimisationFlag:
        rows = {row.name: row for row in await self.list(organization_id)}
        row = rows.get(data.name) or OptimisationFlag(
            organization_id=organization_id, name=data.name
        )
        row.percent = data.percent
        self._session.add(row)
        await self._session.flush()
        await ConfigVersionService(self._session).bump(organization_id)
        return row


class AgentConfigService:
    """Reads everything the agent needs about a tenant's configuration, once."""

    def __init__(
        self,
        session: AsyncSession,
        injection_tokens: int = 800,
        detection: DetectionSettings | None = None,
        latency: LatencySettings | None = None,
    ) -> None:
        self._session = session
        self._injection_tokens = injection_tokens
        self._detection = detection or DetectionSettings()
        self._latency = latency or LatencySettings()
        self._counter = TokenCounter()

    async def load(self, organization_id: uuid.UUID) -> AgentConfig:
        version = await ConfigVersionService(self._session).current(organization_id)
        skills = [s for s in await SkillService(self._session).list(organization_id) if s.enabled]
        profile = await ModelProfileService(self._session).get(organization_id)
        result = await self._session.execute(
            select(ToolBinding)
            .where(ToolBinding.organization_id == organization_id, ToolBinding.enabled.is_(True))
            .order_by(ToolBinding.key)
        )
        tools = result.scalars().all()
        return AgentConfig(
            organization_id=organization_id,
            version=version,
            skills=tuple(SkillSpec.model_validate(skill) for skill in skills),
            profile=ProfileSpec.model_validate(profile) if profile else ProfileSpec(),
            tools=tuple(BindingSpec.model_validate(binding) for binding in tools),
            collections=await self._collections(organization_id),
            slot_keys=tuple(
                SlotKeySpec.model_validate(slot_key)
                for slot_key in await SlotKeyService(self._session).list(organization_id)
                if slot_key.enabled
            ),
            routes=tuple(
                RouteSpec.model_validate(route)
                for route in await RouteService(self._session).list(organization_id)
                if route.enabled
            ),
            detection=await self._detection_spec(organization_id),
            optimisations={
                **self._latency.optimisations,
                **{
                    Optimisation(row.name): row.percent
                    for row in await OptimisationService(self._session).list(organization_id)
                },
            },
        )

    async def _detection_spec(self, organization_id: uuid.UUID) -> DetectionSpec:
        rows = await DetectionSettingService(self._session).list(organization_id)
        thresholds = dict(self._detection.thresholds)
        thresholds.update({FlagType(row.flag_type): row.threshold for row in rows})
        return DetectionSpec(
            thresholds=thresholds,
            disabled=frozenset(FlagType(row.flag_type) for row in rows if not row.enabled),
            severe_abuse=self._detection.severe_abuse,
        )

    async def _collections(self, organization_id: uuid.UUID) -> tuple[CollectionSpec, ...]:
        result = await self._session.execute(
            select(KnowledgeCollection)
            .where(
                KnowledgeCollection.organization_id == organization_id,
                KnowledgeCollection.active_version.is_not(None),
            )
            .order_by(KnowledgeCollection.key)
        )
        return tuple([await self._spec(collection) for collection in result.scalars()])

    async def _spec(self, collection: KnowledgeCollection) -> CollectionSpec:
        return CollectionSpec(
            id=collection.id,
            key=collection.key,
            name=collection.name,
            is_default=collection.is_default,
            active_version=collection.active_version,
            filter_keys=tuple(collection.filter_keys),
            injected_text=await self._injection(collection),
        )

    async def _injection(self, collection: KnowledgeCollection) -> str | None:
        """A collection with no documents and few enough cards goes straight into
        the prompt. For a corpus that small, no search is faster than any search."""
        version = collection.active_version
        chunks = await self._session.scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.collection_id == collection.id, KnowledgeChunk.version == version)
        )
        if chunks:
            return None
        result = await self._session.execute(
            select(
                CardVariant.card_key, func.min(CardVariant.question), func.min(CardVariant.answer)
            )
            .where(CardVariant.collection_id == collection.id, CardVariant.version == version)
            .group_by(CardVariant.card_key)
            .order_by(CardVariant.card_key)
        )
        lines = [f"- Q: {question}\n  A: {answer}" for _, question, answer in result.all()]
        if not lines:
            return None
        text = f"Answers from {collection.name}:\n" + "\n".join(lines)
        return text if self._counter.count(text) <= self._injection_tokens else None
