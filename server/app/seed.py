"""Demo data: uv run python -m app.seed

Idempotent, and deliberately two unrelated services under one organization —
the agent is not written around either of them.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select

from app.core.container import Container
from app.core.enums import IdentityLevel, LatencyClass, Priority, ReleaseStatus, TaskStatus
from app.core.logging import LogConfig
from app.eval.suite import SuiteRunner
from app.modules.accounts.models import Organization, Staff, User
from app.modules.catalog.models import Enrollment, Service
from app.modules.config.schemas import BindingWrite, RouteWrite, SkillWrite, SlotKeyWrite
from app.modules.config.service import (
    RouteService,
    SkillService,
    SlotKeyService,
    ToolBindingService,
)
from app.modules.evaluation.models import SuiteResult
from app.modules.knowledge.ingestion import PublisherFactory
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
from app.modules.releases.service import (
    ReleaseBuilder,
    ReleaseCatalog,
    ReleasePromoter,
    ReleaseStore,
)
from app.modules.work.models import Task
from app.modules.work.utils import TaskReference


@dataclass(frozen=True)
class ServiceBlueprint:
    key: str
    name: str
    description: str
    instructions: str


@dataclass(frozen=True)
class StaffBlueprint:
    full_name: str
    team: str


@dataclass(frozen=True)
class TaskBlueprint:
    title: str
    details: str
    status: TaskStatus
    service_key: str
    assignee: str | None = None
    due_in_days: int | None = None


@dataclass(frozen=True)
class UserBlueprint:
    full_name: str
    email: str
    phone: str
    profile: dict[str, Any]
    service_keys: list[str]
    tasks: list[TaskBlueprint] = field(default_factory=list)


@dataclass(frozen=True)
class OrganizationBlueprint:
    slug: str
    name: str
    timezone: str
    persona_name: str
    persona_instructions: str
    services: list[ServiceBlueprint]
    staff: list[StaffBlueprint]
    users: list[UserBlueprint]


class DemoData:
    @classmethod
    def organization(cls) -> OrganizationBlueprint:
        return OrganizationBlueprint(
            slug="northwind",
            name="Northwind",
            timezone="Europe/London",
            persona_name="Alex",
            persona_instructions=(
                "Northwind sells home appliances and runs the repair service for them. "
                "You handle both. You can explain status and answer questions; you cannot "
                "take payment or change a price."
            ),
            services=[
                ServiceBlueprint(
                    key="repairs",
                    name="Appliance Repair",
                    description="Engineer visits for appliances bought from Northwind.",
                    instructions=(
                        "Visits are two-hour windows. If the caller wants to move one, "
                        "take the preference and say the scheduling team will confirm."
                    ),
                ),
                ServiceBlueprint(
                    key="care-plan",
                    name="Northwind Care Plan",
                    description="Monthly cover: parts, labour and an annual service.",
                    instructions=(
                        "Cover starts 14 days after sign-up. Never promise a claim is "
                        "approved; say it goes to the claims team."
                    ),
                ),
            ],
            staff=[
                StaffBlueprint(full_name="Sam Okoye", team="Field Engineers"),
                StaffBlueprint(full_name="Priya Raman", team="Claims"),
            ],
            users=[
                UserBlueprint(
                    full_name="Daniel Whitfield",
                    email="daniel.whitfield@example.com",
                    phone="+447700900123",
                    profile={
                        "address": "14 Rosemount Road, Manchester M20 3TF",
                        "customer_since": "2023",
                        "preferred_contact": "phone",
                        "plan": "legacy",
                    },
                    service_keys=["repairs", "care-plan"],
                    tasks=[
                        TaskBlueprint(
                            title="Washing machine leaking from the door seal",
                            details=(
                                "Engineer visited, ordered a replacement door seal. "
                                "Waiting on the part before a second visit."
                            ),
                            status=TaskStatus.IN_PROGRESS,
                            service_key="repairs",
                            assignee="Sam Okoye",
                            due_in_days=3,
                        ),
                        TaskBlueprint(
                            title="Care Plan claim for the oven fan",
                            details="Claim submitted, waiting on the engineer report.",
                            status=TaskStatus.BLOCKED,
                            service_key="care-plan",
                            assignee="Priya Raman",
                        ),
                    ],
                ),
                UserBlueprint(
                    full_name="Aisha Bello",
                    email="aisha.bello@example.com",
                    phone="+447700900456",
                    profile={"address": "3 Kiln Court, Leeds LS1 4DT", "customer_since": "2026"},
                    service_keys=["repairs"],
                    tasks=[
                        TaskBlueprint(
                            title="Fridge running warm",
                            details="Reported this morning, not yet assigned.",
                            status=TaskStatus.OPEN,
                            service_key="repairs",
                        )
                    ],
                ),
            ],
        )


class DemoSeeder:
    def __init__(self, session) -> None:
        self._session = session

    async def run(self, blueprint: OrganizationBlueprint) -> uuid.UUID:
        existing = await self._session.execute(
            select(Organization).where(Organization.slug == blueprint.slug)
        )
        if (organization := existing.scalar_one_or_none()) is not None:
            logger.info("organization {} already seeded", blueprint.slug)
            return organization.id

        organization = Organization(
            slug=blueprint.slug,
            name=blueprint.name,
            timezone=blueprint.timezone,
            persona_name=blueprint.persona_name,
            persona_instructions=blueprint.persona_instructions,
        )
        self._session.add(organization)
        await self._session.flush()

        services = await self._services(organization, blueprint.services)
        staff = await self._staff(organization, blueprint.staff)
        for user_blueprint in blueprint.users:
            await self._user(organization, user_blueprint, services, staff)

        logger.info("seeded {} ({})", blueprint.name, organization.id)
        return organization.id

    async def _services(self, organization: Organization, blueprints) -> dict[str, Service]:
        services = {
            blueprint.key: Service(
                organization_id=organization.id,
                key=blueprint.key,
                name=blueprint.name,
                description=blueprint.description,
                instructions=blueprint.instructions,
            )
            for blueprint in blueprints
        }
        self._session.add_all(services.values())
        await self._session.flush()
        return services

    async def _staff(self, organization: Organization, blueprints) -> dict[str, Staff]:
        staff = {
            blueprint.full_name: Staff(
                organization_id=organization.id,
                full_name=blueprint.full_name,
                team=blueprint.team,
            )
            for blueprint in blueprints
        }
        self._session.add_all(staff.values())
        await self._session.flush()
        return staff

    async def _user(
        self,
        organization: Organization,
        blueprint: UserBlueprint,
        services: dict[str, Service],
        staff: dict[str, Staff],
    ) -> None:
        user = User(
            organization_id=organization.id,
            full_name=blueprint.full_name,
            email=blueprint.email,
            phone=blueprint.phone,
            timezone=organization.timezone,
            profile=blueprint.profile,
        )
        self._session.add(user)
        await self._session.flush()

        self._session.add_all(
            Enrollment(
                organization_id=organization.id,
                user_id=user.id,
                service_id=services[key].id,
            )
            for key in blueprint.service_keys
        )
        self._session.add_all(
            Task(
                organization_id=organization.id,
                reference=TaskReference.generate(),
                user_id=user.id,
                service_id=services[task.service_key].id,
                assignee_id=staff[task.assignee].id if task.assignee else None,
                title=task.title,
                details=task.details,
                status=task.status,
                due_at=(
                    datetime.now(UTC) + timedelta(days=task.due_in_days)
                    if task.due_in_days
                    else None
                ),
            )
            for task in blueprint.tasks
        )
        await self._session.flush()
        logger.info("  user {} ({})", user.full_name, user.id)


@dataclass(frozen=True)
class SkillBlueprint:
    skill: SkillWrite
    system: bool = False
    service_key: str | None = None


class DemoSkills:
    @classmethod
    def all(cls) -> list[SkillBlueprint]:
        return [
            SkillBlueprint(
                system=True,
                skill=SkillWrite(
                    key="escalate_to_human",
                    name="Hand over to a person",
                    trigger=(
                        "When the caller asks for a person, is upset, or needs something "
                        "you cannot do. Never refuse a request for a person."
                    ),
                    goal="The caller knows a person will help, and what happens next.",
                    tool_keys=["escalate_to_human"],
                    steps=(
                        "1. Do not ask why they want a person.\n"
                        "2. Call escalate_to_human. It tells the caller what happens next, so "
                        "say nothing yourself."
                    ),
                    position=100,
                ),
            ),
            SkillBlueprint(
                system=True,
                skill=SkillWrite(
                    key="general_inquiry",
                    name="General question",
                    trigger=(
                        "When the caller has a general question, or it is not yet clear "
                        "what they need. Use it until another skill fits."
                    ),
                    goal="You know what the caller needs, or you answered their question.",
                    tool_keys=["request_callback", "create_work_item", "end_call"],
                    steps=(
                        "1. Ask one short question to find out what they need.\n"
                        "2. If another skill on your list fits, load it.\n"
                        "3. If nothing fits, offer to pass them to a colleague."
                    ),
                    position=90,
                ),
            ),
            SkillBlueprint(
                service_key="repairs",
                skill=SkillWrite(
                    key="book_visit",
                    name="Book an engineer visit",
                    trigger=(
                        "When the caller wants an engineer to come out, or wants to move "
                        "or cancel a visit. Not for a part that has already shipped."
                    ),
                    goal="The caller has a visit time they agreed to, or a clear next step.",
                    steps=(
                        "1. Confirm the address you have on file.\n"
                        "2. Ask which day and part of the day suits them.\n"
                        "3. Log it with request_callback, with the day and part of day as "
                        "the preferred time.\n"
                        "4. Read the reference back and say a colleague will confirm the time."
                    ),
                    tool_keys=["request_callback"],
                    guardrails="Never invent a time. Never quote a price.",
                    done_when="The caller agreed a day and part of the day.",
                    escalate_if="The caller needs a visit today, or it is an emergency.",
                    knowledge_key="repairs",
                    position=10,
                ),
            ),
            SkillBlueprint(
                skill=SkillWrite(
                    key="check_order",
                    name="Check an order or job",
                    trigger=(
                        "When the caller asks where an order is, if a part has shipped, "
                        "or what is happening with an existing job. Not for a new visit."
                    ),
                    goal="The caller knows the current status and the next step.",
                    steps=(
                        "1. Find the job in their current work, or use get_open_work.\n"
                        "2. For detail on one job, use get_work_item with its reference.\n"
                        "3. Say the status and who is handling it.\n"
                        "4. If they want to add information, use add_work_note."
                    ),
                    tool_keys=["get_open_work", "get_work_item", "add_work_note"],
                    guardrails="Never guess a status. Never guess a reference.",
                    done_when="The caller heard the status of the job they asked about.",
                    position=20,
                ),
            ),
        ]


class SkillSeeder:
    """Keyed by skill key: creates missing skills and updates changed ones, so
    it also fills databases seeded before skills existed."""

    def __init__(self, session) -> None:
        self._session = session

    async def run(self, organization_id: uuid.UUID) -> None:
        service = SkillService(self._session)
        existing = {skill.key: skill for skill in await service.list(organization_id)}
        services = await self._services(organization_id)
        for blueprint in DemoSkills.all():
            data = blueprint.skill
            if blueprint.service_key:
                data = data.model_copy(update={"service_id": services[blueprint.service_key]})
            if data.key not in existing:
                await service.create(organization_id, data, system=blueprint.system)
                logger.info("  skill {} created", data.key)
            elif self._differs(existing[data.key], data):
                await service.update(organization_id, data.key, data)
                logger.info("  skill {} updated", data.key)

    @staticmethod
    def _differs(skill, data: SkillWrite) -> bool:
        return any(getattr(skill, field) != value for field, value in data.model_dump().items())

    async def _services(self, organization_id: uuid.UUID) -> dict[str, uuid.UUID]:
        result = await self._session.execute(
            select(Service.key, Service.id).where(Service.organization_id == organization_id)
        )
        return dict(result.all())


class DemoTools:
    @classmethod
    def all(cls) -> list[BindingWrite]:
        reference = {
            "type": "string",
            "description": "The job reference, for example 7KQ3MA. Never guess it.",
        }
        return [
            BindingWrite(
                key="get_open_work",
                implementation="get_open_work",
                description=(
                    "List the caller's open jobs, orders and requests. Use it when they ask "
                    "about their work and it is not already in what you know."
                ),
                latency_class=LatencyClass.A,
                required_level=IdentityLevel.RECOGNISED,
                timeout_ms=1500,
                fallback_text="The job list is not available now. Offer a callback instead.",
            ),
            BindingWrite(
                key="get_work_item",
                implementation="get_work_item",
                description=(
                    "Get the full detail of one job by its reference. Use it before you add "
                    "a note to a job. Never guess a reference; ask the caller to read it out."
                ),
                parameters={
                    "type": "object",
                    "properties": {"reference": reference},
                    "required": ["reference"],
                },
                latency_class=LatencyClass.A,
                required_level=IdentityLevel.RECOGNISED,
                timeout_ms=1500,
                cache_ttl_s=30,
                fallback_text="The job details are not available now. Offer a callback instead.",
            ),
            BindingWrite(
                key="add_work_note",
                implementation="add_work_note",
                description=(
                    "Add information from the caller to one of their jobs. Look up the job "
                    "with get_work_item first. Not for new requests; use request_callback."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reference": reference,
                        "note": {"type": "string", "description": "What the caller said."},
                    },
                    "required": ["reference", "note"],
                },
                latency_class=LatencyClass.C,
                required_level=IdentityLevel.RECOGNISED,
                invalidates=["get_work_item", "get_open_work"],
                fallback_text="The note could not be saved. Offer a callback instead.",
            ),
            BindingWrite(
                key="request_callback",
                implementation="request_callback",
                description=(
                    "Ask for a person to call the caller back. Use it when you cannot finish "
                    "the request yourself, or they ask for a person."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "A few words on the subject."},
                        "reason": {
                            "type": "string",
                            "description": "What they need, in a sentence.",
                        },
                        "preferred_time": {"type": "string", "description": "When suits them."},
                    },
                    "required": ["topic", "reason"],
                },
                latency_class=LatencyClass.C,
                invalidates=["get_open_work"],
                fallback_text="The callback could not be logged. Give the office phone number.",
            ),
            BindingWrite(
                key="search_knowledge",
                implementation="search_knowledge",
                description=(
                    "Search the company's own knowledge for policies, cover, returns and how "
                    "things work. Give a short, complete question. Not for the status of the "
                    "caller's own jobs; use get_open_work for that."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question, in a sentence."}
                    },
                    "required": ["query"],
                },
                latency_class=LatencyClass.A,
                timeout_ms=1500,
                fallback_text="The knowledge is not available now. Offer a callback.",
            ),
            BindingWrite(
                key="create_work_item",
                implementation="create_work_item",
                description=(
                    "Pass a task to the team when a person must do it: a complaint, a refund "
                    "request, a repair or a message. Give the type and what is needed. Not for "
                    "a callback; use request_callback."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "message, repair, complaint or refund_request.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the person must do, with every detail.",
                        },
                    },
                    "required": ["type", "description"],
                },
                latency_class=LatencyClass.C,
                invalidates=["get_open_work"],
                fallback_text="The request could not be passed on. Give the office phone number.",
            ),
            BindingWrite(
                key="escalate_to_human",
                implementation="escalate_to_human",
                description=(
                    "Hand the call to a person. Use it when the caller asks for a person, or "
                    "when you cannot help. Do not ask why first."
                ),
                latency_class=LatencyClass.A,
                fallback_text="Say you will pass the details to a colleague.",
            ),
            BindingWrite(
                key="get_call_history",
                implementation="get_call_history",
                description=(
                    "Get short notes on the caller's earlier calls, newest first. Use it when "
                    "they refer to an earlier call and it is not in what you know."
                ),
                latency_class=LatencyClass.A,
                required_level=IdentityLevel.RECOGNISED,
                timeout_ms=1500,
                fallback_text="Earlier calls are not available now. Ask the caller to remind you.",
            ),
            BindingWrite(
                key="end_call",
                implementation="end_call",
                description=(
                    "End the call. When the caller says goodbye or that they need nothing "
                    "else, say goodbye and call this tool in the same reply."
                ),
                latency_class=LatencyClass.A,
                fallback_text="Say goodbye; the call will end.",
            ),
        ]


class ToolSeeder:
    def __init__(self, session, registry) -> None:
        self._service = ToolBindingService(session, registry)

    async def run(self, organization_id: uuid.UUID) -> None:
        existing = {b.key: b for b in await self._service.list(organization_id)}
        for data in DemoTools.all():
            if data.key not in existing:
                await self._service.save(organization_id, data)
                logger.info("  tool {} created", data.key)
            elif SkillSeeder._differs(existing[data.key], data):
                await self._service.save(organization_id, data, key=data.key)
                logger.info("  tool {} updated", data.key)


class DemoEscalation:
    PHONE = "0161 496 0000"
    SAFETY = "If you or anyone with you is in danger, please hang up and call nine nine nine now."

    @staticmethod
    def routes() -> list[RouteWrite]:
        return [
            RouteWrite(work_item_type="handoff", description="Live transfers.", queue="front_desk"),
            RouteWrite(
                work_item_type="callback",
                description="A person calls the customer back.",
                queue="callbacks",
            ),
            RouteWrite(
                work_item_type="message",
                description="A message for the team.",
                queue="front_desk",
            ),
            RouteWrite(
                work_item_type="repair",
                description="A repair that needs an engineer.",
                queue="field_engineers",
            ),
            RouteWrite(
                work_item_type="complaint",
                description="A complaint about the service.",
                queue="customer_care",
                priority=Priority.HIGH,
            ),
            RouteWrite(
                work_item_type="refund_request",
                description="A request for money back.",
                queue="billing",
            ),
        ]


class EscalationSeeder:
    """Routes, and the phone number and safety message that fixed speech uses.
    Existing values are kept."""

    def __init__(self, session) -> None:
        self._session = session

    async def run(self, organization_id: uuid.UUID) -> None:
        organization = await self._session.get(Organization, organization_id)
        organization.phone = organization.phone or DemoEscalation.PHONE
        organization.safety_message = organization.safety_message or DemoEscalation.SAFETY
        service = RouteService(self._session)
        if not await service.list(organization_id):
            await service.replace(organization_id, DemoEscalation.routes())
            logger.info("  escalation routes saved")


class ReleaseSeeder:
    """The tenant's first release goes live here, so calls have one. Later
    changes get a draft; promoting it is a decision for a person, through the
    API, after the suite."""

    def __init__(self, container) -> None:
        self._container = container

    async def run(self, organization_id: uuid.UUID) -> None:
        container, settings = self._container, self._container.settings
        catalog = ReleaseCatalog(container.database, container.cache, settings)
        async with container.database.session() as session:
            release = await ReleaseBuilder(session, settings).build(organization_id, "seed")
            if release.status != ReleaseStatus.DRAFT:
                logger.info("  release v{} is {}", release.version, release.status)
                return
            if await ReleaseStore(session).live(organization_id):
                logger.info(
                    "  release v{} built as a draft; promote it through the API", release.version
                )
                return
            await ReleasePromoter(session, container.registry, settings).stage(release)
            suite = SuiteResult(organization_id=organization_id, release_id=release.id)
            session.add(suite)
        await SuiteRunner(container).run(suite.id)
        async with container.database.session() as session:
            release = await ReleaseStore(session).get(organization_id, release.id)
            await ReleasePromoter(session, container.registry, settings).go_live(release)
        await catalog.resolver.invalidate(organization_id)
        logger.info("  release v{} is live", release.version)


class DemoMemory:
    @staticmethod
    def slot_keys() -> list[SlotKeyWrite]:
        return [
            SlotKeyWrite(
                key="times_to_avoid",
                description=(
                    "Times of day or days of the week that the caller says do not suit them "
                    "for visits or calls, as a general rule."
                ),
                example="afternoons",
            ),
            SlotKeyWrite(
                key="preferred_visit_time",
                description=(
                    "The part of the day the caller prefers for engineer visits, as a general "
                    "rule. Not the time of one booking."
                ),
                example="mornings",
            ),
            SlotKeyWrite(
                key="access_notes",
                description=(
                    "How an engineer gets into the home: a side gate, a key safe, a pet to "
                    "keep in, or where to park."
                ),
                example="Use the side gate; the dog is kept inside",
            ),
            SlotKeyWrite(
                key="contact_channel",
                description="How the caller wants to be contacted, if they say so.",
                example="text message",
                spoken=False,
            ),
        ]


class MemorySeeder:
    def __init__(self, session) -> None:
        self._service = SlotKeyService(session)

    async def run(self, organization_id: uuid.UUID) -> None:
        existing = {k.key: k for k in await self._service.list(organization_id)}
        for data in DemoMemory.slot_keys():
            if data.key not in existing or SkillSeeder._differs(existing[data.key], data):
                await self._service.save(organization_id, data)
                logger.info("  memory slot {} saved", data.key)


class DemoKnowledge:
    DATA = Path(__file__).parent / "seed_data"

    @classmethod
    def collections(cls) -> list[tuple[CollectionWrite, list, list, list]]:
        """Each collection with its documents, cards and retrieval tests."""
        general = CollectionWrite(
            key="general", name="Northwind help", is_default=True, filter_keys=["plan"]
        )
        repairs = CollectionWrite(key="repairs", name="Engineer visits")
        return [
            (general, cls._general_documents(), cls._general_cards(), cls._general_tests()),
            (
                repairs,
                [
                    cls._document("visits", "Engineer visits"),
                    cls._document("error_codes", "Appliance error codes"),
                ],
                [],
                cls._repairs_tests(),
            ),
        ]

    @classmethod
    def _document(cls, key: str, title: str, **metadata: str) -> DocumentWrite:
        return DocumentWrite(
            key=key,
            title=title,
            source_type="upload",
            content=(cls.DATA / f"{key}.md").read_text(),
            metadata=metadata,
        )

    @classmethod
    def _general_documents(cls) -> list[DocumentWrite]:
        return [
            cls._document("returns_warranty", "Returns and Warranty"),
            cls._document("care_plan_current", "Northwind Care Plan (current)", plan="current"),
            cls._document("care_plan_legacy", "Northwind Care Plan (legacy)", plan="legacy"),
        ]

    @staticmethod
    def _general_cards() -> list[CardWrite]:
        return [
            CardWrite(
                key="contact_hours",
                questions=[
                    "What are your opening hours?",
                    "When can I call you?",
                    "Are you open at the weekend?",
                    "What time does the office close?",
                ],
                answer="The phone line is open Monday to Friday from 8 in the morning to 6 in "
                "the evening, and on Saturday from 9 to 1.",
            ),
            CardWrite(
                key="payment_methods",
                questions=[
                    "How can I pay?",
                    "Do you take card payments?",
                    "Can I pay by bank transfer?",
                ],
                answer="We accept debit cards, credit cards and bank transfer. "
                "We do not accept cash.",
            ),
            CardWrite(
                key="complaint",
                questions=[
                    "How do I make a complaint?",
                    "I want to complain.",
                    "Who do I complain to?",
                ],
                answer="You can complain by phone, or by email to complaints@northwind.example. "
                "We reply within 5 working days.",
            ),
        ]

    @staticmethod
    def _expect(question: str, heading: str, **filters: str) -> RetrievalTestWrite:
        return RetrievalTestWrite(question=question, expected_heading=heading, filters=filters)

    @classmethod
    def _general_tests(cls) -> list[RetrievalTestWrite]:
        heading = cls._expect
        return [
            heading("Can I send back a fridge I bought last week?", "Return an appliance"),
            heading("How long is the guarantee on a washing machine?", "Warranty cover"),
            heading("Am I covered if I dropped my microwave?", "Warranty cover"),
            heading("Can I extend my warranty?", "Extended warranty"),
            heading(
                "My oven is used but broken, can I return it?", "Return a used or damaged appliance"
            ),
            heading(
                "When does my care plan cover start?",
                "When cover starts (current plan)",
                plan="current",
            ),
            heading(
                "How do I cancel the care plan?",
                "Cancel the Care Plan (legacy plan)",
                plan="legacy",
            ),
            heading(
                "How do I cancel the care plan?",
                "Cancel the Care Plan (current plan)",
                plan="current",
            ),
            heading(
                "What does the care plan cover?",
                "What the Care Plan covers (legacy plan)",
                plan="legacy",
            ),
            heading(
                "How do I make a claim on my cover?", "Make a claim (current plan)", plan="current"
            ),
            RetrievalTestWrite(
                question="What time are you open?", expected_card_key="contact_hours"
            ),
            RetrievalTestWrite(
                question="Can I pay with a credit card?", expected_card_key="payment_methods"
            ),
            RetrievalTestWrite(
                question="I would like to make a complaint", expected_card_key="complaint"
            ),
            RetrievalTestWrite(question="Do you sell televisions?", expect_refusal=True),
            RetrievalTestWrite(question="What is the capital of France?", expect_refusal=True),
        ]

    @classmethod
    def _repairs_tests(cls) -> list[RetrievalTestWrite]:
        heading = cls._expect
        return [
            heading("How long is the visit window?", "Visit windows"),
            heading(
                "Will the engineer phone before arriving?", "The engineer phones before arriving"
            ),
            heading("Can I change my appointment?", "Move or cancel a visit"),
            heading("Is there a fee to cancel the engineer?", "Move or cancel a visit"),
            heading("What should I do before the engineer comes?", "What to do before the visit"),
            heading("Where do I find the model number?", "What to do before the visit"),
            heading("Do I have to pay for the second visit?", "Second visits"),
            heading("What happens if the engineer needs to order a part?", "Second visits"),
            heading(
                "My washer is showing E24, what does that mean?",
                "Error code E24 (washing machines)",
            ),
            heading("The dishwasher says F05", "Error code F05 (dishwashers)"),
            RetrievalTestWrite(question="Can you recommend a new dishwasher?", expect_refusal=True),
            RetrievalTestWrite(question="What is the weather tomorrow?", expect_refusal=True),
        ]


class KnowledgeSeeder:
    """Writes the demo collections, then publishes each one that has no live
    version. Later changes are published through the API."""

    def __init__(self, container) -> None:
        self._container = container

    async def run(self, organization_id: uuid.UUID) -> None:
        database = self._container.database
        for data, documents, cards, tests in DemoKnowledge.collections():
            async with database.session() as session:
                collection = await CollectionService(session).save(organization_id, data)
                for document in documents:
                    await DocumentService(session).save(collection, document)
                for card in cards:
                    await CardService(session).save(collection, card)
                known = {t.question for t in await RetrievalTestService(session).list(collection)}
                for test in tests:
                    if test.question not in known:
                        await RetrievalTestService(session).add(collection, test)
                live = collection.active_version is not None
                if not live:
                    run = KnowledgePublish(
                        organization_id=organization_id, collection_id=collection.id, version=1
                    )
                    session.add(run)
                    await session.flush()
            if live:
                logger.info("  knowledge {} already published", data.key)
                continue
            report = await PublisherFactory.build(self._container).publish(run.id)
            async with database.session() as session:
                stored = await session.get(KnowledgePublish, run.id)
                stored.status = "published" if report["passed"] else "failed"
                stored.report = report
            gate = report.get("gate", {})
            logger.info(
                "  knowledge {}: {} (gate {} of {} tests, {} warnings)",
                data.key,
                stored.status,
                gate.get("pass_rate"),
                gate.get("tests"),
                len(report["warnings"]),
            )
            for failure in gate.get("failures", []):
                logger.info("    gate failure: {}", failure)


async def main() -> None:
    LogConfig.install()
    container = Container()
    container.settings.livekit.export_env()
    try:
        async with container.database.session() as session:
            organization_id = await DemoSeeder(session).run(DemoData.organization())
        async with container.database.session() as session:
            await ToolSeeder(session, container.registry).run(organization_id)
        async with container.database.session() as session:
            await SkillSeeder(session).run(organization_id)
        async with container.database.session() as session:
            await MemorySeeder(session).run(organization_id)
        async with container.database.session() as session:
            await EscalationSeeder(session).run(organization_id)
        await ReleaseSeeder(container).run(organization_id)
        await KnowledgeSeeder(container).run(organization_id)
    finally:
        await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
