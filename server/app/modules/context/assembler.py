from __future__ import annotations

import uuid
from datetime import timedelta

from loguru import logger

from app.core.config import ContextSettings
from app.core.database import Database
from app.core.enums import IdentityLevel
from app.modules.accounts.models import Organization, User
from app.modules.accounts.service import (
    OrganizationService,
    StaffService,
    UserService,
)
from app.modules.calls.models import Call
from app.modules.calls.service import CallService
from app.modules.catalog.models import Enrollment, Service
from app.modules.catalog.service import EnrollmentService
from app.modules.context.schemas import (
    CallerContext,
    ContextPack,
    OrganizationContext,
    ServiceContext,
    TaskContext,
)
from app.modules.context.utils import DeadlineGather
from app.modules.memory.schemas import LoadedMemory
from app.modules.memory.store import MemoryStore
from app.modules.work.models import Task
from app.modules.work.service import TaskService


class ContextAssembler:
    # Maximum number of context loaders that can run in parallel.
    PARALLEL_LOADERS = 5

    def __init__(self, database: Database, settings: ContextSettings) -> None:
        self._database = database
        self._settings = settings
        self._gather = DeadlineGather(settings.deadline_ms, settings.required_deadline_ms)

    async def assemble(self, call_id: uuid.UUID, identity_level: IdentityLevel) -> ContextPack:
        # Build the complete context pack for a call before the agent starts speaking.
        call = await self._call(call_id)
        result = await self._gather.run(
            required={
                "organization": self._organization(call.organization_id),
                "caller": self._caller(call.user_id),
            },
            optional={
                "services": self._services(call.user_id),
                "tasks": self._tasks(call.user_id),
                "memory": self._memory(call),
            },
        )
        memory: LoadedMemory | None = result.values.get("memory")
        degraded = result.missing + (["memory"] if memory and memory.pending else [])
        if degraded:
            logger.warning("context degraded for call {}: {}", call_id, degraded)

        organization: Organization = result.values["organization"]
        user: User = result.values["caller"]
        services = result.values.get("services", [])
        return ContextPack(
            call_id=call.id,
            identity_level=identity_level,
            organization=self._organization_context(organization),
            caller=self._caller_context(user, organization),
            focus_service=next((s for s in services if s.id == call.service_id), None),
            services=services,
            tasks=result.values.get("tasks", []),
            digest=memory.digest if memory else None,
            degraded=degraded,
        )

    async def _call(self, call_id: uuid.UUID):
        # Load the call record to get the IDs needed by the other context loaders.
        async with self._database.session() as session:
            return await CallService(session).get(call_id)

    async def _organization(self, organization_id: uuid.UUID) -> Organization:
        # Load the organization associated with the caller's call.
        async with self._database.session() as session:
            return await OrganizationService(session).get(organization_id)

    async def _caller(self, user_id: uuid.UUID) -> User:
        # Load the caller's user/profile information.
        async with self._database.session() as session:
            return await UserService(session).get(user_id)

    async def _services(self, user_id: uuid.UUID) -> list[ServiceContext]:
        # Load the services the caller is enrolled in or has access to.
        async with self._database.session() as session:
            rows = await EnrollmentService(session).list_for_user(user_id)
        return [self._service_context(enrollment, service) for enrollment, service in rows]

    async def _tasks(self, user_id: uuid.UUID) -> list[TaskContext]:
        # Load the caller's relevant open tasks and their assignment information.
        async with self._database.session() as session:
            tasks = await TaskService(session).list_for_user(
                user_id, open_only=True, limit=self._settings.max_tasks
            )
            staff = await StaffService(session).map_by_id(
                [task.assignee_id for task in tasks if task.assignee_id]
            )
            names = await EnrollmentService(session).service_names(
                [task.service_id for task in tasks if task.service_id]
            )
        return [self._task_context(task, names, staff) for task in tasks]

    async def _memory(self, call: Call) -> LoadedMemory:
        # Load the caller's memory from earlier calls: the digest, and whether it is still pending.
        async with self._database.session() as session:
            return await MemoryStore(session).load(
                call.organization_id,
                call.user_id,
                call.id,
                timedelta(seconds=self._settings.memory_pending_s),
            )

    def _organization_context(self, organization: Organization) -> OrganizationContext:
        # Convert the organization database model into the context object used by the agent.
        return OrganizationContext(
            id=organization.id,
            name=organization.name,
            timezone=organization.timezone,
            persona_name=organization.persona_name,
            persona_instructions=organization.persona_instructions,
            voice=organization.voice,
            phone=organization.phone,
            safety_message=organization.safety_message,
            blocked_phrases=list(organization.blocked_phrases),
        )

    def _caller_context(self, user: User, organization: Organization) -> CallerContext:
        # Convert the user database model into the caller context used by the agent.
        return CallerContext(
            id=user.id,
            full_name=user.full_name,
            locale=user.locale,
            timezone=user.timezone or organization.timezone,
            email=user.email,
            phone=user.phone,
            details=user.profile,
        )

    def _service_context(self, enrollment: Enrollment, service: Service) -> ServiceContext:
        # Convert a service database record into the service context used by the agent.
        return ServiceContext(
            id=service.id,
            key=service.key,
            name=service.name,
            description=service.description,
            instructions=service.instructions,
            status=enrollment.status,
            details=enrollment.details,
        )

    def _task_context(self, task: Task, names: dict, staff: dict) -> TaskContext:
        # Convert a task database record into the task context used by the agent.
        assignee = staff.get(task.assignee_id)
        return TaskContext(
            reference=task.reference,
            title=task.title,
            status=task.status,
            service_name=names.get(task.service_id),
            assignee_name=assignee.full_name if assignee else None,
            assignee_team=assignee.team if assignee else None,
            due_at=task.due_at,
            updated_at=task.updated_at,
        )
