from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import EnrollmentStatus, IdentityLevel, TaskStatus
from app.modules.memory.schemas import Digest


class OrganizationContext(BaseModel):
    id: uuid.UUID
    name: str
    timezone: str
    persona_name: str
    persona_instructions: str | None = None
    voice: str | None = None
    phone: str | None = None
    safety_message: str | None = None
    blocked_phrases: list[str] = Field(default_factory=list)


class CallerContext(BaseModel):
    id: uuid.UUID
    full_name: str
    locale: str
    timezone: str
    email: str | None = None
    phone: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ServiceContext(BaseModel):
    id: uuid.UUID
    key: str
    name: str
    description: str | None = None
    instructions: str | None = None
    status: EnrollmentStatus
    details: dict[str, Any] = Field(default_factory=dict)


class TaskContext(BaseModel):
    reference: str
    title: str
    status: TaskStatus
    service_name: str | None = None
    assignee_name: str | None = None
    assignee_team: str | None = None
    due_at: datetime | None = None
    updated_at: datetime


class ContextPack(BaseModel):
    """Everything the agent knows about this caller before they speak."""

    call_id: uuid.UUID
    identity_level: IdentityLevel
    organization: OrganizationContext
    caller: CallerContext
    focus_service: ServiceContext | None = None
    services: list[ServiceContext] = Field(default_factory=list)
    tasks: list[TaskContext] = Field(default_factory=list)
    digest: Digest | None = None
    degraded: list[str] = Field(default_factory=list)
    service_count: int | None = None
