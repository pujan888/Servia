from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import EnrollmentStatus


class ServiceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    key: str
    name: str
    description: str | None


class EnrollmentCreate(BaseModel):
    service_id: uuid.UUID
    details: dict[str, Any] = Field(default_factory=dict)


class EnrollmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    service_id: uuid.UUID
    status: EnrollmentStatus
    details: dict[str, Any]
