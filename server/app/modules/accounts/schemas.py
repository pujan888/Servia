from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class OrganizationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    timezone: str


class UserCreate(BaseModel):
    organization_id: uuid.UUID
    full_name: str = Field(min_length=1, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=32)
    locale: str = "en-US"
    timezone: str | None = None
    profile: dict[str, Any] = Field(default_factory=dict)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    full_name: str
    email: str | None
    phone: str | None
    locale: str
    timezone: str | None
    profile: dict[str, Any]
