from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import ReleaseStatus


class LiveSet(BaseModel):
    """Which releases new calls get: the stable one, and a canary for a share."""

    stable: uuid.UUID | None = None
    canary: uuid.UUID | None = None
    canary_percent: int = 0


class ReleaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    hash: str
    status: ReleaseStatus
    canary_percent: int
    created_by: str
    live_at: datetime | None
    retired_at: datetime | None
    notes: list[str]
    created_at: datetime


class ReleaseBuild(BaseModel):
    created_by: str = Field(min_length=1, max_length=200)


class ReleasePromote(BaseModel):
    to: Literal["staged", "live", "full"]


class ItemDiff(BaseModel):
    item_type: str
    item_key: str
    change: Literal["added", "removed", "changed"]
    before: Any = None
    after: Any = None
