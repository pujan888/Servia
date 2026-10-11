from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import Priority, TaskStatus


class TaskCreate(BaseModel):
    user_id: uuid.UUID
    title: str = Field(min_length=1, max_length=300)
    details: str | None = None
    service_id: uuid.UUID | None = None
    assignee_id: uuid.UUID | None = None
    status: TaskStatus = TaskStatus.OPEN
    due_at: datetime | None = None


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    reference: str
    user_id: uuid.UUID
    service_id: uuid.UUID | None
    assignee_id: uuid.UUID | None
    title: str
    details: str | None
    status: TaskStatus
    due_at: datetime | None
    resolved_at: datetime | None
    updated_at: datetime
    kind: str | None
    priority: Priority
    queue: str | None
    confidence: float | None
    update_count: int
