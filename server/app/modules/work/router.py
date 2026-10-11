from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Session
from app.modules.accounts.service import UserService
from app.modules.work.schemas import TaskCreate, TaskRead
from app.modules.work.service import TaskService

router = APIRouter(tags=["work"])


@router.post("/tasks", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
async def create_task(data: TaskCreate, session: Session):
    user = await UserService(session).get(data.user_id)
    return await TaskService(session).create(user.organization_id, data)


@router.get("/work-items", response_model=list[TaskRead])
async def list_work_items(
    organization_id: uuid.UUID, session: Session, queue: str | None = None, open_only: bool = True
):
    return await TaskService(session).list_work_items(organization_id, queue, open_only)


@router.get("/users/{user_id}/tasks", response_model=list[TaskRead])
async def list_tasks(user_id: uuid.UUID, session: Session, open_only: bool = False):
    return await TaskService(session).list_for_user(user_id, open_only=open_only)
