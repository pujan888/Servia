from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.core.dependencies import Session
from app.modules.accounts.service import UserService
from app.modules.memory.schemas import ErasureRead, MemoryRead, SlotRead
from app.modules.memory.service import ErasureService
from app.modules.memory.store import MemoryStore

router = APIRouter(tags=["memory"])


@router.get("/users/{user_id}/memory", response_model=MemoryRead)
async def get_memory(
    user_id: uuid.UUID, session: Session, summaries: int = Query(default=20, ge=1, le=100)
):
    user = await UserService(session).get(user_id)
    store = MemoryStore(session)
    return MemoryRead(
        digest=await store.digest(user.organization_id, user.id),
        slots=[
            SlotRead.model_validate(slot)
            for slot in await store.current_slots(user.organization_id, user.id)
        ],
        summaries=await store.summaries(user.organization_id, user.id, summaries),
    )


@router.delete("/users/{user_id}/memory", response_model=ErasureRead)
async def erase_memory(
    user_id: uuid.UUID,
    session: Session,
    requested_by: str = Query(min_length=1, max_length=200),
):
    user = await UserService(session).get(user_id)
    return await ErasureService(session).erase(user.organization_id, user.id, requested_by)
