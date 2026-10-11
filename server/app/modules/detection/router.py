from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Resources, Session
from app.core.enums import ReviewStatus
from app.modules.calls.service import CallService, TurnRecordService
from app.modules.detection.schemas import (
    CallerBlockRead,
    CallerBlockWrite,
    FlagDraft,
    ReviewClose,
    ReviewItemRead,
)
from app.modules.detection.service import CallerBlockService, ReviewService

router = APIRouter(tags=["detection"])


@router.get("/review-items", response_model=list[ReviewItemRead])
async def list_review_items(
    organization_id: uuid.UUID, session: Session, status: ReviewStatus | None = ReviewStatus.OPEN
):
    return await ReviewService(session).list(organization_id, status)


@router.put("/review-items/{item_id}", response_model=ReviewItemRead)
async def close_review_item(item_id: uuid.UUID, data: ReviewClose, session: Session):
    return await ReviewService(session).close(item_id, data)


@router.get("/calls/{call_id}/flags", response_model=list[FlagDraft])
async def list_flags(call_id: uuid.UUID, session: Session):
    call = await CallService(session).get(call_id)
    turns = await TurnRecordService(session).list_for_call(call.organization_id, call_id)
    return [FlagDraft.model_validate(flag) for turn in turns for flag in turn.flags]


@router.get("/config/caller-blocks", response_model=list[CallerBlockRead])
async def list_blocks(organization_id: uuid.UUID, session: Session, resources: Resources):
    return await CallerBlockService(session, resources.cache).list(organization_id)


@router.post(
    "/config/caller-blocks", response_model=CallerBlockRead, status_code=status.HTTP_201_CREATED
)
async def add_block(
    organization_id: uuid.UUID, data: CallerBlockWrite, session: Session, resources: Resources
):
    return await CallerBlockService(session, resources.cache).add(organization_id, data)


@router.delete("/config/caller-blocks/{block_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_block(
    organization_id: uuid.UUID, block_id: uuid.UUID, session: Session, resources: Resources
):
    await CallerBlockService(session, resources.cache).remove(organization_id, block_id)
