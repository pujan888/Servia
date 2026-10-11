from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query, status
from loguru import logger

from app.core.dependencies import Resources, Session
from app.core.enums import Optimisation
from app.core.errors import Forbidden
from app.modules.accounts.service import UserService
from app.modules.calls.schemas import (
    CacheKindRead,
    CallCredentials,
    CallRead,
    CallStart,
    JournalEntryRead,
    LatencyComparison,
    LatencyReportRead,
    ToolHealthRead,
    TranscriptLineRead,
    TurnRecordRead,
)
from app.modules.calls.service import (
    CacheReportService,
    CallService,
    LatencyReportService,
    ToolHealthService,
    ToolJournalService,
    TranscriptService,
    TurnRecordService,
)
from app.modules.config.service import AgentConfigService
from app.modules.detection.service import PreAnswerGate
from app.voice.token import TokenService

router = APIRouter(tags=["calls"])


@router.post("/calls", response_model=CallCredentials, status_code=status.HTTP_201_CREATED)
async def start_call(data: CallStart, session: Session, resources: Resources):
    user = await UserService(session).get(data.user_id)
    logger.info("the user details are %s", user)
    gate = PreAnswerGate(resources.database, resources.cache)
    if await gate.blocked(user.organization_id, [str(user.id), user.phone, user.email]):
        raise Forbidden("This caller cannot be connected")
    call = await CallService(session).start(user, data.service_id)
    logger.info("the call details are %s", call)
    tokens = TokenService(resources.settings.livekit).issue(call)
    logger.info("the token details are %s", tokens)
    return tokens


@router.get("/calls/{call_id}", response_model=CallRead)
async def get_call(call_id: uuid.UUID, session: Session):
    return await CallService(session).get(call_id)


@router.get("/calls/{call_id}/turns", response_model=list[TurnRecordRead])
async def list_turns(call_id: uuid.UUID, session: Session):
    call = await CallService(session).get(call_id)
    return await TurnRecordService(session).list_for_call(call.organization_id, call_id)


@router.get("/calls/{call_id}/transcript", response_model=list[TranscriptLineRead])
async def get_transcript(call_id: uuid.UUID, session: Session):
    call = await CallService(session).get(call_id)
    return await TranscriptService(session).list_for_call(call.organization_id, call_id)


@router.get("/metrics/latency", response_model=LatencyReportRead | LatencyComparison)
async def latency_report(
    session: Session,
    organization_id: uuid.UUID,
    hours: int = Query(default=24, ge=1, le=24 * 90),
    optimisation: Optimisation | None = None,
):
    """With `optimisation`, the calls with it on against the calls with it off."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    service = LatencyReportService(session)
    if optimisation is not None:
        return await service.compare(organization_id, since, optimisation)
    return await service.report(organization_id, since)


@router.get("/metrics/cache", response_model=list[CacheKindRead])
async def cache_report(
    session: Session, organization_id: uuid.UUID, hours: int = Query(default=24, ge=1, le=24 * 90)
):
    since = datetime.now(UTC) - timedelta(hours=hours)
    return await CacheReportService(session).report(organization_id, since)


@router.get("/calls/{call_id}/journal", response_model=list[JournalEntryRead])
async def list_journal(call_id: uuid.UUID, session: Session):
    call = await CallService(session).get(call_id)
    return await ToolJournalService(session).list_for_call(call.organization_id, call_id)


@router.get("/metrics/tools", response_model=list[ToolHealthRead])
async def tool_health(
    session: Session,
    organization_id: uuid.UUID,
    hours: int = Query(default=24, ge=1, le=24 * 90),
):
    config = await AgentConfigService(session).load(organization_id)
    classes = {binding.key: binding.latency_class for binding in config.tools}
    since = datetime.now(UTC) - timedelta(hours=hours)
    return await ToolHealthService(session).report(organization_id, since, classes)
