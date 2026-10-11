from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query, status
from sqlalchemy import select

from app.core.dependencies import Resources, Session
from app.core.errors import Invalid, NotFound
from app.eval.assertions import Assertions
from app.eval.recorder import ScenarioRecorder
from app.eval.suite import SuiteJobs
from app.modules.calls.service import CallService
from app.modules.evaluation.models import (
    Annotation,
    DailyMeasurement,
    GapGroup,
    Scenario,
    SuiteResult,
)
from app.modules.evaluation.schemas import (
    AnnotationRead,
    AnnotationWrite,
    DailyRead,
    GapRead,
    ScenarioRead,
    ScenarioUpdate,
    SuiteRead,
    SuiteRun,
)
from app.modules.releases.service import ReleaseCatalog, ReleaseStore

router = APIRouter(tags=["evaluation"])


@router.post("/annotations", response_model=AnnotationRead, status_code=status.HTTP_201_CREATED)
async def annotate(data: AnnotationWrite, session: Session):
    call = await CallService(session).get(data.call_id)
    annotation = Annotation(organization_id=call.organization_id, **data.model_dump())
    session.add(annotation)
    await session.flush()
    return annotation


@router.post("/scenarios/from-call/{call_id}", response_model=ScenarioRead, status_code=201)
async def record_scenario(call_id: uuid.UUID, session: Session, resources: Resources):
    catalog = ReleaseCatalog(resources.database, resources.cache, resources.settings)
    return await ScenarioRecorder(session, resources.registry, catalog).record(call_id)


@router.get("/scenarios", response_model=list[ScenarioRead])
async def list_scenarios(organization_id: uuid.UUID, session: Session):
    result = await session.scalars(
        select(Scenario).where(Scenario.organization_id == organization_id).order_by(Scenario.name)
    )
    return result.all()


@router.put("/scenarios/{scenario_id}", response_model=ScenarioRead)
async def update_scenario(scenario_id: uuid.UUID, data: ScenarioUpdate, session: Session):
    scenario = await session.get(Scenario, scenario_id)
    if scenario is None:
        raise NotFound("Scenario not found", scenario_id=str(scenario_id))
    for spec in data.assertions:
        try:
            Assertions.build(spec)
        except (KeyError, ValueError) as error:
            raise Invalid("An assertion is not valid", assertion=spec, error=str(error)) from error
    scenario.assertions, scenario.enabled = data.assertions, data.enabled
    return scenario


@router.post("/suites/run", response_model=SuiteRead, status_code=status.HTTP_202_ACCEPTED)
async def run_suite(organization_id: uuid.UUID, data: SuiteRun, session: Session):
    release = await ReleaseStore(session).get(organization_id, data.release_id)
    return await SuiteJobs.enqueue(session, organization_id, release.id, nightly=False)


@router.get("/suites/{suite_id}", response_model=SuiteRead)
async def get_suite(suite_id: uuid.UUID, session: Session):
    suite = await session.get(SuiteResult, suite_id)
    if suite is None:
        raise NotFound("Suite not found", suite_id=str(suite_id))
    return suite


@router.get("/analytics/daily", response_model=list[DailyRead])
async def daily(organization_id: uuid.UUID, session: Session, days: int = Query(30, ge=1, le=396)):
    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
    result = await session.scalars(
        select(DailyMeasurement)
        .where(DailyMeasurement.organization_id == organization_id, DailyMeasurement.day >= since)
        .order_by(DailyMeasurement.day)
    )
    return result.all()


@router.get("/analytics/gaps", response_model=list[GapRead])
async def gaps(organization_id: uuid.UUID, session: Session):
    latest = await session.scalar(
        select(GapGroup.week)
        .where(GapGroup.organization_id == organization_id)
        .order_by(GapGroup.week.desc())
        .limit(1)
    )
    result = await session.scalars(
        select(GapGroup)
        .where(GapGroup.organization_id == organization_id, GapGroup.week == latest)
        .order_by(GapGroup.customers.desc(), GapGroup.size.desc())
    )
    return result.all()
