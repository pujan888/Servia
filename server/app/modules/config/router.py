from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Resources, Session
from app.modules.config.schemas import (
    BindingRead,
    BindingSaved,
    BindingWrite,
    DetectionSpec,
    ModelProfileRead,
    ModelProfileWrite,
    OptimisationRead,
    OptimisationWrite,
    RouteRead,
    RouteWrite,
    SkillRead,
    SkillWrite,
    SlotKeyRead,
    SlotKeyWrite,
    ValidationReport,
)
from app.modules.config.service import (
    AgentConfigService,
    DetectionSettingService,
    ModelProfileService,
    OptimisationService,
    RouteService,
    SkillService,
    SkillValidator,
    SlotKeyService,
    ToolBindingService,
)
from app.modules.detection.schemas import DetectionSettingWrite

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/skills", response_model=list[SkillRead])
async def list_skills(organization_id: uuid.UUID, session: Session):
    return await SkillService(session).list(organization_id)


@router.post("/skills", response_model=SkillRead, status_code=status.HTTP_201_CREATED)
async def create_skill(organization_id: uuid.UUID, data: SkillWrite, session: Session):
    return await SkillService(session).create(organization_id, data)


@router.put("/skills/{key}", response_model=SkillRead)
async def update_skill(organization_id: uuid.UUID, key: str, data: SkillWrite, session: Session):
    return await SkillService(session).update(organization_id, key, data)


@router.post("/skills/validate", response_model=ValidationReport)
async def validate_skill(organization_id: uuid.UUID, data: SkillWrite, session: Session):
    return await SkillValidator(session).validate(organization_id, data)


@router.get("/model-profile", response_model=ModelProfileRead | None)
async def get_model_profile(organization_id: uuid.UUID, session: Session):
    return await ModelProfileService(session).get(organization_id)


@router.put("/model-profile", response_model=ModelProfileRead)
async def save_model_profile(organization_id: uuid.UUID, data: ModelProfileWrite, session: Session):
    return await ModelProfileService(session).save(organization_id, data)


@router.get("/tools", response_model=list[BindingRead])
async def list_tools(organization_id: uuid.UUID, session: Session, resources: Resources):
    return await ToolBindingService(session, resources.registry).list(organization_id)


@router.post("/tools", response_model=BindingSaved, status_code=status.HTTP_201_CREATED)
async def create_tool(
    organization_id: uuid.UUID, data: BindingWrite, session: Session, resources: Resources
):
    binding, report = await ToolBindingService(session, resources.registry).save(
        organization_id, data
    )
    return BindingSaved(binding=BindingRead.model_validate(binding), warnings=report.warnings)


@router.put("/tools/{key}", response_model=BindingSaved)
async def update_tool(
    organization_id: uuid.UUID, key: str, data: BindingWrite, session: Session, resources: Resources
):
    binding, report = await ToolBindingService(session, resources.registry).save(
        organization_id, data, key=key
    )
    return BindingSaved(binding=BindingRead.model_validate(binding), warnings=report.warnings)


@router.get("/memory-slots", response_model=list[SlotKeyRead])
async def list_slot_keys(organization_id: uuid.UUID, session: Session):
    return await SlotKeyService(session).list(organization_id)


@router.post("/memory-slots", response_model=SlotKeyRead, status_code=status.HTTP_201_CREATED)
async def save_slot_key(organization_id: uuid.UUID, data: SlotKeyWrite, session: Session):
    return await SlotKeyService(session).save(organization_id, data)


@router.get("/escalation-routes", response_model=list[RouteRead])
async def list_routes(organization_id: uuid.UUID, session: Session):
    return await RouteService(session).list(organization_id)


@router.put("/escalation-routes", response_model=list[RouteRead])
async def replace_routes(organization_id: uuid.UUID, data: list[RouteWrite], session: Session):
    return await RouteService(session).replace(organization_id, data)


@router.get("/detection", response_model=DetectionSpec)
async def get_detection(organization_id: uuid.UUID, session: Session, resources: Resources):
    service = AgentConfigService(session, detection=resources.settings.detection)
    return (await service.load(organization_id)).detection


@router.put("/detection", response_model=DetectionSpec)
async def save_detection(
    organization_id: uuid.UUID, data: DetectionSettingWrite, session: Session, resources: Resources
):
    await DetectionSettingService(session).save(
        organization_id, data.flag_type, data.threshold, data.enabled
    )
    service = AgentConfigService(session, detection=resources.settings.detection)
    return (await service.load(organization_id)).detection


@router.get("/optimisations", response_model=list[OptimisationRead])
async def list_optimisations(organization_id: uuid.UUID, session: Session):
    return await OptimisationService(session).list(organization_id)


@router.put("/optimisations", response_model=OptimisationRead)
async def save_optimisation(organization_id: uuid.UUID, data: OptimisationWrite, session: Session):
    return await OptimisationService(session).save(organization_id, data)
