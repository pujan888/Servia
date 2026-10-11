from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Resources, Session
from app.modules.releases.models import AgentRelease
from app.modules.releases.schemas import ItemDiff, ReleaseBuild, ReleasePromote, ReleaseRead
from app.modules.releases.service import (
    ReleaseBuilder,
    ReleaseCatalog,
    ReleasePromoter,
    ReleaseStore,
)

router = APIRouter(tags=["releases"])


@router.get("/releases", response_model=list[ReleaseRead])
async def list_releases(organization_id: uuid.UUID, session: Session):
    return await ReleaseStore(session).list(organization_id)


@router.post("/releases", response_model=ReleaseRead, status_code=status.HTTP_201_CREATED)
async def build_release(
    organization_id: uuid.UUID, data: ReleaseBuild, session: Session, resources: Resources
):
    return await ReleaseBuilder(session, resources.settings).build(organization_id, data.created_by)


@router.post("/releases/{release_id}/promote", response_model=ReleaseRead)
async def promote_release(
    organization_id: uuid.UUID, release_id: uuid.UUID, data: ReleasePromote, resources: Resources
):
    async def move(promoter: ReleasePromoter, release: AgentRelease) -> AgentRelease:
        match data.to:
            case "staged":
                return await promoter.stage(release)
            case "live":
                return await promoter.go_live(release)
            case "full":
                return await promoter.complete(release)

    return await _change(organization_id, release_id, resources, move)


@router.post("/releases/{release_id}/rollback", response_model=ReleaseRead)
async def rollback_release(organization_id: uuid.UUID, release_id: uuid.UUID, resources: Resources):
    async def move(promoter: ReleasePromoter, release: AgentRelease) -> AgentRelease:
        return await promoter.rollback(release)

    return await _change(organization_id, release_id, resources, move)


@router.get("/releases/{first_id}/diff/{second_id}", response_model=list[ItemDiff])
async def diff_releases(
    organization_id: uuid.UUID, first_id: uuid.UUID, second_id: uuid.UUID, session: Session
):
    store = ReleaseStore(session)
    return await store.diff(
        await store.get(organization_id, first_id), await store.get(organization_id, second_id)
    )


async def _change(organization_id, release_id, resources, move) -> AgentRelease:
    """Commits the change, then drops the cached live set, in that order."""
    catalog = ReleaseCatalog(resources.database, resources.cache, resources.settings)
    async with resources.database.session() as session:
        release = await ReleaseStore(session).get(organization_id, release_id)
        promoter = ReleasePromoter(session, resources.registry, resources.settings)
        release = await move(promoter, release)
    await catalog.resolver.invalidate(organization_id)
    return release
