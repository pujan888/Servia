from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Session
from app.modules.accounts.schemas import OrganizationRead, UserCreate, UserRead
from app.modules.accounts.service import OrganizationService, UserService

router = APIRouter(tags=["accounts"])


@router.get("/organizations", response_model=list[OrganizationRead])
async def list_organizations(session: Session):
    return await OrganizationService(session).list()


@router.get("/organizations/{organization_id}/users", response_model=list[UserRead])
async def list_users(organization_id: uuid.UUID, session: Session):
    return await UserService(session).list_for_organization(organization_id)


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, session: Session):
    await OrganizationService(session).get(data.organization_id)
    return await UserService(session).create(data)


@router.get("/users/{user_id}", response_model=UserRead)
async def get_user(user_id: uuid.UUID, session: Session):
    return await UserService(session).get(user_id)
