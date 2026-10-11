from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Session
from app.modules.accounts.service import UserService
from app.modules.catalog.schemas import EnrollmentCreate, EnrollmentRead, ServiceRead
from app.modules.catalog.service import CatalogService, EnrollmentService

router = APIRouter(tags=["catalog"])


@router.get("/organizations/{organization_id}/services", response_model=list[ServiceRead])
async def list_services(organization_id: uuid.UUID, session: Session):
    return await CatalogService(session).list(organization_id)


@router.post(
    "/users/{user_id}/enrollments",
    response_model=EnrollmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def join_service(user_id: uuid.UUID, data: EnrollmentCreate, session: Session):
    user = await UserService(session).get(user_id)
    return await EnrollmentService(session).join(user.organization_id, user.id, data)


@router.get("/users/{user_id}/enrollments", response_model=list[EnrollmentRead])
async def list_enrollments(user_id: uuid.UUID, session: Session):
    return [enrollment for enrollment, _ in await EnrollmentService(session).list_for_user(user_id)]
