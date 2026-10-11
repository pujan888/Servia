from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import EnrollmentStatus
from app.core.errors import Conflict, NotFound
from app.modules.catalog.models import Enrollment, Service
from app.modules.catalog.schemas import EnrollmentCreate


class CatalogService:
    """Manages reading available services from the catalog."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, organization_id: uuid.UUID) -> Sequence[Service]:
        result = await self._session.execute(
            select(Service).where(Service.organization_id == organization_id).order_by(Service.name)
        )
        return result.scalars().all()

    async def get(self, service_id: uuid.UUID) -> Service:
        service = await self._session.get(Service, service_id)
        if service is None:
            raise NotFound("Service not found", service_id=str(service_id))
        return service


class EnrollmentService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def join(
        self, organization_id: uuid.UUID, user_id: uuid.UUID, data: EnrollmentCreate
    ) -> Enrollment:
        service = await CatalogService(self._session).get(data.service_id)
        if service.organization_id != organization_id:
            raise Conflict("Service belongs to another organization")

        existing = await self._find(user_id, data.service_id)
        if existing is not None:
            existing.status = EnrollmentStatus.ACTIVE
            existing.details = {**existing.details, **data.details}
            return existing

        enrollment = Enrollment(
            organization_id=organization_id,
            user_id=user_id,
            service_id=data.service_id,
            details=data.details,
        )
        self._session.add(enrollment)
        await self._session.flush()
        return enrollment

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[tuple[Enrollment, Service]]:
        result = await self._session.execute(
            select(Enrollment, Service)
            .join(Service, Service.id == Enrollment.service_id)
            .where(Enrollment.user_id == user_id)
            .order_by(Service.name)
        )
        return result.all()

    async def _find(self, user_id: uuid.UUID, service_id: uuid.UUID) -> Enrollment | None:
        result = await self._session.execute(
            select(Enrollment).where(
                Enrollment.user_id == user_id, Enrollment.service_id == service_id
            )
        )
        return result.scalar_one_or_none()

    async def service_names(self, service_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
        if not service_ids:
            return {}
        result = await self._session.execute(
            select(Service.id, Service.name).where(Service.id.in_(set(service_ids)))
        )
        return dict(result.all())
