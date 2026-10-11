from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CallChannel, IdentityLevel
from app.core.errors import NotFound
from app.modules.accounts.models import Organization, Staff, User
from app.modules.accounts.schemas import UserCreate


class OrganizationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self) -> Sequence[Organization]:
        result = await self._session.execute(select(Organization).order_by(Organization.name))
        return result.scalars().all()

    async def get(self, organization_id: uuid.UUID) -> Organization:
        organization = await self._session.get(Organization, organization_id)
        if organization is None:
            raise NotFound("Organization not found", organization_id=str(organization_id))
        return organization

    async def get_by_slug(self, slug: str) -> Organization:
        result = await self._session.execute(select(Organization).where(Organization.slug == slug))
        organization = result.scalar_one_or_none()
        if organization is None:
            raise NotFound("Organization not found", slug=slug)
        return organization


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: UserCreate) -> User:
        user = User(**data.model_dump())
        self._session.add(user)
        await self._session.flush()
        return user

    async def get(self, user_id: uuid.UUID) -> User:
        user = await self._session.get(User, user_id)
        if user is None:
            raise NotFound("User not found", user_id=str(user_id))
        return user

    async def list_for_organization(self, organization_id: uuid.UUID) -> Sequence[User]:
        result = await self._session.execute(
            select(User).where(User.organization_id == organization_id).order_by(User.full_name)
        )
        return result.scalars().all()


class StaffService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def map_by_id(self, staff_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Staff]:
        if not staff_ids:
            return {}
        result = await self._session.execute(select(Staff).where(Staff.id.in_(set(staff_ids))))
        return {member.id: member for member in result.scalars()}


class IdentityResolver:
    """Sets how sure we are about who is on the call. Only code sets it.

    The web path returns VERIFIED because a web user is expected to be logged in. There is
    no login yet, so today this trusts the browser; once authentication exists, it is what
    makes VERIFIED true. The phone path returns UNKNOWN until caller-number matching exists.
    """

    def resolve(self, channel: CallChannel, user_id: uuid.UUID | None = None) -> IdentityLevel:
        if channel == CallChannel.WEB and user_id is not None:
            return IdentityLevel.VERIFIED
        return IdentityLevel.UNKNOWN
