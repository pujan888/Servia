from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


async def get_session(
    container: Annotated[Container, Depends(get_container)],
) -> AsyncIterator[AsyncSession]:
    async with container.database.session() as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]
Resources = Annotated[Container, Depends(get_container)]
