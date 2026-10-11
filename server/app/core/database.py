from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import DatabaseSettings


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._engine = create_async_engine(
            settings.url,
            echo=settings.echo,
            pool_pre_ping=True,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_recycle=settings.pool_recycle,
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Commits on clean exit, rolls back on failure."""
        async with self._sessions() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            await session.commit()

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def warm(self, connections: int = 1) -> None:
        """Warm up the connection pool by creating a number of connections."""

        async def touch() -> None:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

        await asyncio.gather(*(touch() for _ in range(connections)))

