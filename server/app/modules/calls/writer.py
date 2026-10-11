from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from app.core.database import Database


class BufferedWriter[DraftT](ABC):
    """Collects drafts and writes them in the background, off the turn's path.
    A failed write is logged; it never stops the call."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._buffer: list[DraftT] = []
        self._flushing: asyncio.Task | None = None

    def record(self, draft: DraftT) -> None:
        # Add a draft to the buffer and start background saving.
        self._buffer.append(draft)
        if self._flushing is None or self._flushing.done():
            self._flushing = asyncio.create_task(self._flush())

    async def aclose(self) -> None:
        # Flush remaining drafts before the writer shuts down.
        if self._flushing is not None:
            await self._flushing
        await self._flush()

    async def _flush(self) -> None:
        # Save buffered drafts to the database in the background.
        while self._buffer:
            batch, self._buffer = self._buffer, []
            try:
                async with self._database.session() as session:
                    session.add_all([self.to_row(draft) for draft in batch])
            except Exception:
                logger.exception("{} write failed", type(self).__name__)

    @abstractmethod
    def to_row(self, draft: DraftT) -> Any: ...
