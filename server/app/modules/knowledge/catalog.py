from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from app.core.database import Database
from app.modules.knowledge.models import EMBEDDING_DIM, CardVariant


@dataclass(frozen=True)
class CardIndex:
    """All question variants of one published collection version, in memory."""

    keys: tuple[str, ...]
    answers: tuple[str, ...]
    vectors: np.ndarray

    def best(self, query: np.ndarray) -> tuple[str, str, float] | None:
        if not self.keys:
            return None
        scores = self.vectors @ query
        index = int(np.argmax(scores))
        return self.keys[index], self.answers[index], float(scores[index])


class CardCatalog:
    """A published version never changes, so its card index is cached for the
    life of the process."""
    # Checks the question against cached card-question vectors in memory.

    def __init__(self, database: Database) -> None:
        self._database = database
        self._indexes: dict[tuple[uuid.UUID, int], CardIndex] = {}

    async def get(
        self, organization_id: uuid.UUID, collection_id: uuid.UUID, version: int
    ) -> CardIndex:
        key = (collection_id, version)
        if key not in self._indexes:
            async with self._database.session() as session:
                result = await session.execute(
                    select(CardVariant.card_key, CardVariant.answer, CardVariant.embedding).where(
                        CardVariant.organization_id == organization_id,
                        CardVariant.collection_id == collection_id,
                        CardVariant.version == version,
                    )
                )
                rows = result.all()
            vectors = np.array([row.embedding for row in rows], dtype=np.float32).reshape(
                -1, EMBEDDING_DIM
            )
            self._indexes[key] = CardIndex(
                tuple(r.card_key for r in rows), tuple(r.answer for r in rows), vectors
            )
        return self._indexes[key]
