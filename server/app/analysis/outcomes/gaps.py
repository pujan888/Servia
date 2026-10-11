from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AnalyticsSettings
from app.core.enums import CallChannel
from app.modules.calls.models import Call, TurnRecord
from app.modules.knowledge.embedder import Embedder


@dataclass(slots=True)
class Gap:
    size: int
    customers: int
    examples: list[str]
    cohesion: float


class GapMiner:
    """Groups the questions the knowledge could not answer, so the business
    knows which answers to write next. Agglomerative clustering with average
    linkage and a distance threshold, not a fixed number of groups."""

    def __init__(
        self, session: AsyncSession, embedder: Embedder, settings: AnalyticsSettings
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._settings = settings

    async def mine(self, organization_id: uuid.UUID, since: datetime, until: datetime) -> list[Gap]:
        rows = (
            await self._session.execute(
                select(TurnRecord.transcript_in, Call.user_id, TurnRecord.retrieval)
                .join(Call, Call.id == TurnRecord.call_id)
                .where(
                    TurnRecord.organization_id == organization_id,
                    Call.channel != CallChannel.REPLAY,
                    TurnRecord.created_at >= since,
                    TurnRecord.created_at < until,
                    TurnRecord.transcript_in.is_not(None),
                )
                .order_by(TurnRecord.created_at.desc())
            )
        ).all()
        asked = [
            (text, user)
            for text, user, retrieval in rows
            if any(item.get("route") == "none" for item in retrieval)
        ][: self._settings.gap_max_texts]
        if not asked:
            return []
        vectors = np.asarray(
            await self._embedder.passages([text for text, _ in asked]), dtype=np.float32
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        groups = self.cluster(vectors @ vectors.T, 1 - self._settings.gap_distance)
        gaps = [self._gap(members, asked, vectors) for members in groups]
        gaps.sort(key=lambda gap: (gap.customers, gap.size), reverse=True)
        return gaps[: self._settings.gap_groups]

    @staticmethod
    def cluster(similarity: np.ndarray, minimum: float) -> list[list[int]]:
        """Merges the two closest groups while their average similarity is at
        least `minimum`. Each group's similarity to the others is the size-
        weighted average of its members'."""
        groups = [[i] for i in range(len(similarity))]
        matrix = similarity.astype(np.float64).copy()
        np.fill_diagonal(matrix, -np.inf)
        alive = np.ones(len(groups), dtype=bool)
        while alive.sum() > 1:
            masked = np.where(np.outer(alive, alive), matrix, -np.inf)
            a, b = np.unravel_index(np.argmax(masked), masked.shape)
            if masked[a, b] < minimum:
                break
            size_a, size_b = len(groups[a]), len(groups[b])
            merged = (matrix[a] * size_a + matrix[b] * size_b) / (size_a + size_b)
            matrix[a], matrix[:, a] = merged, merged
            matrix[a, a] = -np.inf
            groups[a] += groups[b]
            alive[b] = False
        return [groups[i] for i in np.flatnonzero(alive)]

    def _gap(self, members: list[int], asked, vectors: np.ndarray) -> Gap:
        centre = vectors[members].mean(axis=0)
        closeness = vectors[members] @ centre
        order = [members[i] for i in np.argsort(-closeness)]
        pairs = vectors[members] @ vectors[members].T
        cohesion = float(pairs.mean()) if len(members) > 1 else 1.0
        return Gap(
            size=len(members),
            customers=len({asked[i][1] for i in members}),
            examples=[asked[i][0] for i in order[: self._settings.gap_examples]],
            cohesion=round(cohesion, 3),
        )
