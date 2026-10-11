from __future__ import annotations

import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import TypeAdapter
from sqlalchemy import and_, func, or_, select

from app.core.cache import CacheKeys, CacheLayer
from app.core.config import KnowledgeSettings
from app.core.database import Database
from app.modules.config.schemas import CollectionSpec
from app.modules.knowledge.catalog import CardCatalog
from app.modules.knowledge.embedder import Embedder
from app.modules.knowledge.models import KnowledgeChunk
from app.modules.knowledge.schemas import Passage, RetrievalResult


@dataclass(frozen=True)
class SearchScope:
    """One collection version to search, with the metadata filters that apply."""

    collection_id: uuid.UUID
    version: int
    filters: tuple[tuple[str, str], ...]


class MetadataFilter:
    """A chunk passes when it has no value for a filter key, or the caller's value.
    Only the keys a collection declares are used."""

    @staticmethod
    def scope(collection: CollectionSpec, caller_values: dict[str, str]) -> SearchScope:
        pairs = tuple(
            (key, str(caller_values[key]))
            for key in collection.filter_keys
            if caller_values.get(key)
        )
        return SearchScope(collection.id, collection.active_version, pairs)


class HybridSearcher:
    """Vector similarity ranks the results. Lexical search does one job: find an
    exact identifier the caller said, such as an error code or a part number,
    which a vector blurs. A chunk with that identifier gets a bonus, and is kept
    even below the similarity threshold.

    Ranking natural questions by word overlap was measured on the demo corpus
    and made results worse: common words such as "window" outvoted the meaning."""

    CODE = re.compile(r"\b(?=\w*\d)\w{2,}\b")

    def __init__(self, database: Database, settings: KnowledgeSettings) -> None:
        self._database = database
        self._settings = settings

    @classmethod
    def codes(cls, query: str) -> list[str]:
        return [code.lower() for code in cls.CODE.findall(query)]

    async def search(
        self,
        organization_id: uuid.UUID,
        scopes: Sequence[SearchScope],
        query: str,
        vector: list[float],
        threshold: float,
    ) -> list[Passage]:
        if not scopes:
            return []
        chunk, k = KnowledgeChunk, self._settings.candidate_k
        where = and_(
            chunk.organization_id == organization_id, or_(*(self._scope(s) for s in scopes))
        )
        distance = chunk.embedding.cosine_distance(vector)

        async with self._database.session() as session:
            dense = (
                (await session.execute(select(chunk.id).where(where).order_by(distance).limit(k)))
                .scalars()
                .all()
            )
            exact = set(await self._exact(session, where, self.codes(query), k))
            candidates = set(dense) | exact
            if not candidates:
                return []
            rows = (
                await session.execute(
                    select(
                        chunk.id,
                        chunk.title,
                        chunk.heading,
                        chunk.text,
                        (1 - distance).label("similarity"),
                    ).where(chunk.id.in_(candidates))
                )
            ).all()

        bonus = self._settings.code_bonus
        scored = [
            (row.similarity + (bonus if row.id in exact else 0.0), row)
            for row in rows
            if row.similarity >= threshold or row.id in exact
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Passage(
                title=row.title,
                heading=row.heading,
                text=row.text,
                similarity=round(row.similarity, 4),
                chunk_id=row.id,
            )
            for _, row in scored[: self._settings.top_k]
        ]

    async def _exact(self, session, where, codes: list[str], k: int) -> list[uuid.UUID]:
        if not codes:
            return []
        tsquery = func.to_tsquery("english", " | ".join(codes))
        result = await session.execute(
            select(KnowledgeChunk.id)
            .where(where, KnowledgeChunk.tsv.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(KnowledgeChunk.tsv, tsquery).desc())
            .limit(k)
        )
        return list(result.scalars().all())

    @staticmethod
    def _scope(scope: SearchScope):
        conditions = [
            KnowledgeChunk.collection_id == scope.collection_id,
            KnowledgeChunk.version == scope.version,
        ]
        for key, value in scope.filters:
            field = KnowledgeChunk.metadata_[key].astext
            conditions.append(or_(field.is_(None), field == value))
        return and_(*conditions)


class KnowledgeRouter:
    """The retrieval ladder:
    1 answer cards in the call's scope      (in memory, a few milliseconds)
    2 hybrid search in the call's scope
    3 hybrid search in every other collection, with a stricter threshold
    4 nothing: the caller gets a refusal and an offer
    """

    RESULT = TypeAdapter(RetrievalResult)
    VECTOR = TypeAdapter(list[float])

    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        cards: CardCatalog,
        cache: CacheLayer,
        settings: KnowledgeSettings,
    ) -> None:
        self._embedder = embedder
        self._cards = cards
        self._cache = cache
        self._settings = settings
        self._searcher = HybridSearcher(database, settings)

    async def search(
        self,
        organization_id: uuid.UUID,
        query: str,
        scope: Sequence[CollectionSpec],
        others: Sequence[CollectionSpec] = (),
        caller_values: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        started = time.perf_counter()
        text = " ".join(query.split()).lower()
        values = caller_values or {}
        cache_key = self._cache_key(organization_id, text, scope, others, values)
        if use_cache and (cached := await self._cache.peek(cache_key, self.RESULT)) is not None:
            return cached.model_copy(update={"latency_ms": self._ms(started)})

        result = await self._route(organization_id, text, scope, others, values)
        result.latency_ms, result.query = self._ms(started), text
        if use_cache:
            await self._cache.put(cache_key, result, self.RESULT, self._settings.search_cache_ttl_s)
        return result

    async def _route(self, organization_id, text, scope, others, values) -> RetrievalResult:
        vector = await self._vector(organization_id, text)

        card = await self._best_card(organization_id, scope, np.asarray(vector, dtype=np.float32))
        if card and card[2] >= self._settings.card_threshold:
            return RetrievalResult(
                route="card",
                card_key=card[0],
                card_answer=card[1],
                best_similarity=round(card[2], 4),
            )

        threshold = self._settings.chunk_threshold
        scoped = [MetadataFilter.scope(c, values) for c in scope]
        passages = await self._searcher.search(organization_id, scoped, text, vector, threshold)
        if passages:
            return RetrievalResult(
                route="scoped", passages=passages, best_similarity=passages[0].similarity
            )

        wider = [MetadataFilter.scope(c, values) for c in others]
        stricter = threshold + self._settings.tenant_threshold_margin
        passages = await self._searcher.search(organization_id, wider, text, vector, stricter)
        if passages:
            return RetrievalResult(
                route="tenant", passages=passages, best_similarity=passages[0].similarity
            )
        return RetrievalResult(route="none", best_similarity=round(card[2], 4) if card else None)

    async def _best_card(self, organization_id, scope, vector: np.ndarray):
        best = None
        for collection in scope:
            index = await self._cards.get(organization_id, collection.id, collection.active_version)
            found = index.best(vector)
            if found and (best is None or found[2] > best[2]):
                best = found
        return best

    async def _vector(self, organization_id: uuid.UUID, text: str) -> list[float]:
        key = CacheKeys.build(organization_id, "qvec", text)
        return await self._cache.get(key, self.VECTOR, lambda: self._embedder.query(text))

    @staticmethod
    def _cache_key(organization_id, text, scope, others, values) -> str:
        versions = ",".join(f"{c.id}:{c.active_version}" for c in (*scope, *others))
        filters = ",".join(f"{k}={v}" for k, v in sorted(values.items()))
        return CacheKeys.build(organization_id, "kb", versions, filters, text)

    @staticmethod
    def _ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)
