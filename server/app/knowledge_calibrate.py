"""Find the similarity thresholds that fit your embedding model and your content.

    uv run python -m app.knowledge_calibrate <organization_id>

Runs every retrieval test against the live versions with no threshold, then
reports the pass rate for each candidate threshold. Run it after changing the
embedding model, and after large content changes."""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from app.core.container import Container
from app.core.logging import LogConfig
from app.modules.config.service import AgentConfigService
from app.modules.knowledge.models import RetrievalTest
from app.modules.knowledge.retrieval import HybridSearcher, MetadataFilter


@dataclass
class Observation:
    question: str
    kind: str
    correct_top: bool
    top_similarity: float
    card_score: float


class Calibrator:
    CHUNK_RANGE = np.arange(0.50, 0.81, 0.01)
    CARD_RANGE = np.arange(0.70, 0.96, 0.01)

    def __init__(self, container: Container) -> None:
        self._container = container
        self._searcher = HybridSearcher(container.database, container.settings.knowledge)

    async def run(self, organization_id: uuid.UUID) -> None:
        async with self._container.database.session() as session:
            config = await AgentConfigService(session).load(organization_id)
            tests = (
                (
                    await session.execute(
                        select(RetrievalTest).where(
                            RetrievalTest.organization_id == organization_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        collections = {c.id: c for c in config.collections}
        observations = []
        for test in tests:
            collection = collections.get(test.collection_id)
            if collection is not None:
                observations.append(await self._observe(organization_id, collection, test))
        self._report(observations)

    async def _observe(self, organization_id, collection, test) -> Observation:
        text = " ".join(test.question.split()).lower()
        vector = await self._container.embedder.query(text)
        scope = MetadataFilter.scope(collection, test.filters)
        passages = await self._searcher.search(
            organization_id, [scope], text, vector, threshold=-1.0
        )
        cards = await self._container.knowledge_router._cards.get(
            organization_id, collection.id, collection.active_version
        )
        best = cards.best(np.asarray(vector, dtype=np.float32))
        kind = "card" if test.expected_card_key else "refusal" if test.expect_refusal else "passage"
        correct = (
            (best is not None and best[0] == test.expected_card_key)
            if kind == "card"
            else bool(passages) and passages[0].heading == test.expected_heading
        )
        return Observation(
            test.question,
            kind,
            correct,
            passages[0].similarity if passages else 0.0,
            best[2] if best else 0.0,
        )

    def _report(self, observations: list[Observation]) -> None:
        print(f"{'kind':8} {'top sim':>8} {'card':>6} {'right':>6}  question")
        for o in sorted(observations, key=lambda o: (o.kind, -o.top_similarity)):
            right = "-" if o.kind == "refusal" else str(o.correct_top)
            print(
                f"{o.kind:8} {o.top_similarity:8.3f} {o.card_score:6.3f} {right:>6}  {o.question}"
            )

        cards = [o for o in observations if o.kind == "card"]
        others = [o for o in observations if o.kind != "card"]
        card_rate = {
            round(t, 2): np.mean(
                [o.card_score >= t and o.correct_top for o in cards]
                + [o.card_score < t for o in others]
            )
            for t in self.CARD_RANGE
        }
        card_threshold = max(card_rate, key=lambda t: (card_rate[t], -abs(t - 0.8)))

        chunk_tests = [o for o in observations if o.kind != "card"]
        chunk_rate = {
            round(t, 2): np.mean(
                [
                    (o.top_similarity < t)
                    if o.kind == "refusal"
                    else (o.correct_top and o.top_similarity >= t)
                    for o in chunk_tests
                ]
            )
            for t in self.CHUNK_RANGE
        }
        best_rate = max(chunk_rate.values())
        band = [t for t, r in chunk_rate.items() if r == best_rate]
        print(f"\ncard_threshold:  {card_threshold}  (pass rate {card_rate[card_threshold]:.2f})")
        print(
            f"chunk_threshold: best pass rate {best_rate:.2f} for {min(band)} to {max(band)}; "
            f"the middle, {band[len(band) // 2]}, leaves the most margin"
        )


async def main() -> None:
    LogConfig.install("WARNING")
    container = Container()
    container.embedder.load()
    try:
        await Calibrator(container).run(uuid.UUID(sys.argv[1]))
    finally:
        await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
