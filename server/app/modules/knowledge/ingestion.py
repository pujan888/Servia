from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy import delete, select

from app.core.config import KnowledgeSettings
from app.core.database import Database
from app.core.tokens import TokenCounter
from app.modules.config.schemas import CollectionSpec
from app.modules.config.service import ConfigVersionService
from app.modules.jobs.models import Job
from app.modules.jobs.runner import JobHandler
from app.modules.knowledge.connectors import Connectors
from app.modules.knowledge.embedder import Embedder
from app.modules.knowledge.enricher import ContextualEnricher
from app.modules.knowledge.models import (
    AnswerCard,
    CardVariant,
    KnowledgeChunk,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgePublish,
    RetrievalTest,
)
from app.modules.knowledge.retrieval import KnowledgeRouter
from app.modules.knowledge.utils import ContentLinter, HeadingSplitter, MarkdownNormaliser
from app.modules.knowledge.verification import VerificationGate


class Publisher:
    """Builds one new version of a collection, tests it, and makes it live only
    if it passes. The live version is never changed in place."""

    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        enricher: ContextualEnricher,
        router: KnowledgeRouter,
        settings: KnowledgeSettings,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._enricher = enricher
        self._router = router
        self._settings = settings
        self._connectors = Connectors(settings.max_url_bytes)
        counter = TokenCounter()
        self._splitters = {
            "article": HeadingSplitter(counter, settings.chunk_tokens, settings.overlap_ratio),
            "faq": HeadingSplitter(counter, settings.faq_chunk_tokens, settings.overlap_ratio),
        }

    async def publish(self, publish_id: uuid.UUID) -> dict[str, Any]:
        started = time.perf_counter()
        async with self._database.session() as session:
            run = await session.get(KnowledgePublish, publish_id)
            collection = await session.get(KnowledgeCollection, run.collection_id)
            documents = await self._all(session, KnowledgeDocument, collection)
            cards = await self._all(session, AnswerCard, collection)
            tests = (
                (
                    await session.execute(
                        select(RetrievalTest).where(RetrievalTest.collection_id == collection.id)
                    )
                )
                .scalars()
                .all()
            )
            await self._clear(session, collection.id, run.version)
        org, version, previous = collection.organization_id, run.version, collection.active_version

        report: dict[str, Any] = {"documents": [], "warnings": [], "errors": []}
        for document in documents:
            await self._document(org, collection, version, previous, document, report)
        report["cards"] = await self._cards(org, collection, version, cards)

        if report["errors"]:
            return self._finish(report, started, passed=False, reason="source errors")

        candidate = CollectionSpec(
            id=collection.id,
            key=collection.key,
            name=collection.name,
            is_default=collection.is_default,
            active_version=version,
            filter_keys=tuple(collection.filter_keys),
        )
        gate = await VerificationGate(
            self._router, self._settings.min_tests, self._settings.gate_pass_rate
        ).run(org, candidate, tests)
        report["gate"] = gate.as_dict()
        if gate.passed:
            await self._activate(collection.id, org, version)
        return self._finish(report, started, passed=gate.passed, reason=gate.reason)

    async def _document(self, org, collection, version, previous, document, report) -> None:
        try:
            source = await self._connectors.for_document(document).fetch(document)
        except Exception as error:
            report["errors"].append({"document": document.key, "error": str(error)})
            return
        markdown = MarkdownNormaliser.normalise(source.content, source.source_format)
        digest = MarkdownNormaliser.digest(markdown)
        report["warnings"] += [
            {"document": document.key, "line": w.line, "kind": w.kind, "text": w.text}
            for w in ContentLinter.lint(markdown)
        ]
        copied = await self._copy_unchanged(org, collection.id, previous, version, document, digest)
        if copied:
            report["documents"].append({"key": document.key, "chunks": copied, "reused": True})
            return

        plans = self._splitters[document.kind].split(markdown, document.title)
        enriched = await self._enricher.enrich(plans)
        vectors = await self._embedder.passages(enriched)
        async with self._database.session() as session:
            session.add_all(
                KnowledgeChunk(
                    organization_id=org,
                    collection_id=collection.id,
                    version=version,
                    document_id=document.id,
                    document_hash=digest,
                    position=plan.position,
                    title=plan.title,
                    heading=plan.heading,
                    text=plan.text,
                    enriched_text=text,
                    metadata_=document.metadata_,
                    embedding=vector,
                    embedding_model=self._embedder.model_name,
                )
                for plan, text, vector in zip(plans, enriched, vectors, strict=True)
            )
        report["documents"].append({"key": document.key, "chunks": len(plans), "reused": False})

    async def _copy_unchanged(self, org, collection_id, previous, version, document, digest) -> int:
        """A document whose text and embedding model did not change keeps its
        chunks. No enrichment or embedding is paid for twice."""
        if previous is None:
            return 0
        async with self._database.session() as session:
            rows = (
                (
                    await session.execute(
                        select(KnowledgeChunk).where(
                            KnowledgeChunk.collection_id == collection_id,
                            KnowledgeChunk.version == previous,
                            KnowledgeChunk.document_id == document.id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not rows or any(
                r.document_hash != digest
                or r.embedding_model != self._embedder.model_name
                or r.metadata_ != document.metadata_
                for r in rows
            ):
                return 0
            session.add_all(
                KnowledgeChunk(
                    organization_id=org,
                    collection_id=collection_id,
                    version=version,
                    document_id=r.document_id,
                    document_hash=r.document_hash,
                    position=r.position,
                    title=r.title,
                    heading=r.heading,
                    text=r.text,
                    enriched_text=r.enriched_text,
                    metadata_=r.metadata_,
                    embedding=r.embedding,
                    embedding_model=r.embedding_model,
                )
                for r in rows
            )
        return len(rows)

    async def _cards(self, org, collection, version, cards) -> int:
        pairs = [(card, question) for card in cards for question in card.questions]
        vectors = await self._embedder.passages([q for _, q in pairs])
        async with self._database.session() as session:
            session.add_all(
                CardVariant(
                    organization_id=org,
                    collection_id=collection.id,
                    version=version,
                    card_key=card.key,
                    question=question,
                    answer=card.answer,
                    embedding=vector,
                )
                for (card, question), vector in zip(pairs, vectors, strict=True)
            )
        return len(cards)

    async def _activate(self, collection_id, org, version) -> None:
        keep_from = version - self._settings.keep_versions + 1
        async with self._database.session() as session:
            collection = await session.get(KnowledgeCollection, collection_id)
            collection.active_version = version
            await ConfigVersionService(session).bump(org)
            for model in (KnowledgeChunk, CardVariant):
                await session.execute(
                    delete(model).where(
                        model.collection_id == collection_id, model.version < keep_from
                    )
                )
        logger.info("collection {} version {} is live", collection_id, version)

    @staticmethod
    async def _clear(session, collection_id, version) -> None:
        for model in (KnowledgeChunk, CardVariant):
            await session.execute(
                delete(model).where(model.collection_id == collection_id, model.version == version)
            )

    @staticmethod
    async def _all(session, model, collection):
        result = await session.execute(
            select(model)
            .where(model.collection_id == collection.id, model.enabled.is_(True))
            .order_by(model.key)
        )
        return result.scalars().all()

    @staticmethod
    def _finish(report, started, passed: bool, reason: str | None) -> dict[str, Any]:
        report["passed"] = passed
        report["reason"] = reason
        report["seconds"] = round(time.perf_counter() - started, 2)
        return report


class PublisherFactory:
    @staticmethod
    def build(container) -> Publisher:
        settings = container.settings
        llm = settings.speech.llm_model if settings.knowledge.llm_enrichment else None
        return Publisher(
            container.database,
            container.embedder,
            ContextualEnricher(llm, settings.knowledge.enrichment_concurrency),
            container.knowledge_router,
            settings.knowledge,
        )


class PublishJobHandler(JobHandler):
    kind = "knowledge.publish"
    max_attempts = 2

    def __init__(self, database: Database, publisher: Publisher) -> None:
        self._database = database
        self._publisher = publisher

    async def handle(self, job: Job) -> None:
        publish_id = uuid.UUID(job.payload["publish_id"])
        await self._status(publish_id, "running")
        report = await self._publisher.publish(publish_id)
        await self._status(publish_id, "published" if report["passed"] else "failed", report)

    async def on_failed(self, job: Job, error: str) -> None:
        await self._status(uuid.UUID(job.payload["publish_id"]), "failed", {"errors": [error]})

    async def _status(self, publish_id: uuid.UUID, status: str, report: dict | None = None) -> None:
        async with self._database.session() as session:
            run = await session.get(KnowledgePublish, publish_id)
            run.status = status
            if report is not None:
                run.report = report
                run.finished_at = datetime.now(UTC)
