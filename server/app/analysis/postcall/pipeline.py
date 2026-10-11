from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger
from sqlalchemy import type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.utils import SpokenTime
from app.analysis.model import AnalysisModel
from app.analysis.outcomes.classifier import OutcomeClassifier
from app.analysis.postcall.digest import DigestBuilder
from app.analysis.postcall.slots import SlotExtractor
from app.analysis.postcall.summariser import Summariser
from app.analysis.postcall.transcript import TranscriptBuilder
from app.analysis.postcall.utils import TranscriptText
from app.core.database import Database
from app.core.enums import FlagType, Severity, Speaker
from app.core.tokens import TokenCounter
from app.modules.accounts.service import OrganizationService, UserService
from app.modules.calls.models import Call, Handoff, ToolJournalEntry, TranscriptLine, TurnRecord
from app.modules.calls.service import (
    CallService,
    HandoffService,
    ToolJournalService,
    TranscriptService,
    TurnRecordService,
)
from app.modules.config.schemas import AgentConfig
from app.modules.detection.service import ReviewService
from app.modules.jobs.models import Job
from app.modules.jobs.runner import JobHandler
from app.modules.memory.store import MemoryStore
from app.modules.releases.service import ReleaseCatalog
from app.modules.work.models import ActionOutbox
from app.modules.work.service import ReceiptService


@dataclass(slots=True)
class CallMaterial:
    """Everything the pipeline reads. It comes from the database, never from the
    call process, which may be gone."""

    call: Call
    timezone: str
    config: AgentConfig
    turns: Sequence[TurnRecord]
    journal: Sequence[ToolJournalEntry]
    receipts: Sequence[ActionOutbox]
    handoffs: Sequence[Handoff] = ()
    lines: list[TranscriptLine] = field(default_factory=list)

    @property
    def call_day(self) -> date:
        moment = self.call.started_at or self.call.created_at
        return moment.astimezone(SpokenTime.zone(self.timezone)).date()

    @property
    def caller_spoke(self) -> bool:
        return any(line.speaker == Speaker.CALLER for line in self.lines)

    @property
    def references(self) -> list[str]:
        return [receipt.reference for receipt in self.receipts]

    @property
    def skills(self) -> list[str]:
        names = {skill.key: skill.name for skill in self.config.skills}
        keys = dict.fromkeys(turn.active_skill for turn in self.turns if turn.active_skill)
        return [names.get(key, key) for key in keys]


class StepLedger:
    """Each step commits its writes and its completion mark in one transaction.
    When the job runs again, the finished steps are skipped."""

    def __init__(self, database: Database, job: Job) -> None:
        self._database = database
        self._job_id = job.id
        self.progress: dict[str, Any] = dict(job.progress)

    def done(self, name: str) -> bool:
        return name in self.progress

    @asynccontextmanager
    async def step(
        self, name: str, started: float
    ) -> AsyncIterator[tuple[AsyncSession, dict[str, Any]]]:
        """`started` is when the step began, so its time includes the model."""
        report: dict[str, Any] = {}
        async with self._database.session() as session:
            yield session, report
            report["ms"] = round((time.perf_counter() - started) * 1000)
            await session.execute(
                update(Job)
                .where(Job.id == self._job_id)
                .values(progress=Job.progress.op("||")(type_coerce({name: report}, JSONB)))
            )
        self.progress[name] = report


class PostCallPipeline:
    """1 transcript  2 summary, slots and outcome at the same time  3 digest.
    The model calls run outside any transaction, so no lock is held while the
    model thinks. If one parallel step fails, the others still finish and are
    recorded before the job fails."""

    def __init__(
        self,
        database: Database,
        catalog: ReleaseCatalog,
        transcripts: TranscriptBuilder,
        summariser: Summariser,
        extractor: SlotExtractor,
        outcomes: OutcomeClassifier,
        digests: DigestBuilder,
    ) -> None:
        self._database = database
        self._catalog = catalog
        self._transcripts = transcripts
        self._summariser = summariser
        self._extractor = extractor
        self._outcomes = outcomes
        self._digests = digests

    async def run(self, job: Job) -> dict[str, Any]:
        started = time.perf_counter()
        ledger = StepLedger(self._database, job)
        material = await self._material(uuid.UUID(job.payload["call_id"]))
        material.lines = self._transcripts.build(
            material.call, material.turns, material.journal, material.receipts
        )
        await self._transcript(ledger, material)
        results = await asyncio.gather(
            self._summary(ledger, material),
            self._slots(ledger, material),
            self._outcome(ledger, material),
            self._review(ledger, material),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]
        await self._digest(ledger, material)
        logger.info(
            "post-call for {} done in {} ms: {}",
            material.call.id,
            round((time.perf_counter() - started) * 1000),
            ledger.progress,
        )
        return ledger.progress

    async def _material(self, call_id: uuid.UUID) -> CallMaterial:
        async with self._database.session() as session:
            call = await CallService(session).get(call_id)
            organization = await OrganizationService(session).get(call.organization_id)
            user = await UserService(session).get(call.user_id)
            turns = await TurnRecordService(session).list_for_call(call.organization_id, call_id)
            journal = await ToolJournalService(session).list_for_call(call.organization_id, call_id)
            receipts = await ReceiptService(session).list_for_call(call.organization_id, call_id)
            handoffs = await HandoffService(session).list_for_call(call.organization_id, call_id)
        return CallMaterial(
            call=call,
            timezone=user.timezone or organization.timezone,
            config=await self._catalog.config(call.organization_id, call.release_id),
            turns=turns,
            journal=journal,
            receipts=receipts,
            handoffs=handoffs,
        )

    async def _allowed(self, session: AsyncSession, material: CallMaterial) -> bool:
        """Takes the customer's lock. A call from before an erasure writes nothing."""
        call = material.call
        store = MemoryStore(session)
        await store.lock(call.user_id)
        return not await store.erased_since(call.organization_id, call.user_id, call.created_at)

    async def _transcript(self, ledger: StepLedger, material: CallMaterial) -> None:
        if ledger.done("transcript"):
            return
        started = time.perf_counter()
        async with ledger.step("transcript", started) as (session, report):
            if not await self._allowed(session, material):
                report["skipped"] = "erased"
                return
            await TranscriptService(session).replace(material.call, material.lines)
            report["lines"] = len(material.lines)

    async def _summary(self, ledger: StepLedger, material: CallMaterial) -> None:
        if ledger.done("summary"):
            return
        started = time.perf_counter()
        call = material.call
        text = None
        if material.caller_spoke:
            text = await self._summariser.summarise(
                TranscriptText.render(material.lines),
                material.call_day,
                material.skills,
                material.references,
            )
        async with ledger.step("summary", started) as (session, report):
            if text is None:
                report["skipped"] = "the caller said nothing"
                return
            if not await self._allowed(session, material):
                report["skipped"] = "erased"
                return
            await MemoryStore(session).save_summary(
                call.organization_id, call.user_id, call.id, text, self._summariser.model_name
            )
            report["text"] = text

    async def _slots(self, ledger: StepLedger, material: CallMaterial) -> None:
        if ledger.done("slots"):
            return
        started = time.perf_counter()
        call = material.call
        candidates = await self._extractor.extract(material.lines, material.config.slot_keys)
        async with ledger.step("slots", started) as (session, report):
            if not await self._allowed(session, material):
                report["skipped"] = "erased"
                return
            changes = await MemoryStore(session).write_slots(
                call.organization_id, call.user_id, call.id, candidates
            )
            report["changes"] = [change.model_dump(mode="json") for change in changes]

    async def _outcome(self, ledger: StepLedger, material: CallMaterial) -> None:
        if ledger.done("outcome"):
            return
        started = time.perf_counter()
        outcome, basis = await self._outcomes.classify(
            material.turns,
            material.receipts,
            TranscriptText.render(material.lines),
            material.handoffs,
        )
        async with ledger.step("outcome", started) as (session, report):
            await CallService(session).set_outcome(material.call.id, outcome)
            report.update(outcome=outcome.value, basis=basis)

    async def _review(self, ledger: StepLedger, material: CallMaterial) -> None:
        """Layer 4: a person looks at every turn where danger was flagged or the
        output guard changed what the agent said, and at every handoff that
        ended with nothing passed on."""
        if ledger.done("review"):
            return
        started = time.perf_counter()
        reasons = [
            (flag["type"], turn.turn_index)
            for turn in material.turns
            for flag in turn.flags
            if flag["severity"] == Severity.NOW or flag["type"] == FlagType.GUARD
        ]
        reasons += [("handoff_failed", None) for h in material.handoffs if h.level == 3]
        async with ledger.step("review", started) as (session, report):
            for reason, turn_index in reasons:
                await ReviewService(session).add(
                    material.call.organization_id, material.call.id, reason, turn_index
                )
            report["items"] = len(reasons)

    async def _digest(self, ledger: StepLedger, material: CallMaterial) -> None:
        if ledger.done("digest"):
            return
        started = time.perf_counter()
        call = material.call
        async with ledger.step("digest", started) as (session, report):
            if await self._allowed(session, material):
                store = MemoryStore(session)
                latest = await store.summaries(call.organization_id, call.user_id, limit=1)
                digest = self._digests.build(
                    latest[0] if latest else None,
                    await store.current_slots(call.organization_id, call.user_id),
                    material.config.slot_keys,
                )
                if digest.summary or digest.slots:
                    await store.save_digest(call.organization_id, call.user_id, digest, call.id)
                report["tokens"] = self._digests.tokens(digest)
            else:
                report["skipped"] = "erased"
            await CallService(session).mark_analysed(call.id)


class PostCallJobHandler(JobHandler):
    kind = CallService.POST_CALL_JOB

    def __init__(self, pipeline: PostCallPipeline) -> None:
        self._pipeline = pipeline

    async def handle(self, job: Job) -> None:
        await self._pipeline.run(job)


class PostCallFactory:
    @staticmethod
    def build(container) -> PostCallPipeline:
        settings = container.settings
        model = AnalysisModel(settings.speech.llm_model, settings.memory.llm_timeout_s)
        return PostCallPipeline(
            container.database,
            ReleaseCatalog(container.database, container.cache, settings),
            TranscriptBuilder(),
            Summariser(model, settings.memory.summary_sentences),
            SlotExtractor(model, settings.memory.min_confidence),
            OutcomeClassifier(model, container.registry),
            DigestBuilder(settings.memory, TokenCounter(settings.prompt.token_margin)),
        )
