from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.analysis.model import AnalysisModel
from app.core.config import DetectionSettings, WorkSettings
from app.core.database import Database
from app.core.enums import FlagType, Priority, TaskStatus
from app.core.errors import NotFound
from app.modules.config.models import EscalationRoute
from app.modules.knowledge.embedder import Embedder
from app.modules.work.models import Task, TaskEvidence
from app.modules.work.schemas import TaskCreate
from app.modules.work.service import TaskService


@dataclass(slots=True)
class WorkItemDraft:
    """What the call asks for. The model gives only the type and the
    description; everything else comes from the call."""

    organization_id: uuid.UUID
    user_id: uuid.UUID
    call_id: uuid.UUID
    reference: str
    kind: str
    description: str
    signals: dict[str, Any] = field(default_factory=dict)


class WorkItemSummary(BaseModel):
    title: str = Field(max_length=120)
    summary: str
    confidence: float = Field(ge=0, le=1)


class WorkItemRouter(ABC):
    """Chooses the queue. A later project replaces the table with skill-based
    and geographic routing; nothing else changes."""

    @abstractmethod
    async def route(self, organization_id: uuid.UUID, kind: str) -> EscalationRoute | None: ...


class TableRouter(WorkItemRouter):
    def __init__(self, database: Database) -> None:
        self._database = database

    async def route(self, organization_id: uuid.UUID, kind: str) -> EscalationRoute | None:
        async with self._database.session() as session:
            return await session.scalar(
                select(EscalationRoute).where(
                    EscalationRoute.organization_id == organization_id,
                    EscalationRoute.work_item_type == kind,
                    EscalationRoute.enabled.is_(True),
                )
            )


class WorkItemFactory:
    """Makes a work item, or adds to the open one about the same problem. Runs
    in the jobs worker, off the call's path."""

    INSTRUCTIONS = (
        "Write a work item for the person who will do this task. Give a short title, a "
        "summary of at most {sentences} sentences with every reference, name, date and "
        "number from the request, and your confidence from 0 to 1 that the request is "
        "clear enough to act on. Use only what is given."
    )

    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        model: AnalysisModel,
        router: WorkItemRouter,
        settings: WorkSettings,
        detection: DetectionSettings,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._model = model
        self._router = router
        self._settings = settings
        self._detection = detection
        self._instructions = self.INSTRUCTIONS.format(sentences=settings.summary_sentences)

    async def make_or_update(self, draft: WorkItemDraft) -> tuple[Task, bool]:
        """Returns the item and whether it is new. Safe to run again: a draft
        already stored under its reference is returned as it is."""
        async with self._database.session() as session:
            try:
                return await TaskService(session).get_by_reference(
                    draft.user_id, draft.reference
                ), False
            except NotFound:
                pass
        written = await self._model.structured(
            self._instructions, self._content(draft), WorkItemSummary
        )
        route = await self._router.route(draft.organization_id, draft.kind)
        duplicate = await self._duplicate(draft, written.summary)
        async with self._database.session() as session:
            if duplicate is not None:
                task = await session.get(Task, duplicate.id)
                task.update_count += 1
                session.add(self._evidence(draft, task.id))
                logger.info("work item {} updated from call {}", task.reference, draft.call_id)
                return task, False
            task = await TaskService(session).create(
                draft.organization_id,
                TaskCreate(
                    user_id=draft.user_id,
                    title=written.title,
                    details=written.summary,
                    status=TaskStatus.OPEN,
                ),
                reference=draft.reference,
            )
            task.kind = draft.kind
            task.queue = route.queue if route else None
            task.priority = self.priority(draft.signals, route)
            task.confidence = written.confidence
            session.add(self._evidence(draft, task.id))
        return task, True

    def priority(self, signals: dict[str, Any], route: EscalationRoute | None) -> Priority:
        flags = set(signals.get("flags", []))
        if flags & {FlagType.DISTRESS, FlagType.COMPLIANCE}:
            return Priority.URGENT
        if signals.get("frustration", 0.0) > self._detection.frustration_offer:
            return Priority.HIGH
        return Priority(route.priority) if route else Priority.NORMAL

    async def _duplicate(self, draft: WorkItemDraft, summary: str) -> Task | None:
        since = datetime.now(UTC) - timedelta(days=self._settings.duplicate_days)
        async with self._database.session() as session:
            candidates = await TaskService(session).open_work_items(
                draft.user_id, draft.kind, since
            )
        if not candidates:
            return None
        vectors = await self._embedder.passages(
            [summary, *(t.details or t.title for t in candidates)]
        )
        scored = [
            (self._cosine(vectors[0], v), t) for v, t in zip(vectors[1:], candidates, strict=True)
        ]
        best, task = max(scored, key=lambda pair: pair[0])
        logger.info("closest open {} item {}: similarity {:.3f}", draft.kind, task.reference, best)
        return task if best >= self._settings.duplicate_similarity else None

    @staticmethod
    def _cosine(first: list[float], second: list[float]) -> float:
        dot = sum(a * b for a, b in zip(first, second, strict=True))
        return dot / (math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second)))

    @staticmethod
    def _content(draft: WorkItemDraft) -> str:
        transcript = "\n".join(draft.signals.get("transcript", []))
        return (
            f"Type: {draft.kind}\nRequest: {draft.description}\n\n"
            f"The last part of the call:\n{transcript or '(none)'}"
        )

    @staticmethod
    def _evidence(draft: WorkItemDraft, task_id: uuid.UUID) -> TaskEvidence:
        quote = draft.signals.get("evidence") or draft.description
        return TaskEvidence(
            organization_id=draft.organization_id,
            task_id=task_id,
            call_id=draft.call_id,
            turn_index=draft.signals.get("turn_index"),
            quote=quote[:1000],
            reference=draft.reference,
        )
