from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import Refusal, ToolContext, ToolOutcome
from app.agent.tools.utils import SpokenFormatter
from app.agent.utils import SpokenTime
from app.core.database import Database
from app.core.enums import CallOutcome, SideEffect, ToolStatus
from app.core.errors import NotFound
from app.modules.accounts.service import StaffService
from app.modules.work.factory import WorkItemDraft, WorkItemFactory
from app.modules.work.models import Task
from app.modules.work.service import TaskService
from app.modules.work.utils import TaskReference


class WorkState:
    """Keys that the work tools share through the call's tool_state."""

    VIEWED = "viewed_work_items"

    @classmethod
    def mark_viewed(cls, ctx: ToolContext, reference: str) -> None:
        ctx.tool_state.setdefault(cls.VIEWED, set()).add(reference)

    @classmethod
    def is_known(cls, ctx: ToolContext, reference: str) -> bool:
        """True if the reference came from data the agent was given: the context
        pack, or a lookup in this call. False means the model may have invented it."""
        return reference in ctx.known_references or reference in ctx.tool_state.get(
            cls.VIEWED, set()
        )


class GetOpenWork(ToolImplementation):
    name = "get_open_work"
    LIMIT = 5

    def __init__(self, database: Database) -> None:
        self._database = database

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        async with self._database.session() as session:
            tasks = await TaskService(session).list_for_user(
                ctx.caller_id, open_only=True, limit=self.LIMIT
            )
        if not tasks:
            return ToolOutcome(
                ToolStatus.EMPTY, "They have no open work. Ask what they need help with."
            )
        lines = [
            f'"{t.title}", reference {SpokenFormatter.reference(t.reference)}, '
            f"{t.status.replace('_', ' ')}"
            for t in tasks
        ]
        return ToolOutcome(ToolStatus.OK, "Their open work: " + "; ".join(lines) + ".")


class GetWorkItem(ToolImplementation):  # give me details of the secific item
    name = "get_work_item"

    def __init__(self, database: Database) -> None:
        self._database = database

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        reference = TaskReference.normalise(arguments["reference"])
        async with self._database.session() as session:
            try:
                task = await TaskService(session).get_by_reference(ctx.caller_id, reference)
            except NotFound:
                return ToolOutcome(
                    ToolStatus.EMPTY,
                    f"Nothing on this account matches {SpokenFormatter.reference(reference)}. "
                    "Ask them to read it again, one character at a time.",
                )
            staff = await StaffService(session).map_by_id(
                [task.assignee_id] if task.assignee_id else []
            )
        WorkState.mark_viewed(ctx, task.reference)
        return ToolOutcome(ToolStatus.OK, self._describe(task, staff.get(task.assignee_id), ctx))

    def _describe(self, task: Task, assignee: Any, ctx: ToolContext) -> str:
        parts = [f'"{task.title}", currently {task.status.replace("_", " ")}']
        if assignee is not None:
            team = f" on the {assignee.team} team" if assignee.team else ""
            parts.append(f"handled by {assignee.full_name}{team}")
        due = SpokenTime.day(task.due_at, ctx.timezone)
        if due:
            parts.append(f"due {due}")
        if task.details:
            parts.append(f"notes: {task.details}")
        return ". ".join(parts) + "."


# Add a note to an existing job.
# For example:
# Customer:
# "The leak is getting worse. Add that to my job."


class AddWorkNote(ToolImplementation):
    name = "add_work_note"
    side_effect = SideEffect.WRITE
    receipt_eta = "the team will see it on the job today"
    receipt_outcome = CallOutcome.RESOLVED

    def __init__(self, database: Database) -> None:
        self._database = database

    def missing_precondition(self, ctx: ToolContext, arguments: dict[str, Any]) -> Refusal | None:
        reference = TaskReference.normalise(arguments.get("reference", ""))
        if WorkState.is_known(ctx, reference):
            return None
        return Refusal(
            "unknown_reference",
            "That reference is not one you were given. Look it up with get_work_item, "
            "and confirm it with the caller.",
        )

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        reference = TaskReference.normalise(arguments["reference"])
        tag = f"[note {ctx.receipt_reference}]"
        async with self._database.session() as session:
            task = await TaskService(session).get_by_reference(ctx.caller_id, reference)
            if tag in (task.details or ""):
                return ToolOutcome(ToolStatus.OK, "The note was already added.")
            stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            note = f"{tag} {stamp}, from the caller: {arguments['note'].strip()}"
            task.details = f"{task.details}\n{note}" if task.details else note
        return ToolOutcome(ToolStatus.OK, "The note was added to the job.")


class RequestCallback(ToolImplementation):  # callback escalte human callback
    """A work item of type `callback`, through the same factory as every other
    work item, so a second call about the same thing adds to the first."""

    name = "request_callback"
    side_effect = SideEffect.WRITE
    receipt_eta = "someone from the team will call back within one working day"
    receipt_outcome = CallOutcome.CALLBACK_PROMISED
    KIND = "callback"

    def __init__(self, factory: WorkItemFactory) -> None:
        self._factory = factory

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        description = f"{arguments['topic'].strip()}: {arguments['reason'].strip()}"
        if arguments.get("preferred_time"):
            description += f" Preferred time: {arguments['preferred_time'].strip()}."
        _, created = await self._factory.make_or_update(
            WorkItemDrafts.build(ctx, self.KIND, description)
        )
        return ToolOutcome(
            ToolStatus.OK, "The callback was logged." if created else "Added to the open request."
        )


class CreateWorkItem(ToolImplementation):
    """A task for a person. The model gives the type and a description; the
    priority, the queue and the customer come from the call and the tenant."""

    name = "create_work_item"
    side_effect = SideEffect.WRITE
    receipt_eta = "the right team will pick it up"
    receipt_outcome = CallOutcome.CALLBACK_PROMISED

    def __init__(self, factory: WorkItemFactory) -> None:
        self._factory = factory

    def missing_precondition(self, ctx: ToolContext, arguments: dict[str, Any]) -> Refusal | None:
        if arguments.get("type") in ctx.work_item_types:
            return None
        return Refusal(
            "unknown_type",
            f"That type does not exist. Use one of: {', '.join(sorted(ctx.work_item_types))}.",
        )

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        draft = WorkItemDrafts.build(ctx, arguments["type"], arguments["description"].strip())
        _, created = await self._factory.make_or_update(draft)
        return ToolOutcome(
            ToolStatus.OK, "The work item was made." if created else "Added to the open item."
        )


class WorkItemDrafts:
    @staticmethod
    def build(ctx: ToolContext, kind: str, description: str) -> WorkItemDraft:
        return WorkItemDraft(
            organization_id=ctx.organization_id,
            user_id=ctx.caller_id,
            call_id=ctx.call_id,
            reference=ctx.receipt_reference,
            kind=kind,
            description=description,
            signals=ctx.signals,
        )
