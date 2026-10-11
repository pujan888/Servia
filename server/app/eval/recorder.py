from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import ToolImplementation
from app.agent.tools.registry import ToolRegistry
from app.core.enums import Mark, SideEffect
from app.core.errors import Conflict
from app.modules.calls.service import (
    CallContextService,
    CallService,
    HandoffService,
    ToolJournalService,
    TurnRecordService,
)
from app.modules.config.schemas import AgentConfig
from app.modules.evaluation.models import Annotation, Scenario
from app.modules.releases.service import ReleaseCatalog
from app.modules.work.service import ReceiptService


class ScenarioRecorder:
    """Makes a scenario from a real call: what the agent was given, what the
    caller said, what each tool answered, and the behaviour to expect. It
    starts disabled; a person reads it once and enables it."""

    def __init__(
        self, session: AsyncSession, registry: ToolRegistry, catalog: ReleaseCatalog
    ) -> None:
        self._session = session
        self._registry = registry
        self._catalog = catalog

    async def record(self, call_id: uuid.UUID) -> Scenario:
        call = await CallService(self._session).get(call_id)
        org = call.organization_id
        context = await CallContextService(self._session).get(org, call_id)
        if context is None:
            raise Conflict("The call has no saved context to replay", call_id=str(call_id))
        turns = await TurnRecordService(self._session).list_for_call(org, call_id)
        journal = await ToolJournalService(self._session).list_for_call(org, call_id)
        receipts = await ReceiptService(self._session).list_for_call(org, call_id)
        handoffs = await HandoffService(self._session).list_for_call(org, call_id)
        references = {r.idempotency_key: r.reference for r in receipts}
        annotations = (
            await self._session.scalars(select(Annotation).where(Annotation.call_id == call_id))
        ).all()
        config = await self._catalog.config(org, call.release_id)

        responses, assertions = [], [{"type": "no_invented_price"}]
        for entry in journal:
            implementation = self._implementation(config, entry.tool_key)
            if entry.decision != "permitted" or implementation is None:
                continue
            if implementation.answers_questions:
                if entry.status == "empty":
                    assertions.append({"type": "refused"})
                continue
            if implementation.deterministic:
                continue
            reference = references.get(entry.idempotency_key)
            responses.append(
                {
                    "tool_key": entry.tool_key,
                    "arguments": entry.arguments,
                    "status": entry.status,
                    "text": entry.result_text or "",
                    "reference": reference,
                }
            )
            assertions.append({"type": "tool_called", "tool": entry.tool_key})
            if implementation.side_effect == SideEffect.WRITE and reference:
                assertions.append({"type": "reference_said"})
        for skill in dict.fromkeys(t.skill_loaded for t in turns if t.skill_loaded):
            assertions.append({"type": "skill_loaded", "skill": skill})
        for handoff in handoffs:
            assertions.append({"type": "escalated", "reason": handoff.reason})

        scenario = Scenario(
            organization_id=org,
            name=f"{context.pack['caller']['full_name']}, call of {call.created_at:%Y-%m-%d %H:%M}",
            source_call_id=call_id,
            pack=context.pack,
            caller_turns=[t.transcript_in for t in turns if t.transcript_in],
            responses=responses,
            assertions=self._unique(assertions),
            review_notes=[
                f"turn {a.turn_index}: marked bad by {a.person}: {a.note}"
                for a in annotations
                if a.mark == Mark.BAD
            ],
        )
        self._session.add(scenario)
        await self._session.flush()
        return scenario

    def _implementation(self, config: AgentConfig, tool_key: str) -> ToolImplementation | None:
        """The call's own release says which code each tool key ran."""
        binding = config.binding(tool_key)
        return self._registry.get(binding.implementation) if binding else None

    @staticmethod
    def _unique(assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen, unique = set(), []
        for assertion in assertions:
            key = tuple(sorted(assertion.items()))
            if key not in seen:
                seen.add(key)
                unique.append(assertion)
        return unique
