from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.modules.calls.models import ToolJournalEntry
from app.modules.calls.writer import BufferedWriter


@dataclass(slots=True)
class JournalDraft:
    organization_id: uuid.UUID
    call_id: uuid.UUID
    turn_index: int
    tool_key: str
    decision: str
    status: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    latency_ms: int | None = None
    result_text: str | None = None
    idempotency_key: str | None = None
    from_cache: bool = False


class ToolJournalWriter(BufferedWriter[JournalDraft]):
    def to_row(self, draft: JournalDraft) -> ToolJournalEntry:
        return ToolJournalEntry(
            organization_id=draft.organization_id,
            call_id=draft.call_id,
            turn_index=draft.turn_index,
            tool_key=draft.tool_key,
            arguments=draft.arguments,
            decision=draft.decision,
            reason=draft.reason,
            status=draft.status,
            latency_ms=draft.latency_ms,
            result_text=draft.result_text,
            idempotency_key=draft.idempotency_key,
            from_cache=draft.from_cache,
        )
