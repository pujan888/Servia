from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.enums import IdentityLevel, ToolStatus
from app.modules.config.schemas import CollectionSpec


@dataclass(slots=True)
class ToolContext:
    """What an implementation may know about the call. The gateway builds it.
    Identity is here and never in the arguments."""

    organization_id: uuid.UUID
    call_id: uuid.UUID
    caller_id: uuid.UUID
    identity_level: IdentityLevel
    turn_index: int
    timezone: str
    tool_state: dict[str, Any] = field(default_factory=dict)
    known_references: frozenset[str] = frozenset()
    knowledge_scope: tuple[CollectionSpec, ...] = ()
    knowledge_others: tuple[CollectionSpec, ...] = ()
    caller_values: dict[str, str] = field(default_factory=dict)
    work_item_types: frozenset[str] = frozenset()
    signals: dict[str, Any] = field(default_factory=dict)
    receipt_reference: str | None = None


@dataclass(slots=True)
class ToolOutcome:
    """What an implementation returns: a status and a sentence the agent can use."""

    status: ToolStatus
    speech: str
    control: str | None = None


@dataclass(slots=True, frozen=True)
class Refusal:
    reason: str
    instruction: str


@dataclass(slots=True)
class ToolResult:
    """What the gateway returns to the model."""

    status: ToolStatus
    text: str
    reason: str | None = None
    reference: str | None = None
    latency_ms: int | None = None
    control: str | None = None
    from_cache: bool = False
    idempotency_key: str | None = None
