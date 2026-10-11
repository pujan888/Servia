from __future__ import annotations

import asyncio
from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import Refusal, ToolContext, ToolOutcome
from app.agent.tools.outbox import Receipt
from app.core.enums import SideEffect, ToolStatus


class FakeJournal:
    def __init__(self) -> None:
        self.entries = []

    def record(self, draft) -> None:
        self.entries.append(draft)


class FakeOutbox:
    def __init__(self) -> None:
        self.rows: dict[str, str] = {}

    async def enqueue(self, ctx, tool_key, implementation, arguments, idempotency_key, eta):
        existing = idempotency_key in self.rows
        reference = self.rows.setdefault(idempotency_key, f"REF{len(self.rows):03d}")
        return Receipt(reference, eta, existing=existing)


class FakeTool(ToolImplementation):
    def __init__(
        self,
        name: str,
        speech: str = "done",
        delay: float = 0,
        error: Exception | None = None,
        side_effect: SideEffect = SideEffect.NONE,
        precondition: Refusal | None = None,
    ) -> None:
        self.name = name
        self.side_effect = side_effect
        self._speech = speech
        self._delay = delay
        self._error = error
        self._precondition = precondition
        self.calls: list[ToolContext] = []

    def missing_precondition(self, ctx: ToolContext, arguments: dict[str, Any]) -> Refusal | None:
        return self._precondition

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        self.calls.append(ctx)
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return ToolOutcome(ToolStatus.OK, self._speech)


class FakeModel:
    """Stands in for AnalysisModel. Each call takes the next queued answer; a
    queued exception is raised instead."""

    name = "fake-model"

    def __init__(self, texts=(), structured=(), repeat: bool = False) -> None:
        self.texts = list(texts)
        self.answers = list(structured)
        self.repeat = repeat
        self.prompts: list[str] = []

    async def text(self, instructions: str, content: str) -> str:
        return self._next(self.texts, content)

    async def structured(self, instructions: str, content: str, schema):
        return schema.model_validate(self._next(self.answers, content))

    def _next(self, queue: list, content: str):
        self.prompts.append(content)
        answer = queue[0] if self.repeat and len(queue) == 1 else queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeCache:
    """The cache layer's interface in process memory, for tests without Redis."""

    def __init__(self) -> None:
        self.items: dict[str, object] = {}

    async def get(self, key, adapter, loader, ttl_s=None):
        if key not in self.items:
            self.items[key] = await loader()
        return self.items[key]

    async def peek(self, key, adapter):
        return self.items.get(key)

    async def put(self, key, value, adapter, ttl_s=None) -> None:
        self.items[key] = value

    async def delete_prefix(self, prefix: str) -> None:
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]
