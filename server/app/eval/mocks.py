from __future__ import annotations

from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import ToolContext, ToolResult
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.utils import ArgumentNormaliser
from app.core.cache import CacheLayer
from app.core.enums import ToolStatus
from app.eval.assertions import ToolCall
from app.modules.config.schemas import BindingSpec


class MockToolLayer:
    """Answers a tool call with the answer recorded in the real call. A call
    with the same normalised arguments is matched first; then the next unused
    answer for that tool, because free-text arguments are worded differently
    on every run. A call with no recorded answer fails, and is reported."""

    UNEXPECTED = "The recorded conversation has no answer for this."

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = [dict(response) for response in responses]
        self._used: set[int] = set()
        self.unexpected: list[str] = []

    def answer(self, key: str, arguments: dict[str, Any]) -> ToolResult:
        index = self._match(key, arguments)
        if index is None:
            self.unexpected.append(key)
            return ToolResult(ToolStatus.FAILED, self.UNEXPECTED, reason="unexpected")
        self._used.add(index)
        response = self._responses[index]
        return ToolResult(
            ToolStatus(response["status"]), response["text"], reference=response.get("reference")
        )

    def _match(self, key: str, arguments: dict[str, Any]) -> int | None:
        digest = ArgumentNormaliser.digest(arguments)
        free = [
            i for i, r in enumerate(self._responses) if i not in self._used and r["tool_key"] == key
        ]
        same = [
            i for i in free if ArgumentNormaliser.digest(self._responses[i]["arguments"]) == digest
        ]
        return (same or free or [None])[0]


class ReplayGateway(ToolGateway):
    """The real gateway: the same schema checks, policy and journal. Only the
    execution differs: a tool whose answer depends on the release, such as the
    knowledge search, runs for real; every other tool gets its recorded answer,
    so a replay never writes to a real system."""

    def __init__(
        self, registry: ToolRegistry, journal, cache: CacheLayer, mocks: MockToolLayer
    ) -> None:
        super().__init__(registry, journal, outbox=None, cache=cache)
        self._mocks = mocks
        self.calls: list[ToolCall] = []

    async def invoke(self, state, key: str, arguments: dict[str, Any]) -> ToolResult:
        result = await super().invoke(state, key, arguments)
        binding = state.config.binding(key)
        implementation = self._registry.get(binding.implementation) if binding else None
        self.calls.append(
            ToolCall(
                state.turn_index,
                key,
                arguments,
                result.status,
                result.text,
                result.reference,
                knowledge=bool(implementation and implementation.answers_questions),
            )
        )
        return result

    async def _run(
        self,
        state,
        binding: BindingSpec,
        implementation: ToolImplementation,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if implementation.deterministic:
            return await super()._run(state, binding, implementation, ctx, arguments)
        return self._mocks.answer(binding.key, arguments)

    async def _defer(self, state, binding, implementation, ctx, arguments) -> ToolResult:
        return self._mocks.answer(binding.key, arguments)
