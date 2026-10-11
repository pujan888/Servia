from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import TypeAdapter

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import Refusal, ToolContext, ToolOutcome, ToolResult
from app.agent.tools.outbox import OutboxService
from app.agent.tools.policy import PolicyEngine
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.utils import (
    ArgumentMasker,
    ArgumentNormaliser,
    IdempotencyKey,
    SpokenFormatter,
)
from app.core.cache import CacheKeys, CacheLayer
from app.core.enums import LatencyClass, SideEffect, ToolStatus
from app.modules.calls.journal import JournalDraft, ToolJournalWriter
from app.modules.config.schemas import BindingSpec
from app.modules.config.utils import SchemaGuard

if TYPE_CHECKING:
    from app.agent.state import CallState


class ToolGateway:
    """The one place every action goes through. The model requests; this decides.

    1 find the binding   2 validate the arguments   3 build the context
    4 evaluate policy    5 choose the path by class  6 apply the timeout
    7 turn the result into speech                   8 write the journal
    """

    NOT_BOUND = "There is no tool with that name. Do not try it again."
    OUTCOME = TypeAdapter(ToolOutcome)
    NOT_DONE = "NOT DONE. {text} Do not tell the caller that it is done."
    BAD_ARGUMENTS = "The details were not complete: {errors}. Ask the caller for what is missing."

    def __init__(
        self,
        registry: ToolRegistry,
        journal: ToolJournalWriter,
        outbox: OutboxService,
        cache: CacheLayer,
        policy: PolicyEngine | None = None,
    ) -> None:
        self._registry = registry
        self._journal = journal
        self._outbox = outbox
        self._cache = cache
        self._policy = policy or PolicyEngine()

    async def invoke(self, state: CallState, key: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        binding = state.config.binding(key)
        try:
            result = await self._invoke(state, binding, key, arguments)
        except Exception:
            logger.exception("tool {} crashed in the gateway", key)
            result = ToolResult(
                ToolStatus.FAILED, binding.fallback_text if binding else self.NOT_BOUND
            )
        result.latency_ms = round((time.perf_counter() - started) * 1000)
        self._observe(state, binding, result)
        if result.status in (ToolStatus.REFUSED, ToolStatus.FAILED):
            result.text = self.NOT_DONE.format(text=result.text)
        self._record(state, key, arguments, result)
        return result

    def _observe(self, state: CallState, binding: BindingSpec | None, result: ToolResult) -> None:
        """What the rest of the call needs to know: failures for the escalation
        policy, facts and confirmations for the output guard, references for the
        briefing."""
        signals = state.signals
        implementation = self._registry.get(binding.implementation) if binding else None
        if result.status == ToolStatus.FAILED:
            signals.tool_failures += 1
        elif result.status in (ToolStatus.OK, ToolStatus.EMPTY, ToolStatus.ACCEPTED):
            state.tool_texts.append(result.text)
        if result.status == ToolStatus.ACCEPTED:
            state.confirmed = True
        if result.reference and result.reference not in state.references:
            state.references.append(result.reference)
        if (
            result.status == ToolStatus.EMPTY
            and implementation
            and implementation.answers_questions
        ):
            signals.unanswered_turns.add(state.turn_index)

    async def _invoke(
        self, state: CallState, binding: BindingSpec | None, key: str, arguments: dict[str, Any]
    ) -> ToolResult:
        if binding is None:
            return self._refused(Refusal("not_bound", self.NOT_BOUND))
        implementation = self._registry.get(binding.implementation)
        if implementation is None:
            return ToolResult(ToolStatus.FAILED, binding.fallback_text, reason="not_registered")

        errors = SchemaGuard.argument_errors(binding.parameters, arguments)
        if errors:
            return self._refused(
                Refusal("invalid_arguments", self.BAD_ARGUMENTS.format(errors="; ".join(errors)))
            )

        ctx = self._context(state)
        refusal = self._policy.evaluate(
            binding, implementation, ctx, arguments, state.tool_counts[key]
        )
        if refusal is not None:
            return self._refused(refusal)
        state.tool_counts[key] += 1

        if (
            implementation.side_effect == SideEffect.WRITE
            or binding.latency_class == LatencyClass.C
        ):
            return await self._defer(state, binding, implementation, ctx, arguments)
        return await self._run(state, binding, implementation, ctx, arguments)

    async def _run(
        self,
        state: CallState,
        binding: BindingSpec,
        implementation: ToolImplementation,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        cache_key = self._cache_key(state, binding.key, arguments)
        if binding.cache_ttl_s:
            cached = await self._cache.peek(cache_key, self.OUTCOME)
            if cached is not None:
                return ToolResult(cached.status, cached.speech, from_cache=True)

        try:
            async with asyncio.timeout(binding.timeout_ms / 1000):
                outcome: ToolOutcome = await implementation.execute(ctx, arguments)
        except TimeoutError:
            return ToolResult(ToolStatus.FAILED, binding.fallback_text, reason="timeout")
        except Exception as error:
            logger.warning("tool {} failed: {!r}", binding.key, error)
            return ToolResult(ToolStatus.FAILED, binding.fallback_text, reason="error")

        if binding.cache_ttl_s and outcome.status == ToolStatus.OK:
            await self._cache.put(cache_key, outcome, self.OUTCOME, binding.cache_ttl_s)
        return ToolResult(outcome.status, outcome.speech, control=outcome.control)

    async def _defer(
        self,
        state: CallState,
        binding: BindingSpec,
        implementation: ToolImplementation,
        ctx: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        idempotency_key = IdempotencyKey.build(state.call_id, binding.key, arguments)
        receipt = await self._outbox.enqueue(
            ctx,
            binding.key,
            implementation.name,
            arguments,
            idempotency_key,
            implementation.receipt_eta,
        )
        for read_key in binding.invalidates:
            await self._cache.delete_prefix(
                CacheKeys.build(state.organization_id, "toolres", read_key, state.caller_id, "")
            )
        spoken = SpokenFormatter.reference(receipt.reference)
        text = (
            f"The request is logged. Tell the caller their reference, one character at a time: "
            f"{spoken}. Also tell them {receipt.eta}."
        )
        return ToolResult(
            ToolStatus.ACCEPTED,
            text,
            reference=receipt.reference,
            idempotency_key=idempotency_key,
        )

    def _context(self, state: CallState) -> ToolContext:
        scope = self._knowledge_scope(state)
        return ToolContext(
            organization_id=state.organization_id,
            call_id=state.call_id,
            caller_id=state.caller_id,
            identity_level=state.identity_level,
            turn_index=state.turn_index,
            timezone=state.pack.caller.timezone,
            tool_state=state.tool_state,
            known_references=frozenset(task.reference for task in state.pack.tasks),
            knowledge_scope=scope,
            knowledge_others=tuple(c for c in state.config.collections if c not in scope),
            caller_values={k: str(v) for k, v in state.pack.caller.details.items()},
            work_item_types=state.config.work_item_types,
            signals=self.signals(state),
        )

    @staticmethod
    def signals(state: CallState) -> dict[str, Any]:
        """What a deferred write needs from the call, stored with it: the job
        that runs it later has no call state."""
        return {
            "flags": sorted(state.signals.flag_types),
            "frustration": round(state.signals.frustration, 3),
            "turn_index": state.turn_index,
            "evidence": state.signals.last_caller_text,
            "transcript": [f"{speaker}: {text}" for speaker, text in state.recent(3)],
        }

    @staticmethod
    def _knowledge_scope(state: CallState) -> tuple:
        """The active skill's collection. With none, the tenant's default ones."""
        skill = next((s for s in state.config.skills if s.key == state.active_skill), None)
        chosen = state.config.collection(skill.knowledge_key if skill else None)
        return (chosen,) if chosen else state.config.default_collections

    def _cache_key(self, state: CallState, key: str, arguments: dict[str, Any]) -> str:
        return CacheKeys.build(
            state.organization_id,
            "toolres",
            key,
            state.caller_id,
            ArgumentNormaliser.digest(arguments),
        )

    def _refused(self, refusal: Refusal) -> ToolResult:
        return ToolResult(ToolStatus.REFUSED, refusal.instruction, reason=refusal.reason)

    def _record(
        self, state: CallState, key: str, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        self._journal.record(
            JournalDraft(
                organization_id=state.organization_id,
                call_id=state.call_id,
                turn_index=state.turn_index,
                tool_key=key,
                decision="refused" if result.status == ToolStatus.REFUSED else "permitted",
                status=result.status,
                arguments=ArgumentMasker.mask(arguments),
                reason=result.reason,
                latency_ms=result.latency_ms,
                result_text=result.text,
                idempotency_key=result.idempotency_key,
                from_cache=result.from_cache,
            )
        )
