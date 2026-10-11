from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import Refusal, ToolContext
from app.modules.config.schemas import BindingSpec


class PolicyRule(ABC):
    @abstractmethod
    def evaluate(
        self,
        binding: BindingSpec,
        implementation: ToolImplementation,
        ctx: ToolContext,
        arguments: dict[str, Any],
        calls_so_far: int,
    ) -> Refusal | None: ...


class IdentityRule(PolicyRule):
    INSTRUCTION = (
        "You cannot do this until the caller is verified. Offer to pass them to a colleague."
    )

    def evaluate(self, binding, implementation, ctx, arguments, calls_so_far) -> Refusal | None:
        if ctx.identity_level >= binding.required_level:
            return None
        return Refusal("identity_too_low", self.INSTRUCTION)


class PreconditionRule(PolicyRule):
    def evaluate(self, binding, implementation, ctx, arguments, calls_so_far) -> Refusal | None:
        return implementation.missing_precondition(ctx, arguments)


class RateRule(PolicyRule):
    INSTRUCTION = "Do not try this again in this call. Offer to take the details for a colleague."

    def evaluate(self, binding, implementation, ctx, arguments, calls_so_far) -> Refusal | None:
        if calls_so_far < binding.max_calls:
            return None
        return Refusal("rate_exceeded", self.INSTRUCTION)


class PolicyEngine:
    """Evaluates the rules in order. The first refusal wins."""

    def __init__(self, rules: Sequence[PolicyRule] | None = None) -> None:
        self._rules = list(rules or (IdentityRule(), PreconditionRule(), RateRule()))

    def evaluate(
        self,
        binding: BindingSpec,
        implementation: ToolImplementation,
        ctx: ToolContext,
        arguments: dict[str, Any],
        calls_so_far: int,
    ) -> Refusal | None:
        for rule in self._rules:
            refusal = rule.evaluate(binding, implementation, ctx, arguments, calls_so_far)
            if refusal is not None:
                return refusal
        return None
