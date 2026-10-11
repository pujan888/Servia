from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from app.agent.tools.utils import SpokenFormatter
from app.modules.detection.guard import Amounts


@dataclass(slots=True)
class ToolCall:
    turn: int
    key: str
    arguments: dict[str, Any]
    status: str
    text: str
    reference: str | None = None
    knowledge: bool = False


@dataclass(slots=True)
class TurnSeen:
    index: int
    caller: str
    agent: str
    skill: str | None
    ms: int
    unanswered: bool


@dataclass(slots=True)
class Observation:
    """What one replay did. Assertions read only this."""

    turns: list[TurnSeen]
    tools: list[ToolCall]
    escalation: str | None
    unexpected: list[str]
    known_text: str
    passages: dict[int, str] = field(default_factory=dict)

    @property
    def agent_text(self) -> str:
        return " ".join(turn.agent for turn in self.turns)


@dataclass(slots=True, frozen=True)
class Checked:
    passed: bool
    detail: str


class Assertion(ABC):
    """One examination of behaviour, never of exact words: the model says the
    same thing differently each run."""

    kind: ClassVar[str]

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    @abstractmethod
    def check(self, seen: Observation) -> Checked: ...


class ToolCalled(Assertion):
    kind = "tool_called"

    def check(self, seen: Observation) -> Checked:
        tool, expected = self.spec["tool"], self.spec.get("arguments", {})
        for call in seen.tools:
            if (
                call.key == tool
                and call.status not in ("refused", "failed")
                and all(
                    str(value).lower() in str(call.arguments.get(name, "")).lower()
                    for name, value in expected.items()
                )
            ):
                return Checked(True, f"{tool} at turn {call.turn}")
        return Checked(False, f"{tool} was not called with {expected or 'any arguments'}")


class ToolNotCalled(Assertion):
    kind = "tool_not_called"

    def check(self, seen: Observation) -> Checked:
        calls = [call.turn for call in seen.tools if call.key == self.spec["tool"]]
        return Checked(not calls, f"called at turns {calls}" if calls else "not called")


class ReferenceSaid(Assertion):
    kind = "reference_said"

    def check(self, seen: Observation) -> Checked:
        references = [call.reference for call in seen.tools if call.reference]
        if not references:
            return Checked(False, "no tool gave a reference")
        spoken = seen.agent_text.upper()
        missing = [
            ref
            for ref in references
            if ref not in spoken and SpokenFormatter.reference(ref).upper() not in spoken
        ]
        return Checked(not missing, f"not said: {missing}" if missing else f"said {references}")


class Refused(Assertion):
    """The knowledge had nothing, and the agent said so and offered a next step."""

    kind = "refused"
    OFFER = re.compile(
        r"call(?:\s+you)?\s+back|callback|colleague|find out|pass (?:it|this|you)", re.I
    )

    def check(self, seen: Observation) -> Checked:
        for turn in seen.turns:
            if turn.unanswered and self.OFFER.search(turn.agent):
                return Checked(True, f"refused at turn {turn.index}")
        return Checked(False, "no turn refused and offered a next step")


class Escalated(Assertion):
    kind = "escalated"

    def check(self, seen: Observation) -> Checked:
        reason = self.spec.get("reason")
        ok = seen.escalation is not None and (reason is None or seen.escalation == reason)
        return Checked(ok, f"escalation: {seen.escalation}")


class SkillLoaded(Assertion):
    kind = "skill_loaded"

    def check(self, seen: Observation) -> Checked:
        skill, turn = self.spec["skill"], self.spec.get("turn")
        active = [t.index for t in seen.turns if t.skill == skill]
        ok = bool(active) and (turn is None or turn in active)
        return Checked(ok, f"{skill} active at turns {active}")


class NoInventedPrice(Assertion):
    kind = "no_invented_price"

    def check(self, seen: Observation) -> Checked:
        invented = Amounts.find(seen.agent_text) - Amounts.find(seen.known_text)
        return Checked(not invented, f"invented: {sorted(invented)}" if invented else "none")


class TurnCount(Assertion):
    """The expected tool, or the escalation, came by a given turn."""

    kind = "turn_count"

    def check(self, seen: Observation) -> Checked:
        limit, tool = self.spec["max"], self.spec.get("tool")
        turns = [c.turn for c in seen.tools if c.key == tool] if tool else []
        reached = min(turns) if turns else None
        ok = reached is not None and reached <= limit
        return Checked(ok, f"reached at turn {reached}, limit {limit}")


class Assertions:
    TYPES: ClassVar[tuple[type[Assertion], ...]] = (
        ToolCalled,
        ToolNotCalled,
        ReferenceSaid,
        Refused,
        Escalated,
        SkillLoaded,
        NoInventedPrice,
        TurnCount,
    )

    @classmethod
    def build(cls, spec: dict[str, Any]) -> Assertion:
        for kind in cls.TYPES:
            if kind.kind == spec["type"]:
                return kind(spec)
        raise ValueError(f"unknown assertion type '{spec['type']}'")
