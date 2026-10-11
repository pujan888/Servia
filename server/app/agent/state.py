from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agent.prompt.compiler import CompiledPrompt
from app.agent.prompt.section import PromptInputs
from app.agent.skills.selector import SkillSelection
from app.core.enums import IdentityLevel, Optimisation
from app.modules.config.schemas import AgentConfig
from app.modules.context.schemas import ContextPack
from app.modules.detection.schemas import CallSignals

if TYPE_CHECKING:
    from app.voice.greeting import PreparedGreeting


@dataclass(slots=True)
class CallState:
    """Session userdata. Identity lives here, never in a tool argument."""

    pack: ContextPack
    config: AgentConfig
    skills: SkillSelection
    compiled: CompiledPrompt
    active_skill: str | None = None
    skill_history: list[tuple[str, int]] = field(default_factory=list)
    skill_loaded_this_turn: str | None = None
    tool_state: dict[str, Any] = field(default_factory=dict)
    tool_counts: Counter = field(default_factory=Counter)
    tool_set_changed_this_turn: bool = False
    turn_index: int = 0
    last_prompt_ms: int | None = None
    flags: list[dict[str, Any]] = field(default_factory=list)
    signals: CallSignals = field(default_factory=CallSignals)
    transcript: list[tuple[int, str, str]] = field(default_factory=list)
    next_turn_notes: list[str] = field(default_factory=list)
    rendered_notes: tuple[str, ...] = ()
    briefing: str = ""
    briefing_turn: int = 0
    turn_relevance: str | None = None
    turn_guard_actions: list[str] = field(default_factory=list)
    tool_texts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    confirmed: bool = False
    last_reply_interrupted: bool = False
    pending_handoff: str | None = None
    optimisations: frozenset[Optimisation] = frozenset()
    release_id: uuid.UUID | None = None
    greeting: PreparedGreeting | None = None

    @property
    def inputs(self) -> PromptInputs:
        return PromptInputs(
            pack=self.pack,
            skills=self.skills,
            collections=self.config.collections,
            notes=tuple(self.next_turn_notes),
        )

    @property
    def call_id(self) -> uuid.UUID:
        return self.pack.call_id

    @property
    def organization_id(self) -> uuid.UUID:
        return self.pack.organization.id

    @property
    def caller_id(self) -> uuid.UUID:
        return self.pack.caller.id

    @property
    def identity_level(self) -> IdentityLevel:
        return self.pack.identity_level

    def recent(self, turns: int) -> list[tuple[str, str]]:
        """The last lines of the call, as (speaker, text)."""
        last = set(sorted({index for index, _, _ in self.transcript})[-turns:])
        return [(speaker, text) for index, speaker, text in self.transcript if index in last]

    def activate_skill(self, key: str) -> None:
        self.active_skill = key
        self.skill_history.append((key, self.turn_index))
        self.skill_loaded_this_turn = key


# The high-level goal of CallState is very simple:

# CallState is the central working-memory object for ONE active call.

# Instead of passing around 10 different things everywhere, the agent keeps the important state of
# the current call inside one object.

# You need CallState because the agent has to remember things while one call is running.


# Google/its model infrastructure provides the cache. We are arranging our prompt so Google's cache
# can actually be used.
