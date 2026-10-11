from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from app.modules.context.schemas import ContextPack

if TYPE_CHECKING:
    from app.agent.skills.selector import SkillSelection
    from app.modules.config.schemas import CollectionSpec


class CacheTier(IntEnum):
    # Its job is to classify each prompt section by how often it changes,
    # so the prompt compiler can put the more stable sections first
    # and the frequently changing sections later.
    STATIC = 0
    TENANT = 1
    CALL = 2
    TURN = 3


@dataclass(slots=True)
class PromptInputs:
    pack: ContextPack
    skills: SkillSelection | None = None
    collections: tuple[CollectionSpec, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class Rendered:
    text: str | None
    truncations: list[str] = field(default_factory=list)


class PromptSection(ABC):
    """A section of a prompt, which can be rendered to text."""

    name: str
    tier: CacheTier
    budget_key: str

    @abstractmethod
    def render(self, inputs: PromptInputs, budget: int) -> Rendered: ...


# Budget = the maximum amount of text/tokens a section is allowed to use.
