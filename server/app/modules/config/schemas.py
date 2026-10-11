from __future__ import annotations

import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import (
    Eagerness,
    FlagType,
    IdentityLevel,
    LatencyClass,
    Optimisation,
    Priority,
    SlotValueType,
)


class SkillWrite(BaseModel):
    # Schema for creating/updating a skill
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    name: str = Field(min_length=3, max_length=200)
    trigger: str
    goal: str
    steps: str
    guardrails: str | None = None
    done_when: str | None = None
    escalate_if: str | None = None
    knowledge_key: str | None = None
    tool_keys: list[str] = Field(default_factory=list)
    service_id: uuid.UUID | None = None
    enabled: bool = True
    position: int = 0


class SkillRead(SkillWrite):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    system: bool


class ValidationReport(BaseModel):
    # schema for validation errors (block the save) and warnings (returned with it)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class ModelProfileWrite(BaseModel):
    # Schema for creating/updating an organization's model profile
    stt_model: str | None = None
    stt_language: str | None = None
    llm_model: str | None = None
    tts_model: str | None = None
    voice: str | None = None
    eagerness: Eagerness = Eagerness.MEDIUM


class ModelProfileRead(ModelProfileWrite):
    # Schema for reading a model profile from the database
    model_config = ConfigDict(from_attributes=True)
    organization_id: uuid.UUID


class SkillSpec(BaseModel):
    # Immutable runtime representation of a skill
    model_config = ConfigDict(from_attributes=True, frozen=True)
    key: str
    name: str
    trigger: str
    goal: str
    steps: str
    guardrails: str | None
    done_when: str | None
    escalate_if: str | None
    knowledge_key: str | None
    tool_keys: tuple[str, ...]
    service_id: uuid.UUID | None
    system: bool


class ProfileSpec(BaseModel):
    # Immutable runtime representation of the model profile
    model_config = ConfigDict(from_attributes=True, frozen=True)
    stt_model: str | None = None
    stt_language: str | None = None
    llm_model: str | None = None
    tts_model: str | None = None
    voice: str | None = None
    eagerness: Eagerness = Eagerness.MEDIUM


class BindingWrite(BaseModel):
    # Schema for creating/updating a tool binding
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    implementation: str
    description: str = Field(min_length=20)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    latency_class: LatencyClass
    required_level: IdentityLevel = IdentityLevel.UNKNOWN
    timeout_ms: int = Field(default=2000, ge=50, le=30000)
    max_calls: int = Field(default=5, ge=1, le=50)
    filler_category: str | None = None
    fallback_text: str = Field(min_length=10)
    cache_ttl_s: int = Field(default=0, ge=0, le=3600)
    invalidates: list[str] = Field(default_factory=list)
    enabled: bool = True


class BindingRead(BindingWrite):
    # Schema for reading a tool binding from the database
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID


class BindingSaved(BaseModel):
    # schema returned after saving a binding, with its warnings
    binding: BindingRead
    warnings: list[str]


class BindingSpec(BaseModel):
    # Immutable runtime representation of a tool binding
    model_config = ConfigDict(from_attributes=True, frozen=True)
    key: str
    implementation: str
    description: str
    parameters: dict[str, Any]
    latency_class: LatencyClass
    required_level: IdentityLevel
    timeout_ms: int
    max_calls: int
    filler_category: str | None
    fallback_text: str
    cache_ttl_s: int
    invalidates: tuple[str, ...]


class CollectionSpec(BaseModel):
    """A published knowledge collection, as the agent sees it."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    key: str
    name: str
    is_default: bool
    active_version: int
    filter_keys: tuple[str, ...]
    injected_text: str | None = None

    @property
    def injectable(self) -> bool:
        return self.injected_text is not None


class SlotKeyWrite(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    description: str = Field(min_length=20)
    value_type: SlotValueType = SlotValueType.TEXT
    example: str = Field(min_length=1, max_length=200)
    spoken: bool = True
    enabled: bool = True


class SlotKeyRead(SlotKeyWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID


class SlotKeySpec(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    key: str
    description: str
    value_type: SlotValueType
    example: str
    spoken: bool

    @property
    def label(self) -> str:
        return self.key.replace("_", " ").capitalize()


class RouteWrite(BaseModel):
    work_item_type: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    description: str = Field(min_length=10)
    queue: str = Field(min_length=1, max_length=64)
    priority: Priority = Priority.NORMAL
    enabled: bool = True


class RouteRead(RouteWrite):
    model_config = ConfigDict(from_attributes=True)


class RouteSpec(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    work_item_type: str
    description: str
    queue: str
    priority: Priority


class DetectionSpec(BaseModel):
    """The tenant's thresholds: the platform defaults, with the tenant's rows on top."""

    model_config = ConfigDict(frozen=True)

    thresholds: dict[FlagType, float]
    disabled: frozenset[FlagType] = frozenset()
    severe_abuse: float

    def fires(self, flag_type: FlagType, score: float) -> bool:
        if flag_type in self.disabled or flag_type not in self.thresholds:
            return False
        return score >= self.thresholds[flag_type]


class OptimisationWrite(BaseModel):
    name: Optimisation
    percent: int = Field(ge=0, le=100)


class OptimisationRead(OptimisationWrite):
    model_config = ConfigDict(from_attributes=True)


class AgentConfig(BaseModel):
    # Immutable runtime configuration shared by the agent
    model_config = ConfigDict(frozen=True)
    HANDOFF_ROUTE: ClassVar[str] = "handoff"

    organization_id: uuid.UUID
    version: int
    skills: tuple[SkillSpec, ...]
    profile: ProfileSpec
    tools: tuple[BindingSpec, ...] = ()
    collections: tuple[CollectionSpec, ...] = ()
    slot_keys: tuple[SlotKeySpec, ...] = ()
    routes: tuple[RouteSpec, ...] = ()
    detection: DetectionSpec | None = None
    optimisations: dict[Optimisation, int] = Field(default_factory=dict)

    def route(self, work_item_type: str) -> RouteSpec | None:
        return next((r for r in self.routes if r.work_item_type == work_item_type), None)

    @property
    def work_item_types(self) -> frozenset[str]:
        """The types the agent may create. The handoff route is a queue for live
        transfers, not a type of work."""
        return frozenset(r.work_item_type for r in self.routes) - {self.HANDOFF_ROUTE}

    def slot_key(self, key: str) -> SlotKeySpec | None:
        return next((slot_key for slot_key in self.slot_keys if slot_key.key == key), None)

    def collection(self, key: str | None) -> CollectionSpec | None:
        return next((c for c in self.collections if c.key == key), None)

    @property
    def default_collections(self) -> tuple[CollectionSpec, ...]:
        return tuple(c for c in self.collections if c.is_default)

    def binding(self, key: str) -> BindingSpec | None:
        return next((binding for binding in self.tools if binding.key == key), None)
