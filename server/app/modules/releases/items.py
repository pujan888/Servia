from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from app.modules.config.schemas import (
    AgentConfig,
    BindingSpec,
    CollectionSpec,
    DetectionSpec,
    ProfileSpec,
    RouteSpec,
    SkillSpec,
    SlotKeySpec,
)


@dataclass(slots=True, frozen=True)
class ReleaseItem:
    item_type: str
    item_key: str
    position: int
    content: Any
    hash: str


class ReleaseItems:
    """An agent configuration as release items, and back. Each item is
    serialised the same way every time, so the same configuration always gives
    the same hashes."""

    LISTS = (
        ("skills", "key", SkillSpec),
        ("tools", "key", BindingSpec),
        ("collections", "key", CollectionSpec),
        ("slot_keys", "key", SlotKeySpec),
        ("routes", "work_item_type", RouteSpec),
    )
    SINGLES = (("profile", ProfileSpec), ("detection", DetectionSpec))
    PROMPT = "prompt"
    OPTIMISATIONS = "optimisations"

    @classmethod
    def from_config(cls, config: AgentConfig, prompt_fingerprint: str) -> list[ReleaseItem]:
        items = [cls._item(cls.PROMPT, "template", 0, {"fingerprint": prompt_fingerprint})]
        for field, key, _ in cls.LISTS:
            for position, spec in enumerate(getattr(config, field)):
                content = spec.model_dump(mode="json")
                items.append(cls._item(field, content[key], position, content))
        for field, _ in cls.SINGLES:
            value = getattr(config, field)
            if value is not None:
                items.append(cls._item(field, field, 0, value.model_dump(mode="json")))
        optimisations = {str(name): percent for name, percent in config.optimisations.items()}
        items.append(cls._item(cls.OPTIMISATIONS, cls.OPTIMISATIONS, 0, optimisations))
        return items

    @classmethod
    def to_config(
        cls, organization_id: uuid.UUID, version: int, items: list[ReleaseItem]
    ) -> AgentConfig:
        ordered = sorted(items, key=lambda item: (item.item_type, item.position))
        by_type: dict[str, list[Any]] = {}
        for item in ordered:
            by_type.setdefault(item.item_type, []).append(item.content)
        values: dict[str, Any] = {
            field: tuple(spec.model_validate(c) for c in by_type.get(field, []))
            for field, _, spec in cls.LISTS
        }
        for field, spec in cls.SINGLES:
            if by_type.get(field):
                values[field] = spec.model_validate(by_type[field][0])
        values["optimisations"] = (by_type.get(cls.OPTIMISATIONS) or [{}])[0]
        return AgentConfig(organization_id=organization_id, version=version, **values)

    @staticmethod
    def release_hash(items: list[ReleaseItem]) -> str:
        lines = sorted(f"{i.item_type}:{i.position}:{i.item_key}:{i.hash}" for i in items)
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    @classmethod
    def _item(cls, item_type: str, item_key: str, position: int, content: Any) -> ReleaseItem:
        text = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return ReleaseItem(
            item_type, item_key, position, content, hashlib.sha256(text.encode()).hexdigest()
        )
