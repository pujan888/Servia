from __future__ import annotations

from datetime import date

from app.core.enums import SlotSource, SlotValueType
from app.modules.memory.schemas import SlotDecision


class SlotValues:
    """Puts a value in its declared type. None means it is not a value of that type."""

    TRUE = frozenset({"true", "yes", "y", "1"})
    FALSE = frozenset({"false", "no", "n", "0"})

    @classmethod
    def normalise(cls, value: object, value_type: SlotValueType) -> str | None:
        text = " ".join(str(value).split())
        if not text:
            return None
        if value_type == SlotValueType.NUMBER:
            try:
                number = float(text)
            except ValueError:
                return None
            return str(int(number)) if number.is_integer() else str(number)
        if value_type == SlotValueType.BOOLEAN:
            lowered = text.casefold()
            return "yes" if lowered in cls.TRUE else "no" if lowered in cls.FALSE else None
        if value_type == SlotValueType.DATE:
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError:
                return None
        return text

    @staticmethod
    def same(first: str, second: str) -> bool:
        return first.casefold() == second.casefold()


class SlotPrecedence:
    """The fixed rule for a new value against the current one. The same inputs
    always give the same result, and no model is asked."""

    @staticmethod
    def decide(current: tuple[str, str] | None, new_source: str, new_value: str) -> SlotDecision:
        if current is None:
            return SlotDecision.INSERT
        current_source, current_value = current
        if SlotValues.same(current_value, new_value):
            return SlotDecision.CONFIRM
        if SlotSource(new_source).rank >= SlotSource(current_source).rank:
            return SlotDecision.REPLACE
        return SlotDecision.KEEP

    @staticmethod
    def higher(first: str, second: str) -> SlotSource:
        return max(SlotSource(first), SlotSource(second), key=lambda source: source.rank)
