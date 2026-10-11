from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from app.modules.calls.utils import Redactor


class ArgumentNormaliser:
    @staticmethod
    def normalise(arguments: dict[str, Any]) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return " ".join(value.split()).lower()
            if isinstance(value, dict):
                return {k: clean(v) for k, v in sorted(value.items()) if v not in (None, "")}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        return clean(arguments)

    @classmethod
    def digest(cls, arguments: dict[str, Any]) -> str:
        text = json.dumps(cls.normalise(arguments), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:32]


class IdempotencyKey:
    @staticmethod
    def build(call_id: uuid.UUID, tool_key: str, arguments: dict[str, Any]) -> str:
        text = f"{call_id}:{tool_key}:{ArgumentNormaliser.digest(arguments)}"
        return hashlib.sha256(text.encode()).hexdigest()[:48]


class ArgumentMasker:
    """Masks values that must not be stored in the journal."""

    SENSITIVE = re.compile(r"card|cvv|cvc|password|pin|secret|iban|account_number", re.I)

    @classmethod
    def mask(cls, arguments: dict[str, Any]) -> dict[str, Any]:
        masked = {}
        for key, value in arguments.items():
            if cls.SENSITIVE.search(key):
                masked[key] = "***"
            elif isinstance(value, str):
                masked[key] = Redactor.redact(value)
            else:
                masked[key] = value
        return masked


class SpokenFormatter:
    """Turns values into the words the agent should say."""

    DIGITS = "zero one two three four five six seven eight nine".split()

    @classmethod
    def reference(cls, reference: str) -> str:
        return ", ".join(
            cls.DIGITS[int(ch)] if ch.isdigit() else ch.upper() for ch in reference if ch.isalnum()
        )
