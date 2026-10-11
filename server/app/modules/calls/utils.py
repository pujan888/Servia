from __future__ import annotations

import re
import secrets


class RoomName:
    PREFIX = "call"
    ENTROPY_BYTES = 8

    @classmethod
    def generate(cls) -> str:
        return f"{cls.PREFIX}-{secrets.token_hex(cls.ENTROPY_BYTES)}"


class ParticipantIdentity:
    PREFIX = "caller"

    @classmethod
    def for_user(cls, user_id: object) -> str:
        return f"{cls.PREFIX}-{user_id}"


class Redactor:
    """Masks values that must never be stored. It runs before anything is
    written, so the raw value exists only in the live call."""

    CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
    IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
    LONG_DIGITS = re.compile(r"\b\d{9,}\b")
    MARKS = {"card": "[card number]", "iban": "[bank account]", "digits": "[number]"}

    @classmethod
    def redact(cls, text: str | None) -> str | None:
        if not text:
            return text
        text = cls.CARD.sub(cls._card, text)
        text = cls.IBAN.sub(cls.MARKS["iban"], text)
        return cls.LONG_DIGITS.sub(cls.MARKS["digits"], text)

    @classmethod
    def _card(cls, match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        return cls.MARKS["card"] if cls.luhn(digits) else match.group()

    @staticmethod
    def luhn(digits: str) -> bool:
        total = 0
        for index, character in enumerate(reversed(digits)):
            value = int(character)
            if index % 2:
                value = value * 2 - 9 if value > 4 else value * 2
            total += value
        return total % 10 == 0
