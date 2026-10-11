from __future__ import annotations

import secrets


class TaskReference:
    """Short, unambiguous when read aloud: no O/0, no I/1, no S/5."""

    ALPHABET = "ACDEFHJKLMNPQRTUVWXY34679"
    LENGTH = 6

    @classmethod
    def generate(cls) -> str:
        return "".join(secrets.choice(cls.ALPHABET) for _ in range(cls.LENGTH))

    @classmethod
    def normalise(cls, value: str) -> str:
        return "".join(character for character in value.upper() if character in cls.ALPHABET)
