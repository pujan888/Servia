from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field


class Amounts:
    """Money amounts in a text, in figures or in words, as plain numbers. The
    speech rules ask the model to say money as a person says it, so 'forty nine
    pounds' must be found as well as '£49'."""

    SMALL = (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen"
    ).split()
    TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    CURRENCY = r"(?:pounds?|dollars?|euros?|quid|pence|p\b)"
    FIGURES = re.compile(
        rf"[£$€]\s?(\d[\d,]*(?:\.\d{{1,2}})?)|\b(\d[\d,]*(?:\.\d{{1,2}})?)\s?{CURRENCY}",
        re.IGNORECASE,
    )
    WORD = rf"(?:{'|'.join(SMALL + TENS)}|hundred|thousand)"
    WORDS = re.compile(rf"\b((?:{WORD}(?:[\s-]+(?:and[\s-]+)?)?)+)\s*{CURRENCY}", re.IGNORECASE)

    @classmethod
    def find(cls, text: str) -> set[str]:
        found = set()
        for match in cls.FIGURES.finditer(text):
            number = (match.group(1) or match.group(2)).replace(",", "")
            found.add(number.removesuffix(".00").removesuffix(".0"))
        for match in cls.WORDS.finditer(text):
            value = cls._words(match.group(1))
            if value is not None:
                found.add(str(value))
        return found

    @classmethod
    def _words(cls, text: str) -> int | None:
        total = current = 0
        seen = False
        for word in re.split(r"[\s-]+", text.lower().strip()):
            if word in cls.SMALL:
                current += cls.SMALL.index(word)
            elif word in cls.TENS:
                current += (cls.TENS.index(word) + 2) * 10
            elif word == "hundred":
                current = max(current, 1) * 100
            elif word == "thousand":
                total, current = total + max(current, 1) * 1000, 0
            else:
                continue
            seen = True
        return total + current if seen else None


@dataclass(slots=True)
class GuardContext:
    """What a sentence may say: facts the call has, not facts the model made."""

    known_text: str
    confirmed: bool
    caller_contacts: frozenset[str] = frozenset()
    blocked_phrases: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class GuardResult:
    text: str | None
    rule: str | None = None


class OutputGuard:
    """Layer 3: each sentence is examined before it is spoken. Rules only."""

    CONFIRMATION = re.compile(
        r"\b(?:is|are|has been|have been|was|were|now|all)\s+(?:confirmed|booked|scheduled|"
        r"arranged|approved|refunded|processed|cancelled|canceled)\b",
        re.IGNORECASE,
    )
    EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
    PRICE = "I will confirm the exact price for you."
    NOT_YET = "Let me finish that for you first."

    def check(self, sentence: str, context: GuardContext) -> GuardResult:
        if Amounts.find(sentence) - Amounts.find(context.known_text):
            return GuardResult(self.PRICE, "invented_price")
        if self.CONFIRMATION.search(sentence) and not context.confirmed:
            return GuardResult(self.NOT_YET, "unconfirmed")
        known = self._normal(context.known_text)
        for contact in self.EMAIL.findall(sentence) + self.PHONE.findall(sentence):
            normal = self._normal(contact)
            if normal not in context.caller_contacts and normal not in known:
                return GuardResult(None, "other_person_data")
        lowered = sentence.lower()
        if any(phrase.lower() in lowered for phrase in context.blocked_phrases):
            return GuardResult(None, "blocked_phrase")
        return GuardResult(sentence)

    @classmethod
    def contacts(cls, *values: str | None) -> frozenset[str]:
        return frozenset(cls._normal(value) for value in values if value)

    @staticmethod
    def _normal(value: str) -> str:
        return re.sub(r"[\s-]", "", value.lower())
