from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from datetime import date, timedelta

from app.modules.calls.models import TranscriptLine


class Sentences:
    BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

    @classmethod
    def split(cls, text: str) -> list[str]:
        return [part.strip() for part in cls.BOUNDARY.split(text.strip()) if part.strip()]

    @classmethod
    def first(cls, text: str, count: int) -> str:
        return " ".join(cls.split(text)[:count])


class TranscriptText:
    """The transcript as a model reads it. Each line keeps its number, so the
    model can point at the line that shows a fact."""

    @staticmethod
    def render(lines: Sequence[TranscriptLine]) -> str:
        return "\n".join(f"[{line.sequence}] {line.speaker}: {line.text}" for line in lines)


class DayResolver:
    """Dates in a summary are written by code. Measured, the model gave the wrong
    date for 'Thursday' in most runs, even with a calendar in the prompt.

    A weekday gets the date of the first such day on or after the call. A date
    the model wrote after it, in figures or in words, stays only if it is that
    weekday, on or after the call; otherwise the weekday wins, because the
    weekday is what the caller said."""

    WEEKDAYS = tuple(calendar.day_name)
    MONTHS = tuple(calendar.month_name)[1:]
    UNITS = tuple("first second third fourth fifth sixth seventh eighth ninth".split())
    ORDINALS = (
        *UNITS,
        *"tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth".split(),
        *"seventeenth eighteenth nineteenth twentieth".split(),
        *(f"twenty-{unit}" for unit in UNITS),
        "thirtieth",
        "thirty-first",
    )
    _day = rf"(\d{{1,2}}(?:st|nd|rd|th)?|{'|'.join(sorted(ORDINALS, key=len, reverse=True))})"
    _month = f"({'|'.join(MONTHS)})"
    DAY = re.compile(
        rf"(?<!every )(?<!each )\b({'|'.join(WEEKDAYS)})\b(?!s)"
        rf"(?:,? (?:(?:the )?{_day}(?: of)? {_month}|{_month} (?:the )?{_day})(?:,? \d{{4}})?)?"
    )

    @classmethod
    def resolve(cls, text: str, call_day: date) -> str:
        return cls.DAY.sub(lambda match: cls._replace(match, call_day), text)

    @classmethod
    def _replace(cls, match: re.Match, call_day: date) -> str:
        weekday = cls.WEEKDAYS.index(match.group(1))
        written = cls._written(match, call_day)
        if written and written.weekday() == weekday and written >= call_day:
            day = written
        else:
            day = call_day + timedelta(days=(weekday - call_day.weekday()) % 7)
        return f"{cls.WEEKDAYS[weekday]} {day.day} {cls.MONTHS[day.month - 1]}"

    @classmethod
    def _written(cls, match: re.Match, call_day: date) -> date | None:
        day_text = match.group(2) or match.group(5)
        if not day_text:
            return None
        month = cls.MONTHS.index(match.group(3) or match.group(4)) + 1
        number = cls._number(day_text)
        year = call_day.year + (1 if month < call_day.month else 0)
        try:
            return date(year, month, number)
        except ValueError:
            return None

    @classmethod
    def _number(cls, text: str) -> int:
        if text in cls.ORDINALS:
            return cls.ORDINALS.index(text) + 1
        return int(text.rstrip("stndrh"))
