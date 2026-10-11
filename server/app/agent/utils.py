from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


class SpokenTime:
    """The model never does date maths. Everything it reads is already phrased."""

    @staticmethod
    def zone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except Exception:
            return ZoneInfo("UTC")

    @classmethod
    def now(cls, timezone: str) -> str:
        moment = datetime.now(cls.zone(timezone))
        return moment.strftime("%A %-d %B, %-I:%M %p").replace("AM", "am").replace("PM", "pm")

    @classmethod
    def day(cls, moment: datetime | None, timezone: str) -> str | None:
        if moment is None:
            return None
        zone = cls.zone(timezone)
        local = moment.astimezone(zone)
        offset = (local.date() - datetime.now(zone).date()).days
        if offset == 0:
            return "today"
        if offset == 1:
            return "tomorrow"
        if offset == -1:
            return "yesterday"
        if 0 < offset < 7:
            return f"this {local.strftime('%A')}"
        return local.strftime("%-d %B")

    @classmethod
    def elapsed(cls, moment: datetime | None) -> str:
        if moment is None:
            return "previously"
        days = (datetime.now(UTC) - moment.astimezone(UTC)).days
        if days <= 0:
            return "earlier today"
        if days == 1:
            return "yesterday"
        if days < 14:
            return f"{days} days ago"
        return f"{days // 7} weeks ago"


class Section:
    """A prompt block that disappears when it has nothing to say."""

    def __init__(self, title: str) -> None:
        self._title = title
        self._lines: list[str] = []

    def add(self, line: str | None) -> None:
        if line:
            self._lines.append(line)

    def render(self) -> str | None:
        if not self._lines:
            return None
        return "\n".join([self._title, *self._lines])
