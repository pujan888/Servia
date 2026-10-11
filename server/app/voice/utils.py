from __future__ import annotations


def milliseconds(value: float | None) -> int | None:
    """Seconds from the framework to whole milliseconds. A negative value means not measured."""
    if value is None or value < 0:
        return None
    return round(value * 1000)
