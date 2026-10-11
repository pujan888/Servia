# this only used for token counter nothing else
# we will implement a proper token counter later
from __future__ import annotations

import math


class TokenCounter:
    """
    Local estimate, so the budget needs no provider call.
    The provider's real count is recorded on every turn to keep this honest.
    """

    CHARS_PER_TOKEN = 4.0

    def __init__(self, margin: float = 1.1) -> None:
        self._margin = margin

    def count(self, text: str | None) -> int:
        if not text:
            return 0

        return math.ceil(len(text) / self.CHARS_PER_TOKEN * self._margin)
