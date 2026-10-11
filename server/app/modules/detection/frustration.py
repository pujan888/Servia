from __future__ import annotations

import math
import re
from collections import Counter

from app.modules.detection.schemas import CallSignals


class WordOverlap:
    """Word-overlap cosine between two caller messages, after common words are
    removed. Used only to notice that the caller repeated a request; skills have
    no similarity check."""

    STOPWORDS = frozenset(
        "a an and are as ask asks at be but by caller for from has have if in into is it "
        "not of on or that the their them they this to wants want when with who you your".split()
    )
    WORD = re.compile(r"[a-z]+")

    @classmethod
    def words(cls, text: str) -> Counter:
        return Counter(w for w in cls.WORD.findall(text.lower()) if w not in cls.STOPWORDS)

    @classmethod
    def score(cls, first: str, second: str) -> float:
        a, b = cls.words(first), cls.words(second)
        dot = sum(a[word] * b[word] for word in a.keys() & b.keys())
        norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return dot / norm if norm else 0.0


class FrustrationTracker:
    """An exponentially weighted average across the call. A repeated request
    and an interruption each add to the turn's own score."""

    WEIGHT = 0.6
    REPEAT_BONUS = 0.2
    INTERRUPT_BONUS = 0.1
    REPEAT_SIMILARITY = 0.6

    def update(
        self, signals: CallSignals, negativity: float, text: str, interrupted: bool
    ) -> float:
        turn = negativity
        if signals.last_caller_text and (
            WordOverlap.score(text, signals.last_caller_text) >= self.REPEAT_SIMILARITY
        ):
            turn += self.REPEAT_BONUS
        if interrupted:
            turn += self.INTERRUPT_BONUS
        signals.frustration = self.WEIGHT * min(turn, 1.0) + (1 - self.WEIGHT) * signals.frustration
        signals.last_caller_text = text
        return signals.frustration
