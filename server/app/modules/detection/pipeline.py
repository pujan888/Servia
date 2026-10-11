from __future__ import annotations

import asyncio

from loguru import logger

from app.modules.config.schemas import DetectionSpec
from app.modules.detection.classifiers import Classifier, HumanRequestDetector
from app.modules.detection.schemas import Detection, FlagDraft, TurnWindow


class DetectionPipeline:
    """Layer 1. The rules run at once; the model runs next to the reply, never
    before it. A slow or failed classification is dropped, never retried: by
    then the turn has moved on."""

    def __init__(
        self, classifier: Classifier, rules: HumanRequestDetector, timeout_s: float
    ) -> None:
        self._classifier = classifier
        self._rules = rules
        self._timeout_s = timeout_s

    def rules(self, window: TurnWindow) -> FlagDraft | None:
        return self._rules.detect(window)

    async def detect(self, window: TurnWindow, spec: DetectionSpec) -> Detection | None:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._classifier.classify(window, spec)
        except TimeoutError:
            logger.warning("detection for turn {} timed out", window.turn_index)
        except Exception as error:
            logger.warning("detection for turn {} failed: {!r}", window.turn_index, error)
        return None
