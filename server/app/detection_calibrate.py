"""Measures the detection model on labelled caller turns and suggests thresholds.

    uv run python -m app.detection_calibrate [model ...]

A normal turn that fires any flag is a false positive. The plan's target is
under 2% of normal turns."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

from app.analysis.model import AnalysisModel
from app.core.config import SettingsFactory
from app.core.enums import FlagType, Relevance
from app.core.logging import LogConfig
from app.modules.detection.classifiers import TurnClassifier
from app.modules.detection.schemas import TurnAssessment, TurnWindow


class DetectionCalibrator:
    SAMPLES = Path(__file__).parent / "seed_data" / "detection_samples.json"
    SCORED = (
        FlagType.ABUSE,
        FlagType.DISTRESS,
        FlagType.MANIPULATION,
        FlagType.COMPLIANCE,
        FlagType.HUMAN_REQUEST,
    )
    CANDIDATES = (0.5, 0.6, 0.7, 0.8, 0.9)
    MAX_FALSE_POSITIVE = 0.02
    REQUESTS_PER_MINUTE = 80

    def __init__(self, model: AnalysisModel) -> None:
        self._model = model
        self._pace = asyncio.Lock()

    async def run(self) -> None:
        samples = json.loads(self.SAMPLES.read_text())
        normal = await self._assess([(None, *row) for row in samples["normal"]])
        flagged = await self._assess([tuple(row) for row in samples["flagged"]])
        timings = [ms for _, _, ms in normal + flagged]
        print(f"\n{self._model.name}: {len(normal)} normal, {len(flagged)} flagged turns")
        print(f"  latency median {statistics.median(timings):.0f} ms, max {max(timings):.0f} ms")
        chosen = {
            flag_type: self._threshold(flag_type, normal, flagged) for flag_type in self.SCORED
        }
        self._relevance(normal, flagged)
        false_positive = [
            text
            for label, assessment, _ in normal
            if any(getattr(assessment, t.value) >= chosen[t] for t in self.SCORED)
            or assessment.relevance in (Relevance.OFF_TOPIC, Relevance.NONSENSE)
            for text in [label]
        ]
        print(
            f"  with these thresholds: {len(false_positive)} of {len(normal)} normal turns flagged"
        )
        for text in false_positive:
            print(f"    - {text}")

    async def _assess(self, rows) -> list[tuple[str, TurnAssessment, float]]:
        async def one(row):
            label, agent, caller = row
            window = TurnWindow(turn_index=1, text=caller, recent=[("agent", agent)])
            async with self._pace:
                await asyncio.sleep(60 / self.REQUESTS_PER_MINUTE)
            loop = asyncio.get_running_loop()
            started = loop.time()
            assessment = await self._model.structured(
                TurnClassifier.INSTRUCTIONS, TurnClassifier._content(window), TurnAssessment
            )
            return label or caller, assessment, (loop.time() - started) * 1000

        return list(await asyncio.gather(*(one(row) for row in rows)))

    def _threshold(self, flag_type: FlagType, normal, flagged) -> float:
        positives = [a for label, a, _ in flagged if label == flag_type]
        print(f"  {flag_type.value:14s}", end="")
        chosen = 1.0
        for candidate in self.CANDIDATES:
            recall = sum(getattr(a, flag_type.value) >= candidate for a in positives)
            wrong = sum(getattr(a, flag_type.value) >= candidate for _, a, _ in normal)
            wrong += sum(
                getattr(a, flag_type.value) >= candidate
                for label, a, _ in flagged
                if label != flag_type and label not in (FlagType.ABUSE, FlagType.DISTRESS)
            )
            print(f" | {candidate}: {recall}/{len(positives)} fp {wrong}", end="")
            if wrong / len(normal) <= self.MAX_FALSE_POSITIVE and chosen == 1.0:
                chosen = candidate
        print(f"  → {chosen}")
        return chosen

    def _relevance(self, normal, flagged) -> None:
        for relevance in (Relevance.OFF_TOPIC, Relevance.NONSENSE):
            hits = sum(a.relevance == relevance for label, a, _ in flagged if label == relevance)
            total = sum(label == relevance for label, _, _ in flagged)
            wrong = sum(a.relevance == relevance for _, a, _ in normal)
            print(f"  {relevance.value:14s} | {hits}/{total} found, {wrong} normal turns called it")


async def main() -> None:
    LogConfig.install("WARNING")
    settings = SettingsFactory.load()
    settings.livekit.export_env()
    for name in sys.argv[1:] or [settings.detection.model]:
        await DetectionCalibrator(AnalysisModel(name, settings.detection.timeout_s * 4)).run()


if __name__ == "__main__":
    asyncio.run(main())
