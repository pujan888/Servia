from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import tts

from app.agent.agent import SentenceGuard
from app.agent.state import CallState
from app.analysis.model import AnalysisModel
from app.modules.detection.guard import OutputGuard


@dataclass(slots=True)
class PreparedGreeting:
    """The greeting's words, then its audio, both started during setup."""

    text: asyncio.Task[str]
    audio: asyncio.Task[list[rtc.AudioFrame]]

    def ready_audio(self) -> list[rtc.AudioFrame] | None:
        """The audio, only if it is already made. Otherwise it is dropped and
        the words are spoken with live speech, which starts sooner."""
        if self.audio.done() and not self.audio.cancelled() and self.audio.exception() is None:
            return self.audio.result()
        self.audio.cancel()
        return None

    @staticmethod
    async def play(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        for frame in frames:
            yield frame


class GreetingPreparer:
    """Writes the greeting while the call is being set up, then synthesises it.
    Measured: the words take about 1.9 s and the full audio another 2.8 s. That
    helps only where setup has time to spare, such as a phone call that is
    still ringing; on a web call it was slower than a live greeting. The
    greeting names the caller, so it is made for each call, and it passes the
    output guard like every other reply."""

    def __init__(self, guard: OutputGuard) -> None:
        self._guard = guard

    def start(
        self, state: CallState, synthesiser: tts.TTS, model: AnalysisModel, instruction: str
    ) -> PreparedGreeting:
        text = asyncio.create_task(self._write(state, model, instruction))
        audio = asyncio.create_task(self._synthesise(text, synthesiser))
        return PreparedGreeting(text, audio)

    async def _write(self, state: CallState, model: AnalysisModel, instruction: str) -> str:
        written = await model.text(state.compiled.prefix, instruction)
        guarded = SentenceGuard(self._guard, state).filter(self._one(written))
        return "".join([piece async for piece in guarded]).strip()

    @staticmethod
    async def _synthesise(text: asyncio.Task[str], synthesiser: tts.TTS) -> list[rtc.AudioFrame]:
        words = await text
        return [event.frame async for event in synthesiser.synthesize(words)]

    @staticmethod
    async def _one(text: str) -> AsyncIterator[str]:
        yield text
