from __future__ import annotations

from collections.abc import AsyncIterator

from livekit import rtc
from livekit.agents import AgentSession, tts
from loguru import logger


class FillerLibrary:
    """Short phrases for slow tools, synthesised in advance so they play at once.
    One copy per voice per worker process."""

    PHRASES = {
        "lookup": "Let me check that for you.",
        "write": "I am putting that through now.",
        "default": "One moment.",
    }

    def __init__(self) -> None:
        self._audio: dict[tuple[str, str], list[rtc.AudioFrame]] = {}

    def phrase(self, category: str | None) -> str:
        return self.PHRASES.get(category or "default", self.PHRASES["default"])

    def frames(self, voice: str, category: str | None) -> list[rtc.AudioFrame] | None:
        return self._audio.get((voice, category or "default"))

    async def warm(self, engine: tts.TTS, voice: str) -> None:
        for category, text in self.PHRASES.items():
            if (voice, category) in self._audio:
                continue
            try:
                frames = [chunk.frame async for chunk in engine.synthesize(text)]
            except Exception:
                logger.warning(
                    "filler '{}' could not be synthesised; it will be spoken live", category
                )
                continue
            self._audio[(voice, category)] = frames


class FillerSource:
    """The callable that RunContext.with_filler() invokes when a tool is slow."""

    def __init__(
        self, session: AgentSession, library: FillerLibrary, voice: str, category: str | None
    ):
        self._session = session
        self._library = library
        self._voice = voice
        self._category = category

    def __call__(self, step: int):
        text = self._library.phrase(self._category)
        frames = self._library.frames(self._voice, self._category)
        if frames is None:
            return text
        return self._session.say(
            text, audio=self._stream(frames), allow_interruptions=True, add_to_chat_ctx=False
        )

    @staticmethod
    async def _stream(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        for frame in frames:
            yield frame
