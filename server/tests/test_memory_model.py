"""These call the real language model, to check the prompts, not only the code."""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.tools.registry import ToolRegistry
from app.analysis.model import AnalysisModel
from app.analysis.outcomes.classifier import OutcomeClassifier
from app.analysis.postcall.slots import SlotExtractor
from app.analysis.postcall.summariser import Summariser
from app.analysis.postcall.utils import Sentences, TranscriptText
from app.core.config import SettingsFactory
from app.core.enums import CallOutcome, SlotSource, SlotValueType, Speaker
from app.modules.calls.models import TranscriptLine, TurnRecord
from app.modules.config.schemas import SlotKeySpec

pytestmark = pytest.mark.model

KEYS = [
    SlotKeySpec(
        key="preferred_visit_time",
        description="The part of the day the caller prefers for visits, as a general rule.",
        value_type=SlotValueType.TEXT,
        example="mornings",
        spoken=True,
    )
]


@pytest.fixture(scope="module")
def model() -> AnalysisModel:
    settings = SettingsFactory.load()
    settings.livekit.export_env()
    return AnalysisModel(settings.speech.llm_model, settings.memory.llm_timeout_s)


def lines(*said: tuple[Speaker, str]) -> list[TranscriptLine]:
    return [
        TranscriptLine(sequence=i, turn_index=i, speaker=speaker, text=text)
        for i, (speaker, text) in enumerate(said, 1)
    ]


async def test_a_hedged_preference_is_not_stored_as_stated(model):
    transcript = lines(
        (Speaker.AGENT, "Which day suits you?"),
        (Speaker.CALLER, "I am usually free in the morning, but this week is difficult."),
    )
    candidates = await SlotExtractor(model, 0.6).extract(transcript, KEYS)
    assert all(candidate.source == SlotSource.INFERRED for candidate in candidates)


async def test_the_summary_keeps_every_reference(model):
    transcript = lines(
        (Speaker.CALLER, "My fridge is warm and the oven fan is loud."),
        (Speaker.SYSTEM, "request_callback accepted, reference RT4VWX: topic=fridge"),
        (Speaker.SYSTEM, "add_work_note accepted, reference MC7KHP: note=oven fan loud"),
        (Speaker.AGENT, "Your callback is RT4VWX and the note is on job MC7KHP."),
    )
    text = await Summariser(model, 3).summarise(
        TranscriptText.render(transcript), date(2026, 9, 26), [], ["RT4VWX", "MC7KHP"]
    )
    assert "RT4VWX" in text and "MC7KHP" in text
    assert len(Sentences.split(text.split(" References:")[0])) <= 3


async def test_an_answered_question_is_resolved_by_the_model(model):
    transcript = lines(
        (Speaker.CALLER, "What time do you open on Saturday?"),
        (Speaker.AGENT, "We are open from nine until one on Saturdays."),
        (Speaker.CALLER, "Great, that is all I needed. Bye."),
        (Speaker.AGENT, "Goodbye."),
    )
    turns = [TurnRecord(turn_index=1, transcript_in="hours?", text_spoken="Nine to one.")]
    outcome, basis = await OutcomeClassifier(model, ToolRegistry()).classify(
        turns, [], TranscriptText.render(transcript)
    )
    assert outcome == CallOutcome.RESOLVED and basis.startswith("model")
