from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from app.agent.prompt.budget import ContextTruncation
from app.agent.prompt.greeting import Greeting
from app.agent.prompt.sections import ContextSection
from app.agent.tools.implementations.work import AddWorkNote, RequestCallback
from app.agent.tools.registry import ToolRegistry
from app.analysis.outcomes.classifier import OutcomeClassifier
from app.analysis.postcall.digest import DigestBuilder
from app.analysis.postcall.slots import ExtractedSlot, SlotExtractor
from app.analysis.postcall.summariser import Summariser
from app.analysis.postcall.transcript import TranscriptBuilder
from app.analysis.postcall.utils import DayResolver, Sentences
from app.core.config import MemorySettings
from app.core.enums import CallOutcome, SlotSource, SlotValueType, Speaker
from app.core.tokens import TokenCounter
from app.modules.calls.models import Call, ToolJournalEntry, TranscriptLine, TurnRecord
from app.modules.calls.recorder import TurnDraft, TurnRecorder
from app.modules.calls.utils import Redactor
from app.modules.config.schemas import SlotKeySpec
from app.modules.memory.models import MemorySlot
from app.modules.memory.schemas import Digest, DigestSlot, SlotDecision, SummaryRead
from app.modules.memory.utils import SlotPrecedence, SlotValues
from app.modules.work.models import ActionOutbox
from tests.factories import PackFactory
from tests.fakes import FakeModel

CALL = Call(id=uuid.UUID(int=10), organization_id=uuid.UUID(int=2), user_id=uuid.UUID(int=3))


def slot_key(key: str, spoken: bool = True, value_type=SlotValueType.TEXT) -> SlotKeySpec:
    return SlotKeySpec(
        key=key, description=f"The {key}.", value_type=value_type, example="x", spoken=spoken
    )


def line(sequence: int, speaker: Speaker, text: str, turn: int = 1) -> TranscriptLine:
    return TranscriptLine(sequence=sequence, turn_index=turn, speaker=speaker, text=text)


def turn(index: int, heard: str | None, said: str | None = None, **values) -> TurnRecord:
    return TurnRecord(turn_index=index, transcript_in=heard, text_spoken=said, **values)


class TestRedactor:
    def test_a_card_number_is_masked(self) -> None:
        text = Redactor.redact("my card is 4111 1111 1111 1111 thanks")
        assert text == "my card is [card number] thanks"

    def test_a_number_that_fails_the_card_check_is_not_called_a_card(self) -> None:
        assert Redactor.redact("order 4111 1111 1111 1112") == "order 4111 1111 1111 1112"

    def test_a_bank_account_and_a_long_number_are_masked(self) -> None:
        assert Redactor.redact("IBAN GB29 NWBK 6016 1331 9268 19") == "IBAN [bank account]"
        assert Redactor.redact("account 123456789") == "account [number]"

    def test_a_phone_number_is_kept(self) -> None:
        assert Redactor.redact("call 07700 900123") == "call 07700 900123"

    def test_the_turn_recorder_never_stores_the_raw_number(self) -> None:
        recorder = TurnRecorder(database=None, organization_id=uuid.uuid4(), call_id=uuid.uuid4())
        row = recorder.to_row(TurnDraft(turn_index=1, transcript_in="4111111111111111"))
        assert row.transcript_in == "[card number]"


class TestSlotPrecedence:
    """The plan's table, one row per case."""

    @pytest.mark.parametrize(
        ("current", "new", "expected"),
        [
            (None, ("stated", "morning"), SlotDecision.INSERT),
            (("stated", "morning"), ("stated", "morning"), SlotDecision.CONFIRM),
            (("stated", "morning"), ("stated", "evening"), SlotDecision.REPLACE),
            (("stated", "morning"), ("inferred", "evening"), SlotDecision.KEEP),
            (("inferred", "morning"), ("system", "evening"), SlotDecision.REPLACE),
        ],
    )
    def test_the_plan_table(self, current, new, expected) -> None:
        assert SlotPrecedence.decide(current, *new) == expected

    def test_equal_values_ignore_case(self) -> None:
        assert SlotPrecedence.decide(("stated", "Mornings"), "inferred", "mornings") == (
            SlotDecision.CONFIRM
        )

    def test_a_confirmation_keeps_the_higher_source(self) -> None:
        assert SlotPrecedence.higher("inferred", "stated") == SlotSource.STATED


class TestSlotValues:
    @pytest.mark.parametrize(
        ("value", "value_type", "expected"),
        [
            ("  side   gate ", SlotValueType.TEXT, "side gate"),
            ("2.0", SlotValueType.NUMBER, "2"),
            ("two", SlotValueType.NUMBER, None),
            ("Yes", SlotValueType.BOOLEAN, "yes"),
            ("2026-10-01", SlotValueType.DATE, "2026-10-01"),
            ("Thursday", SlotValueType.DATE, None),
        ],
    )
    def test_values_are_put_in_their_type(self, value, value_type, expected) -> None:
        assert SlotValues.normalise(value, value_type) == expected


class TestSlotExtractor:
    LINES = [
        line(1, Speaker.CALLER, "I am usually free in the morning, but this week is difficult."),
        line(2, Speaker.AGENT, "Afternoons then?"),
        line(3, Speaker.CALLER, "Never book afternoons, I work then.", turn=2),
    ]
    KEYS = [slot_key("preferred_visit_time"), slot_key("times_to_avoid")]

    def extracted(self, **values) -> ExtractedSlot:
        slot = {"key": "times_to_avoid", "value": "afternoons", "source": "stated"}
        return ExtractedSlot(**{**slot, "confidence": 0.9, "line": 3, **values})

    def accept(self, *slots) -> list:
        return SlotExtractor(FakeModel(), min_confidence=0.6).accept(slots, self.LINES, self.KEYS)

    def test_a_plain_statement_is_stated_with_its_turn(self) -> None:
        [candidate] = self.accept(self.extracted())
        assert (candidate.source, candidate.evidence_turn) == (SlotSource.STATED, 2)

    def test_a_hedged_line_is_never_stated(self) -> None:
        [candidate] = self.accept(
            self.extracted(key="preferred_visit_time", value="mornings", line=1)
        )
        assert candidate.source == SlotSource.INFERRED

    def test_an_undeclared_key_is_not_stored(self) -> None:
        assert self.accept(self.extracted(key="favourite_colour")) == []

    def test_a_value_with_confidence_under_the_minimum_is_not_stored(self) -> None:
        assert self.accept(self.extracted(confidence=0.5)) == []

    def test_evidence_must_be_something_the_caller_said(self) -> None:
        assert self.accept(self.extracted(line=2)) == []
        assert self.accept(self.extracted(line=99)) == []

    async def test_no_declared_keys_means_no_model_call(self) -> None:
        model = FakeModel()
        assert await SlotExtractor(model, 0.6).extract(self.LINES, []) == []
        assert model.prompts == []


class TestSummariser:
    DAY = date(2026, 9, 26)

    async def summarise(self, *answers, references=("KQ7M3A",)) -> tuple[str, FakeModel]:
        model = FakeModel(texts=answers)
        text = await Summariser(model, 3).summarise("[1] caller: hi", self.DAY, [], references)
        return text, model

    async def test_a_lost_reference_is_asked_for_again_by_name(self) -> None:
        text, model = await self.summarise("A visit was booked.", "Visit booked, KQ7M3A.")
        assert text == "Visit booked, KQ7M3A."
        assert "KQ7M3A" in model.prompts[1] and "left out" in model.prompts[1]

    async def test_a_reference_lost_twice_is_added_by_code(self) -> None:
        text, _ = await self.summarise("A visit was booked.", "A visit was booked.")
        assert text == "A visit was booked. References: KQ7M3A."

    async def test_the_summary_has_at_most_three_sentences(self) -> None:
        text, _ = await self.summarise("One KQ7M3A. Two. Three. Four.")
        assert Sentences.split(text) == ["One KQ7M3A.", "Two.", "Three."]

    async def test_days_are_dated_by_code(self) -> None:
        text, _ = await self.summarise(
            "Visit on Thursday 2 October, KQ7M3A.", references=("KQ7M3A",)
        )
        assert text == "Visit on Thursday 1 October, KQ7M3A."


class TestDayResolver:
    SATURDAY = date(2026, 9, 26)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("on Thursday morning", "on Thursday 1 October morning"),
            ("on Thursday 2 October 2026", "on Thursday 1 October"),
            ("by Monday 28 September", "by Monday 28 September"),
            ("on Thursday 8 October", "on Thursday 8 October"),
            ("prefers Thursdays", "prefers Thursdays"),
            ("every Thursday", "every Thursday"),
            ("on Saturday", "on Saturday 26 September"),
            ("on Thursday October first in the morning", "on Thursday 1 October in the morning"),
            ("on Thursday, the first of October", "on Thursday 1 October"),
            ("on Thursday 1st October, 2026", "on Thursday 1 October"),
            ("on Thursday the eighth of October", "on Thursday 8 October"),
        ],
    )
    def test_the_weekday_decides_the_date(self, text, expected) -> None:
        assert DayResolver.resolve(text, self.SATURDAY) == expected


class TestDigestBuilder:
    def slot(self, key: str, value: str, source: str, confidence: float = 0.9) -> MemorySlot:
        return MemorySlot(key=key, value=value, source=source, confidence=confidence)

    def builder(self, **settings) -> DigestBuilder:
        return DigestBuilder(MemorySettings(**settings), TokenCounter())

    def test_only_spoken_slots_are_used_most_certain_first(self) -> None:
        slots = [
            self.slot("preferred_visit_time", "mornings", "inferred"),
            self.slot("times_to_avoid", "afternoons", "stated"),
            self.slot("contact_channel", "text message", "stated"),
        ]
        keys = [
            slot_key("preferred_visit_time"),
            slot_key("times_to_avoid"),
            slot_key("contact_channel", spoken=False),
        ]
        digest = self.builder().build(None, slots, keys)
        assert [s.value for s in digest.slots] == ["afternoons", "mornings"]

    def test_the_digest_fits_its_budget(self) -> None:
        summary = SummaryRead(
            call_id=uuid.uuid4(),
            text="The caller booked a visit. " * 3 + "A colleague will call back.",
            call_at=None,
        )
        slots = [self.slot("times_to_avoid", "afternoons " * 10, "stated")]
        builder = self.builder(digest_tokens=40)
        digest = builder.build(summary, slots, [slot_key("times_to_avoid")])
        assert builder.tokens(digest) <= 40
        assert digest.slots == () and digest.summary.startswith("The caller booked a visit.")


class TestTranscriptBuilder:
    def test_the_agent_line_is_what_the_caller_heard(self) -> None:
        turns = [
            turn(0, None, "Hello Sam."),
            turn(1, "Where is my part?", "It ships", interrupted=True),
        ]
        lines = TranscriptBuilder().build(CALL, turns, [], [])
        assert [(item.speaker, item.text) for item in lines] == [
            (Speaker.AGENT, "Hello Sam."),
            (Speaker.CALLER, "Where is my part?"),
            (Speaker.AGENT, "It ships"),
        ]

    def test_a_write_is_a_system_line_with_its_reference_between_the_turns_lines(self) -> None:
        entry = ToolJournalEntry(
            turn_index=1,
            tool_key="request_callback",
            decision="permitted",
            status="accepted",
            arguments={"preferred_time": "Thursday morning"},
            idempotency_key="key-1",
        )
        receipt = ActionOutbox(idempotency_key="key-1", reference="KQ7M3A")
        lines = TranscriptBuilder().build(CALL, [turn(1, "Book it", "Done")], [entry], [receipt])
        assert [item.speaker for item in lines] == [Speaker.CALLER, Speaker.SYSTEM, Speaker.AGENT]
        assert lines[1].text == (
            "request_callback accepted, reference KQ7M3A: preferred_time=Thursday morning"
        )
        assert [item.sequence for item in lines] == [1, 2, 3]


class TestOutcomeRules:
    def classifier(self) -> OutcomeClassifier:
        registry = ToolRegistry()
        registry.register(RequestCallback(factory=None))
        registry.register(AddWorkNote(database=None))
        return OutcomeClassifier(FakeModel(), registry)

    def test_a_callback_receipt_is_a_promised_callback(self) -> None:
        receipts = [
            ActionOutbox(implementation="add_work_note"),
            ActionOutbox(implementation="request_callback"),
        ]
        outcome = self.classifier().by_rules([turn(1, "hi", "hello")], receipts)
        assert outcome == CallOutcome.CALLBACK_PROMISED

    def test_a_note_receipt_is_resolved(self) -> None:
        receipts = [ActionOutbox(implementation="add_work_note")]
        assert self.classifier().by_rules([turn(1, "hi", "ok")], receipts) == CallOutcome.RESOLVED

    def test_a_call_where_the_caller_never_spoke_is_abandoned(self) -> None:
        assert self.classifier().by_rules([turn(0, None, "Hello")], []) == CallOutcome.ABANDONED

    def test_a_caller_who_left_before_the_reply_abandoned_the_call(self) -> None:
        turns = [turn(1, "hi", "hello"), turn(2, "check my order", None)]
        assert self.classifier().by_rules(turns, []) == CallOutcome.ABANDONED

    async def test_the_model_decides_only_when_no_rule_applies(self) -> None:
        classifier = self.classifier()
        classifier._model = FakeModel(structured=[{"outcome": "resolved", "reason": "answered"}])
        outcome, basis = await classifier.classify([turn(1, "hours?", "Nine to five.")], [], "")
        assert (outcome, basis) == (CallOutcome.RESOLVED, "model: answered")


class TestMemoryInThePrompt:
    def pack(self):
        digest = Digest(
            summary="Daniel booked a visit for Thursday 1 October, reference KQ7M3A.",
            call_at=datetime.now(UTC) - timedelta(days=1),
            slots=(DigestSlot(label="Times to avoid", value="afternoons"),),
        )
        return PackFactory.build().model_copy(update={"digest": digest})

    def test_the_context_section_phrases_when_the_last_call_was(self) -> None:
        text = ContextSection(TokenCounter())._text(self.pack())
        assert "From earlier calls:\n- Last call, yesterday: Daniel booked" in text
        assert "- What they told us before: Times to avoid: afternoons." in text

    def test_the_greeting_mentions_what_the_last_call_left_open(self) -> None:
        assert "left something open" in Greeting(self.pack()).instruction()
        assert "left something open" not in Greeting(PackFactory.build()).instruction()

    def test_memory_is_dropped_before_services_when_the_context_is_too_long(self) -> None:
        truncation = ContextTruncation(self.pack())
        while truncation.step():
            pass
        assert truncation.pack.digest is None
        assert truncation.applied.index("digest") < truncation.applied.index("services_to_count")
