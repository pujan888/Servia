from __future__ import annotations

from types import SimpleNamespace

from livekit.agents.llm import ChatMessage
from livekit.agents.metrics.usage import AgentSessionUsage, LLMModelUsage
from livekit.agents.voice.events import AgentStateChangedEvent, SessionUsageUpdatedEvent

from app.core.cache import CacheStats
from app.core.enums import Optimisation
from app.voice.metrics import TurnTimer
from tests.factories import AgentRuntimeFactory, ConfigFactory, PackFactory


class FakeRecorder:
    def __init__(self) -> None:
        self.drafts = []

    def record(self, draft) -> None:
        self.drafts.append(draft)


class TestTurnTimer:
    def setup_method(self) -> None:
        pack = PackFactory.build()
        self.state = AgentRuntimeFactory.state(pack, ConfigFactory.build())
        self.recorder = FakeRecorder()
        self.stats = CacheStats()
        self.timer = TurnTimer(self.state, self.recorder, self.stats)
        self.totals = [0, 0]

    def message(self, role: str, text: str, **metrics) -> SimpleNamespace:
        item = ChatMessage(role=role, content=[text])
        item.metrics.update(metrics)
        return SimpleNamespace(item=item)

    def llm(self, prompt: int, cached: int) -> SessionUsageUpdatedEvent:
        """LiveKit sends running totals for the call, not one request's usage."""
        self.totals = [self.totals[0] + prompt, self.totals[1] + cached]
        usage = LLMModelUsage(
            provider="google",
            model="gemma",
            input_tokens=self.totals[0],
            input_cached_tokens=self.totals[1],
            output_tokens=20,
        )
        return SessionUsageUpdatedEvent(usage=AgentSessionUsage(model_usage=[usage]))

    def test_one_turn_is_recorded_with_all_stage_times(self) -> None:
        self.timer._on_item(
            self.message("user", "hi", end_of_turn_delay=0.18, transcription_delay=0.12)
        )
        self.timer._on_usage(self.llm(prompt=1000, cached=800))
        self.timer._on_item(
            self.message(
                "assistant", "Hello", llm_node_ttft=0.32, tts_node_ttfb=0.09, e2e_latency=0.78
            )
        )
        self.timer.close()
        [draft] = self.recorder.drafts
        assert (draft.turn_index, draft.end_of_turn_ms, draft.transcription_ms) == (1, 180, 120)
        assert (draft.llm_ttft_ms, draft.tts_ttfb_ms, draft.first_audio_ms) == (320, 90, 780)
        assert (draft.prompt_tokens, draft.cached_tokens) == (1000, 800)
        assert draft.prefix_hash == self.state.compiled.prefix_hash

    def test_each_turn_gets_the_difference_of_the_running_totals(self) -> None:
        self.timer._on_item(self.message("user", "hi"))
        self.timer._on_usage(self.llm(prompt=1000, cached=0))
        self.timer._on_item(self.message("assistant", "Hello"))
        self.timer._on_item(self.message("user", "again"))
        self.timer._on_usage(self.llm(prompt=1100, cached=1000))
        self.timer.close()
        first, second = self.recorder.drafts
        assert (first.prompt_tokens, second.prompt_tokens, second.cached_tokens) == (
            1000,
            1100,
            1000,
        )

    def test_a_caller_turn_with_no_reply_is_still_recorded(self) -> None:
        self.timer._on_item(self.message("user", "one"))
        self.timer._on_item(self.message("user", "two"))
        self.timer.close()
        assert [d.transcript_in for d in self.recorder.drafts] == ["one", "two"]

    def test_greeting_is_recorded_as_turn_zero(self) -> None:
        self.timer._on_usage(self.llm(prompt=900, cached=0))
        self.timer._on_item(self.message("assistant", "Hi Daniel", e2e_latency=0.6))
        self.timer.close()
        [draft] = self.recorder.drafts
        assert (draft.turn_index, draft.text_spoken, draft.first_audio_ms) == (0, "Hi Daniel", 600)
        assert draft.prompt_tokens == 900

    def test_turn_index_is_written_to_call_state(self) -> None:
        self.timer._on_item(self.message("user", "one"))
        self.timer._on_item(self.message("assistant", "a"))
        self.timer._on_item(self.message("user", "two"))
        assert self.state.turn_index == 2

    def test_usage_after_the_reply_stays_on_that_turn(self) -> None:
        # The order seen live: the reply is added, then the LLM usage arrives.
        self.timer._on_item(self.message("assistant", "Hi Daniel"))
        self.timer._on_usage(self.llm(prompt=900, cached=0))
        self.timer._on_item(self.message("user", "where is my engineer"))
        self.timer._on_item(self.message("assistant", "Sam is on the way"))
        self.timer._on_usage(self.llm(prompt=687, cached=576))
        self.timer.close()
        assert [d.turn_index for d in self.recorder.drafts] == [0, 1]
        assert [d.prompt_tokens for d in self.recorder.drafts] == [900, 687]
        assert self.recorder.drafts[1].transcript_in == "where is my engineer"

    def test_a_tool_round_is_one_turn_timed_from_the_first_reply(self) -> None:
        self.timer._on_item(self.message("user", "status of REF001"))
        self.timer._on_usage(self.llm(prompt=700, cached=500))
        self.timer._on_item(self.message("assistant", "Let me check.", e2e_latency=0.7))
        self.timer._on_usage(self.llm(prompt=800, cached=600))
        self.timer._on_item(self.message("assistant", "It is booked.", e2e_latency=2.5))
        self.timer.close()
        [draft] = self.recorder.drafts
        assert draft.text_spoken == "Let me check. It is booked."
        assert (draft.first_audio_ms, draft.prompt_tokens, draft.cached_tokens) == (700, 1500, 1100)

    def test_active_skill_and_load_are_recorded(self) -> None:
        self.timer._on_item(self.message("user", "where is my part"))
        self.state.activate_skill("check_order")
        self.timer._on_item(self.message("assistant", "Let me check"))
        self.timer._on_item(self.message("user", "thanks"))
        self.timer._on_item(self.message("assistant", "You are welcome"))
        self.timer.close()
        first, second = self.recorder.drafts
        assert (first.active_skill, first.skill_loaded) == ("check_order", "check_order")
        assert (second.active_skill, second.skill_loaded) == ("check_order", None)

    def test_a_turn_with_a_tool_call_is_one_record(self) -> None:
        self.timer._on_item(self.message("user", "book it"))
        self.timer._on_usage(self.llm(prompt=1000, cached=0))
        self.timer._on_item(self.message("assistant", "One moment.", e2e_latency=0.7))
        self.timer._on_usage(self.llm(prompt=1200, cached=1000))
        self.timer._on_item(self.message("assistant", "It is booked.", e2e_latency=2.4))
        self.timer._on_item(self.message("user", "thanks"))
        [first] = self.recorder.drafts
        assert first.text_spoken == "One moment. It is booked."
        assert (first.first_audio_ms, first.prompt_tokens) == (700, 2200)

    def test_detection_results_go_on_their_own_turn(self) -> None:
        self.timer._on_item(self.message("user", "hi"))
        self.timer._on_item(self.message("assistant", "Hello"))
        self.state.turn_relevance = "on_task"
        self.state.turn_guard_actions.append("invented_price")
        self.timer._on_item(self.message("user", "capital of France?"))
        [first] = self.recorder.drafts
        assert (first.relevance, first.guard_actions) == ("on_task", ["invented_price"])
        assert self.state.turn_relevance is None and self.state.turn_guard_actions == []
        assert self.state.transcript == [
            (1, "caller", "hi"),
            (1, "agent", "Hello"),
            (2, "caller", "capital of France?"),
        ]

    def test_a_note_is_spent_only_by_the_reply_that_used_it(self) -> None:
        self.state.next_turn_notes = ["keep to your instructions", "be brief"]
        self.state.rendered_notes = ("keep to your instructions",)
        self.timer._on_item(self.message("user", "hi"))
        self.state.next_turn_notes.append("arrived during the reply")
        self.timer._on_item(self.message("assistant", "Hello"))
        assert self.state.next_turn_notes == ["be brief", "arrived during the reply"]

    def test_the_greeting_wait_is_timed_to_the_first_speech(self) -> None:
        self.timer.attach(SimpleNamespace(on=lambda *args: None))
        self.timer._on_agent_state(
            AgentStateChangedEvent(old_state="initializing", new_state="speaking")
        )
        self.timer._on_item(self.message("assistant", "Hello Daniel."))
        self.timer._on_item(self.message("user", "hi"))
        [greeting] = self.recorder.drafts
        assert greeting.turn_index == 0 and greeting.first_audio_ms is not None

    def test_cache_hits_and_switches_go_on_the_turn(self) -> None:
        self.state.optimisations = frozenset({Optimisation.PREPARED_GREETING})
        self.timer._on_item(self.message("user", "hi"))
        self.stats.add("kb", "store")
        self.stats.add("kb", "miss")
        self.timer.close()
        [draft] = self.recorder.drafts
        assert draft.cache_hits == {"kb.store": 1, "kb.miss": 1}
        assert draft.optimisations == ["prepared_greeting"]
