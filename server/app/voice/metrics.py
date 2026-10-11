from __future__ import annotations

import time

from livekit.agents import AgentSession, ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics.usage import LLMModelUsage
from livekit.agents.voice.events import AgentStateChangedEvent, SessionUsageUpdatedEvent

from app.agent.state import CallState
from app.agent.tools.implementations.knowledge import KnowledgeState
from app.core.cache import CacheStats
from app.modules.calls.recorder import TurnDraft, TurnRecorder
from app.voice.utils import milliseconds


class TurnTimer:
    def __init__(self, state: CallState, recorder: TurnRecorder, cache_stats: CacheStats) -> None:
        self._state = state
        self._recorder = recorder
        self._cache_stats = cache_stats
        self._draft: TurnDraft | None = None
        self._usage_seen = (0, 0, 0)
        self._started: float | None = None

    def attach(self, session: AgentSession):
        # Listen for conversation messages and framework metrics from LiveKit.
        session.on("conversation_item_added", self._on_item) # listenere
        session.on("session_usage_updated", self._on_usage)
        session.on("agent_state_changed", self._on_agent_state)
        self._started = time.perf_counter()

    def close(self) -> None:
        # Record a turn the call ended in the middle of.
        if self._draft is not None:
            self._finish()

    def _on_item(self, event: ConversationItemAddedEvent):
        # Route each chat message to the start or the end of a turn.
        item = event.item
        if not isinstance(item, ChatMessage):
            return
        if item.role == "user":
            self._on_caller(item)
        elif item.role == "assistant":
            self._on_agent(item)

    def _on_caller(self, message: ChatMessage) -> None:
        # A caller message starts a new turn; finish an unfinished previous turn first.

        if self._draft is not None:
            self._finish()
        self._state.turn_index += 1
        self._state.transcript.append((self._state.turn_index, "caller", message.text_content))
        metrics = message.metrics
        self._draft = TurnDraft(
            turn_index=self._state.turn_index,
            transcript_in=message.text_content,
            end_of_turn_ms=milliseconds(metrics.get("end_of_turn_delay")),
            transcription_ms=milliseconds(metrics.get("transcription_delay")),
            prefix_hash=self._state.compiled.prefix_hash,
            release_id=self._state.release_id,
        )

    def _on_agent(self, message: ChatMessage) -> None:
        # A reply does not close the turn: token usage can arrive after it, and a tool call
        # means more than one reply. The next caller message, or close(), finishes the turn.
        # With no caller message first (the greeting), the reply opens a turn of its own.
        draft = self._open()
        first_reply = draft.text_spoken is None
        spoken = filter(None, (draft.text_spoken, message.text_content))
        draft.text_spoken = " ".join(spoken) or None
        draft.interrupted = draft.interrupted or message.interrupted
        self._state.last_reply_interrupted = message.interrupted
        if message.text_content:
            self._state.transcript.append((draft.turn_index, "agent", message.text_content))
        if first_reply:
            # The first reply is what the caller waited for.
            self._spend_notes()
            metrics = message.metrics
            waited = draft.first_audio_ms
            draft.prompt_ms = self._state.last_prompt_ms
            draft.llm_ttft_ms = milliseconds(metrics.get("llm_node_ttft"))
            draft.tts_ttfb_ms = milliseconds(metrics.get("tts_node_ttfb"))
            draft.first_audio_ms = milliseconds(metrics.get("e2e_latency")) or waited

    def _on_agent_state(self, event: AgentStateChangedEvent) -> None:
        """The greeting has no caller turn to measure from, so its wait is timed
        from the session start to the first moment the agent speaks."""
        if event.new_state != "speaking" or self._started is None:
            return
        waited, self._started = milliseconds(time.perf_counter() - self._started), None
        if self._state.turn_index == 0:
            self._open().first_audio_ms = waited

    def _on_usage(self, event: SessionUsageUpdatedEvent) -> None:
        """The event carries running totals for the call; the turn gets the
        difference. A preemptive reply that was thrown away is counted: it was
        paid for."""
        llm = [u for u in event.usage.model_usage if isinstance(u, LLMModelUsage)]
        totals = (
            sum(u.input_tokens for u in llm),
            sum(u.input_cached_tokens for u in llm),
            sum(u.output_tokens for u in llm),
        )
        prompt, cached, completion = (
            now - seen for now, seen in zip(totals, self._usage_seen, strict=True)
        )
        self._usage_seen = totals
        draft = self._open()
        draft.prompt_tokens += prompt
        draft.cached_tokens += cached
        draft.completion_tokens += completion

    def _spend_notes(self) -> None:
        """A note is removed once a reply that used it is spoken. A preemptive
        reply that was thrown away used nothing."""
        state = self._state
        state.next_turn_notes = [n for n in state.next_turn_notes if n not in state.rendered_notes]
        state.rendered_notes = ()

    def _open(self) -> TurnDraft:
        # The current turn, or a new one under the current index when none is open yet.
        if self._draft is None:
            self._draft = TurnDraft(
                turn_index=self._state.turn_index,
                prefix_hash=self._state.compiled.prefix_hash,
                release_id=self._state.release_id,
            )
        return self._draft

    def _finish(self):
        draft, self._draft = self._draft, None
        draft.flags = list(self._state.flags)
        self._state.flags.clear()
        draft.relevance, self._state.turn_relevance = self._state.turn_relevance, None
        draft.frustration = round(self._state.signals.frustration, 3)
        draft.guard_actions, self._state.turn_guard_actions = self._state.turn_guard_actions, []
        draft.cache_hits = self._cache_stats.take()
        draft.optimisations = sorted(self._state.optimisations)
        draft.active_skill = self._state.active_skill
        draft.tool_set_changed, self._state.tool_set_changed_this_turn = (
            self._state.tool_set_changed_this_turn,
            False,
        )
        draft.skill_loaded, self._state.skill_loaded_this_turn = (
            self._state.skill_loaded_this_turn,
            None,
        )
        draft.retrieval = self._state.tool_state.pop(KnowledgeState.KEY, [])
        self._recorder.record(draft)
