from __future__ import annotations

import time
import uuid

from livekit.agents import AgentSession, inference

from app.agent.runtime import AgentRuntime
from app.agent.state import CallState
from app.core.container import Container
from app.core.enums import CallChannel
from app.eval.assertions import Observation, TurnSeen
from app.eval.mocks import MockToolLayer, ReplayGateway
from app.modules.calls.models import Call
from app.modules.calls.recorder import TurnDraft
from app.modules.calls.utils import RoomName
from app.modules.config.schemas import AgentConfig
from app.modules.context.schemas import ContextPack
from app.modules.evaluation.models import Scenario
from app.voice.metrics import TurnTimer


class Collector:
    """Stands in for the database writers: a replay stores nothing it does not
    return."""

    def __init__(self) -> None:
        self.items: list = []

    def record(self, item) -> None:
        self.items.append(item)


class ReplayRunner:
    """Runs a scenario as text, with the release under test. The compiler,
    skills, gateway, knowledge and detection are the ones a real call uses;
    only the transport and the tool execution differ. The model runs at its
    lowest temperature."""

    def __init__(self, container: Container, runtime: AgentRuntime) -> None:
        self._container = container
        self._runtime = runtime

    async def run(
        self, scenario: Scenario, config: AgentConfig, release_id: uuid.UUID
    ) -> Observation:
        call_id = await self._replay_call(scenario, release_id)
        pack = ContextPack.model_validate({**scenario.pack, "call_id": str(call_id)})
        state = self._runtime.state(pack, config)
        state.release_id = release_id
        mocks = MockToolLayer(scenario.responses)
        gateway = ReplayGateway(self._container.registry, Collector(), self._container.cache, mocks)
        supervisor = self._runtime.supervisor(state, gateway, self._container.database)
        turns = Collector()
        timer = TurnTimer(state, turns, self._container.cache.stats)
        model = config.profile.llm_model or self._container.settings.speech.llm_model
        llm = inference.LLM(model=model, extra_kwargs={"temperature": 0})
        timings: list[int] = []
        async with AgentSession[CallState](userdata=state, llm=llm) as session:
            timer.attach(session)
            supervisor.attach(session)
            await session.start(self._runtime.agent(state, gateway, supervisor))
            if session.current_speech is not None:
                await session.current_speech
            for text in scenario.caller_turns:
                started = time.perf_counter()
                supervisor.on_user_turn(text)
                await session.run(user_input=text)
                await supervisor.drain()
                while session.current_speech is not None:
                    await session.current_speech
                timings.append(round((time.perf_counter() - started) * 1000))
        await supervisor.drain()
        timer.close()
        return self._observe(state, turns.items, timings, gateway, mocks)

    async def _replay_call(self, scenario: Scenario, release_id: uuid.UUID) -> uuid.UUID:
        """A handoff records itself against a call, so a replay has a call row
        of its own, marked as a replay and left out of every measurement."""
        async with self._container.database.session() as session:
            call = Call(
                organization_id=scenario.organization_id,
                user_id=uuid.UUID(scenario.pack["caller"]["id"]),
                room_name=RoomName.generate(),
                channel=CallChannel.REPLAY,
                release_id=release_id,
            )
            session.add(call)
            await session.flush()
            return call.id

    @staticmethod
    def _observe(
        state: CallState,
        drafts: list[TurnDraft],
        timings: list[int],
        gateway: ReplayGateway,
        mocks: MockToolLayer,
    ) -> Observation:
        caller_turns = [d for d in drafts if d.turn_index > 0]
        turns = [
            TurnSeen(
                index=draft.turn_index,
                caller=draft.transcript_in or "",
                agent=draft.text_spoken or "",
                skill=draft.active_skill,
                ms=timings[position] if position < len(timings) else 0,
                unanswered=draft.turn_index in state.signals.unanswered_turns,
            )
            for position, draft in enumerate(caller_turns)
        ]
        passages = {call.turn: call.text for call in gateway.calls if call.knowledge}
        return Observation(
            turns=turns,
            tools=gateway.calls,
            escalation=state.signals.escalation_reason,
            unexpected=mocks.unexpected,
            known_text=" ".join([state.compiled.prefix, *state.tool_texts]),
            passages=passages,
        )
