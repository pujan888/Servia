"""Talk to the real agent in text, with no audio and no LiveKit room.

    uv run python -m app.text_call <user_id> "first message" "second message" ...

It uses the real model, the real prompt, skills, gateway and database. Turns
are recorded and the call is ended as a voice call is, so the post-call
pipeline runs on it too. Text input skips the framework's end-of-turn hook,
so the supervisor is called here, where the voice agent's hook would call it."""

from __future__ import annotations

import asyncio
import sys
import uuid

from livekit.agents import AgentSession, inference
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)

from app.agent.runtime import AgentRuntime
from app.agent.setup import CallSetup
from app.agent.state import CallState
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.outbox import OutboxService
from app.core.container import Container
from app.core.logging import LogConfig
from app.modules.accounts.service import UserService
from app.modules.calls.journal import ToolJournalWriter
from app.modules.calls.recorder import TurnRecorder
from app.modules.calls.service import CallService
from app.voice.fillers import FillerLibrary
from app.voice.metrics import TurnTimer


class TextCall:
    def __init__(self, container: Container) -> None:
        self._container = container
        self._runtime = AgentRuntime(container.settings, FillerLibrary())

    async def run(self, user_id: uuid.UUID, messages: list[str]) -> None:
        database = self._container.database
        state = await self._prepare(user_id)
        journal = ToolJournalWriter(database)
        recorder = TurnRecorder(database, state.organization_id, state.call_id)
        gateway = ToolGateway(
            self._container.registry, journal, OutboxService(database), self._container.cache
        )
        llm = inference.LLM(
            model=state.config.profile.llm_model or self._container.settings.speech.llm_model
        )
        timer = TurnTimer(state, recorder, self._container.cache.stats)
        supervisor = self._runtime.supervisor(state, gateway, database)
        async with AgentSession[CallState](userdata=state, llm=llm) as session:
            timer.attach(session)
            supervisor.attach(session)
            await session.start(self._runtime.agent(state, gateway, supervisor))
            if session.current_speech is not None:
                await session.current_speech
            print(f"agent  > {session.history.items[-1].text_content}")
            for message in messages:
                print(f"\ncaller > {message}")
                supervisor.on_user_turn(message)
                result = await session.run(user_input=message)
                self._show(result.events, state)
                await self._settle(session, supervisor, state, len(session.history.items))
        await supervisor.drain()
        timer.close()
        await asyncio.gather(recorder.aclose(), journal.aclose())
        async with database.session() as session:
            await CallService(session).mark_ended(state.call_id)
        print(f"\ncall {state.call_id} ended; the post-call job is queued")

    async def _prepare(self, user_id: uuid.UUID) -> CallState:
        async with self._container.database.session() as session:
            user = await UserService(session).get(user_id)
            call = await CallService(session).start(user, None)
        state = await CallSetup(self._container, self._runtime).prepare(call.id)
        if state.pack.degraded:
            print(f"  [degraded: {state.pack.degraded}]")
        return state

    @staticmethod
    async def _settle(session: AgentSession, supervisor, state: CallState, shown: int) -> None:
        """Waits for this turn's detection and any fixed speech it caused, then
        prints what detection added, so the next message sees its effects."""
        await supervisor.drain()
        while session.current_speech is not None:
            await session.current_speech
        for item in session.history.items[shown:]:
            if item.type == "message" and item.role == "assistant":
                print(f"agent  > {item.text_content}  (fixed text)")
        flags = [f"{f['type']} {f['score']}" for f in state.flags if f["type"] != "guard"]
        print(
            f"  [detection: relevance={state.turn_relevance} frustration="
            f"{state.signals.frustration:.2f} flags={flags} guard={state.turn_guard_actions} "
            f"notes={state.next_turn_notes} escalation={state.signals.escalation}]"
        )

    @staticmethod
    def _show(events, state: CallState) -> None:
        for event in events:
            if isinstance(event, FunctionCallEvent):
                print(f"  tool  > {event.item.name}({event.item.arguments})")
            elif isinstance(event, FunctionCallOutputEvent):
                print(f"  result< {event.item.output[:160]}")
            elif isinstance(event, ChatMessageEvent) and event.item.role == "assistant":
                print(f"agent  > {event.item.text_content}")
        print(f"  [active skill: {state.active_skill}]")


async def main() -> None:
    LogConfig.install("WARNING")
    container = Container()
    container.settings.livekit.export_env()
    container.embedder.load()
    try:
        await TextCall(container).run(uuid.UUID(sys.argv[1]), sys.argv[2:])
    finally:
        await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
