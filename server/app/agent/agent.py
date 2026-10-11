from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterable, Sequence

from livekit.agents import Agent, ModelSettings
from livekit.agents.llm import ChatContext, ChatMessage, Tool, Toolset
from loguru import logger

from app.agent.prompt.compiler import PromptCompiler
from app.agent.state import CallState
from app.agent.supervisor import TurnSupervisor
from app.agent.tools.adapter import FrameworkToolAdapter
from app.agent.tools.exposure import ToolExposure
from app.core.enums import FlagType, Severity
from app.modules.detection.guard import GuardContext, OutputGuard
from app.modules.detection.schemas import FlagDraft


class ConversationAgent(Agent):
    def __init__(
        self,
        *,
        compiler: PromptCompiler,
        state: CallState,
        greeting: str,
        toolsets: Sequence[Toolset],
        adapter: FrameworkToolAdapter,
        exposure: ToolExposure,
        supervisor: TurnSupervisor,
        guard: OutputGuard,
        greeting_timeout_s: float,
    ) -> None:
        self._compiler = compiler
        self._greeting_timeout_s = greeting_timeout_s
        self._supervisor = supervisor
        self._guard = guard
        self._greeting = greeting
        self._toolsets = list(toolsets)
        self._adapter = adapter
        self._exposure = exposure
        self._exposed = self._wanted(state)
        super().__init__(instructions=state.compiled.prefix, tools=self._tools(state))

    async def on_enter(self) -> None:
        """The greeting prepared during setup: its audio if already made, else
        its words with live speech. Without either, the model writes it now."""
        # Speak the greeting when the call starts.
        greeting = self.session.userdata.greeting
        text = await self._prepared_text(greeting)
        if text is None:
            await self.session.generate_reply(instructions=self._greeting)
            return
        frames = greeting.ready_audio()
        if frames is not None:
            self.session.say(text, audio=greeting.play(frames))
        else:
            self.session.say(text)

    async def _prepared_text(self, greeting) -> str | None:
        if greeting is None:
            return None
        try:
            async with asyncio.timeout(self._greeting_timeout_s):
                return await asyncio.shield(greeting.text)
        except Exception as error:
            logger.warning("prepared greeting not used: {!r}", error)
            greeting.audio.cancel()
            return None

    async def refresh_tools(self) -> None:
        """Called after a skill load. Changes the tool set only when it must."""
        state: CallState = self.session.userdata
        wanted = self._wanted(state)
        if wanted == self._exposed:
            return
        self._exposed = wanted
        state.tool_set_changed_this_turn = True
        await self.update_tools(self._tools(state))

    def _wanted(self, state: CallState) -> list[str]:
        skill_keys = [skill.key for skill in state.skills.pinned]
        skill_keys += [key for key, _ in state.skill_history]
        return self._exposure.keys(state.config, skill_keys)

    def _tools(self, state: CallState) -> list:
        bindings = [state.config.binding(key) for key in self._exposed]
        return [*self._toolsets, *self._adapter.tools(bindings)]

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """LiveKit may already have started a preemptive reply. Changing this
        turn's context is its documented way to discard it, so a handoff decided
        here always runs instead of that reply."""
        state: CallState = self.session.userdata
        self._supervisor.on_user_turn(new_message.text_content or "")
        if state.pending_handoff:
            turn_ctx.add_message(role="system", content=f"Handoff: {state.pending_handoff}")

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncIterable:
        state: CallState = self.session.userdata
        if state.pending_handoff:
            self._supervisor.start_pending_handoff()
            return
        suffix = self._compiler.turn_suffix(state.inputs)
        state.last_prompt_ms = suffix.elapsed_ms
        state.rendered_notes = tuple(state.next_turn_notes)
        context = chat_ctx.copy()
        TurnContextInserter.insert(context, suffix.text)
        stream = Agent.default.llm_node(self, context, tools, model_settings)
        async for chunk in SentenceGuard(self._guard, state).filter(stream):
            yield chunk


class TurnContextInserter:
    @staticmethod
    def insert(context: ChatContext, text: str) -> None:
        if not text:
            return
        last_user = next(
            (
                item
                for item in reversed(context.items)
                if item.type == "message" and item.role == "user"
            ),
            None,
        )
        created_at = (last_user.created_at - 1e-6) if last_user else None
        message = ChatMessage(
            role="user",
            content=[text],
        )
        if created_at is not None:
            message.created_at = created_at
        context.insert(message)


class SentenceGuard:
    """Holds text back until a sentence is complete, checks it, then lets it
    through, replaced or removed if a rule says so. Tool calls pass untouched."""

    BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, guard: OutputGuard, state: CallState) -> None:
        self._guard = guard
        self._state = state

    async def filter(self, chunks: AsyncIterable) -> AsyncIterable:
        buffer = ""
        async for chunk in chunks:
            text = chunk if isinstance(chunk, str) else (chunk.delta and chunk.delta.content)
            if text:
                *sentences, buffer = self.BOUNDARY.split(buffer + text)
                for sentence in sentences:
                    if checked := self._check(sentence):
                        yield checked + " "
            if isinstance(chunk, str):
                continue
            if chunk.delta and chunk.delta.tool_calls and buffer.strip():
                if checked := self._check(buffer):
                    yield checked
                buffer = ""
            if not (chunk.delta and chunk.delta.content):
                yield chunk
            elif chunk.delta.tool_calls or chunk.usage:
                yield chunk.model_copy(
                    update={"delta": chunk.delta.model_copy(update={"content": None})}
                )
        if buffer.strip() and (checked := self._check(buffer)):
            yield checked

    def _check(self, sentence: str) -> str | None:
        state = self._state
        caller, organization = state.pack.caller, state.pack.organization
        context = GuardContext(
            known_text=" ".join([state.compiled.prefix, *state.tool_texts]),
            confirmed=state.confirmed,
            caller_contacts=OutputGuard.contacts(caller.email, caller.phone),
            blocked_phrases=organization.blocked_phrases,
        )
        result = self._guard.check(sentence, context)
        if result.rule is None:
            return sentence
        state.turn_guard_actions.append(result.rule)
        state.flags.append(
            FlagDraft(
                type=FlagType.GUARD,
                severity=Severity.INFO,
                score=1.0,
                evidence=f"{result.rule}: {sentence}",
                model="output_guard",
                turn_index=state.turn_index,
            ).model_dump(mode="json")
        )
        return result.text
