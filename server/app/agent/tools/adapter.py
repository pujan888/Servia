from __future__ import annotations

import asyncio
from typing import Any

from livekit.agents import RunContext
from livekit.agents.llm import RawFunctionTool, StopResponse, function_tool

from app.agent.supervisor import TurnSupervisor
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.implementations.call import EndCall, EscalateToHuman
from app.core.enums import LatencyClass
from app.modules.config.schemas import BindingSpec
from app.voice.fillers import FillerLibrary, FillerSource


class FrameworkToolAdapter:
    """Turns bindings into framework tools. Every tool has the same handler, and
    the handler only calls the gateway."""

    FILLER_DELAY_S = 0.3

    def __init__(
        self,
        gateway: ToolGateway,
        fillers: FillerLibrary,
        voice: str,
        supervisor: TurnSupervisor,
    ) -> None:
        self._gateway = gateway
        self._fillers = fillers
        self._voice = voice
        self._supervisor = supervisor

    def tools(self, bindings: list[BindingSpec]) -> list[RawFunctionTool]:
        return [self._tool(binding) for binding in bindings]

    def _tool(self, binding: BindingSpec) -> RawFunctionTool:
        async def handler(context: RunContext, raw_arguments: dict[str, Any]) -> str:
            return await self._handle(binding, context, raw_arguments)

        return function_tool(
            handler,
            raw_schema={
                "name": binding.key,
                "description": binding.description,
                "parameters": binding.parameters,
            },
        )

    async def _handle(
        self, binding: BindingSpec, context: RunContext, arguments: dict[str, Any]
    ) -> str:
        state = context.userdata
        if binding.latency_class == LatencyClass.B:
            source = FillerSource(
                context.session, self._fillers, self._voice, binding.filler_category
            )
            async with context.with_filler(source, delay=self.FILLER_DELAY_S):
                result = await self._gateway.invoke(state, binding.key, arguments)
        else:
            result = await self._gateway.invoke(state, binding.key, arguments)

        if result.control == EndCall.CONTROL:
            asyncio.create_task(self._end_call(context))
        elif result.control == EscalateToHuman.CONTROL:
            self._supervisor.begin_escalation("tool")
            raise StopResponse()
        return result.text

    @staticmethod
    async def _end_call(context: RunContext) -> None:
        await context.wait_for_playout()
        context.session.shutdown(drain=True)
