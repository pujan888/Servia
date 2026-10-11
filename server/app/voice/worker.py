from __future__ import annotations

import asyncio
import uuid

from livekit import agents
from livekit.agents import AgentServer, JobContext, JobProcess
from loguru import logger

from app.agent.runtime import AgentRuntime
from app.agent.setup import CallSetup
from app.agent.supervisor import TurnSupervisor
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.outbox import OutboxService
from app.core.config import Settings, SettingsFactory
from app.core.container import Container
from app.core.logging import LogConfig
from app.modules.calls.journal import ToolJournalWriter
from app.modules.calls.recorder import TurnRecorder
from app.modules.calls.service import CallService
from app.modules.context.assembler import ContextAssembler
from app.voice.dispatch import CallDispatch
from app.voice.fillers import FillerLibrary
from app.voice.metrics import TurnTimer


class CallWorker:
    """One instance per job process. Loads the caller's context, then talks."""

    def __init__(self, container: Container) -> None:
        self._container = container
        self._fillers = FillerLibrary()
        self._runtime = AgentRuntime(container.settings, self._fillers)
        self._outbox = OutboxService(container.database)
        self._setup = CallSetup(container, self._runtime)
        self._warm = False

    async def handle(self, ctx: JobContext) -> None:
        if not self._warm:
            await self._container.database.warm(ContextAssembler.PARALLEL_LOADERS)
            self._warm = True
        logger.info("handling call job {}", ctx.job.id)
        call_id = self._resolve_call_id(ctx, self._container.settings)
        state = await self._setup.prepare(call_id)
        logger.info("the state is ready for call {}: {}", call_id, state)

        database = self._container.database
        recorder = TurnRecorder(database, state.organization_id, call_id)
        journal = ToolJournalWriter(database)
        gateway = ToolGateway(
            self._container.registry, journal, self._outbox, self._container.cache
        )
        timer = TurnTimer(state, recorder, self._container.cache.stats)
        supervisor = self._runtime.supervisor(state, gateway, database)
        session = self._runtime.session(state)
        self._runtime.prepare_greeting(state, session.tts)
        timer.attach(session)
        supervisor.attach(session)
        asyncio.create_task(self._fillers.warm(session.tts, self._runtime.voice(state)))
        ctx.add_shutdown_callback(  # at hang up
            lambda: self._close(call_id, supervisor, timer, recorder, journal)
        )
        await session.start(  # start with the agent
            room=ctx.room, agent=self._runtime.agent(state, gateway, supervisor)
        )

    def _resolve_call_id(self, ctx: JobContext, settings: Settings) -> uuid.UUID:
        if ctx.job.metadata:
            return CallDispatch.decode(ctx.job.metadata).call_id
        if settings.dev_call_id is not None:  # test-only fallback for local development
            logger.warning("no dispatch metadata; falling back to DEV_CALL_ID")
            return settings.dev_call_id
        raise RuntimeError("job dispatched without call metadata and no DEV_CALL_ID is set")

    async def _close(
        self,
        call_id: uuid.UUID,
        supervisor: TurnSupervisor,
        timer: TurnTimer,
        recorder: TurnRecorder,
        journal: ToolJournalWriter,
    ) -> None:
        await supervisor.drain()
        timer.close()
        await asyncio.gather(recorder.aclose(), journal.aclose())
        async with self._container.database.session() as session:
            await CallService(session).mark_ended(call_id)


class WorkerRuntime:
    """Module-level entry points, so the job subprocess can import them by name."""

    KEY = "call_worker"

    @staticmethod
    def setup(proc: JobProcess) -> None:
        container = Container()
        container.embedder.load()
        proc.userdata[WorkerRuntime.KEY] = CallWorker(container)

    @staticmethod
    async def handle(ctx: JobContext) -> None:
        worker: CallWorker = ctx.proc.userdata[WorkerRuntime.KEY]
        await worker.handle(ctx)


settings = SettingsFactory.load()
settings.livekit.export_env()
LogConfig.install(settings.log_level, intercept=False)

server = AgentServer(
    setup_fnc=WorkerRuntime.setup,
    ws_url=settings.livekit.url,
    api_key=settings.livekit.api_key,
    api_secret=settings.livekit.api_secret,
    num_idle_processes=settings.livekit.idle_processes,
)
server.rtc_session(WorkerRuntime.handle, agent_name=settings.livekit.agent_name)


if __name__ == "__main__":
    agents.cli.run_app(server)
# call_id
#   ↓
# Get call from DB
#   ↓
# Resolve caller identity/level
#   ↓
# Assemble context
#   ↓
# Create CallState
#   ↓
# Compile the AI prompt
#   ↓
# Mark call active
#   ↓
# Save complete call context
#   ↓
# Check prompt/token budget
#   ↓
# Return CallState
