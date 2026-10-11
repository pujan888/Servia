from __future__ import annotations

import asyncio
import uuid

from loguru import logger

from app.agent.runtime import AgentRuntime
from app.agent.state import CallState
from app.core.container import Container
from app.modules.accounts.service import IdentityResolver
from app.modules.calls.models import CallContext
from app.modules.calls.service import CallContextService, CallService
from app.modules.context.assembler import ContextAssembler
from app.modules.releases.service import ReleaseCatalog


class CallSetup:
    """Everything before the first word, the same for a voice call and a text
    call: the context pack, the call's release, the compiled prompt, and the
    record of what the agent was given."""

    def __init__(self, container: Container, runtime: AgentRuntime) -> None:
        self._container = container
        self._runtime = runtime
        self._assembler = ContextAssembler(container.database, container.settings.context)
        self.catalog = ReleaseCatalog(container.database, container.cache, container.settings)

    async def prepare(self, call_id: uuid.UUID) -> CallState:
        # Prepares the call runtime by loading context, building the state
        # and prompt, saving call context, and validating token limits.
        database = self._container.database
        async with database.session() as session:
            call = await CallService(session).get(call_id)
        level = IdentityResolver().resolve(call.channel, call.user_id)
        pack, (release_id, config) = await asyncio.gather(
            self._assembler.assemble(call_id, level),
            self.catalog.for_call(call.organization_id, call_id),
        )
        state = self._runtime.state(pack, config)
        state.release_id = release_id
        compiled = state.compiled
        async with database.session() as session:
            await CallService(session).mark_active(
                call_id, level, self._container.settings.worker_region, release_id
            )
            await CallContextService(session).save(
                CallContext(
                    call_id=call_id,
                    organization_id=state.organization_id,
                    pack=pack.model_dump(mode="json"),
                    prefix=compiled.prefix,
                    prefix_hash=compiled.prefix_hash,
                    sections=[report.as_dict() for report in compiled.sections],
                    truncations=compiled.truncations,
                )
            )
        self._log(state)
        return state

    @staticmethod
    def _log(state: CallState) -> None:
        compiled, pack, config = state.compiled, state.pack, state.config
        over = [report.name for report in compiled.sections if report.over_budget]
        if over or compiled.over_total:
            logger.warning(
                "prompt over budget for call {}: sections={} total={}",
                state.call_id,
                over,
                compiled.total_tokens,
            )
        logger.info(
            "context ready for {}: {} tokens, release v{}, active skill {}, tools {}, "
            "degraded={}, truncations={}",
            pack.caller.full_name,
            compiled.total_tokens,
            config.version,
            state.active_skill,
            [binding.key for binding in config.tools],
            pack.degraded,
            compiled.truncations,
        )
