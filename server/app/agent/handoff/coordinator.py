from __future__ import annotations

from livekit.agents import AgentSession
from loguru import logger

from app.agent.handoff.briefing import BriefingBuilder
from app.agent.handoff.transfer import HumanAvailability, TransferTransport
from app.agent.state import CallState
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.utils import SpokenFormatter
from app.core.database import Database
from app.core.enums import Escalation, ToolStatus
from app.modules.calls.models import Handoff
from app.modules.calls.service import HandoffService


class HandoffCoordinator:
    """The escalation. Every word the caller hears here is fixed text: the model
    does not decide what is promised. Nothing is promised before it is true."""

    HOLD = "I'm connecting you to a colleague now. I'll tell them what we've talked about."
    CALLBACK = (
        "Nobody is free to take the call right now, so I've asked a colleague to call you "
        "back. Your reference is {reference}."
    )
    MESSAGE = (
        "Nobody is free right now, so I've passed your message to the team. Your reference "
        "is {reference}."
    )
    FAILED_PHONE = "I'm sorry, I couldn't pass this on. Please call us directly on {phone}."
    FAILED = "I'm sorry, I couldn't pass this on. Please call us again a little later."
    WORK_TOOL = "create_work_item"
    FALLBACKS = ((1, "callback", CALLBACK), (2, "message", MESSAGE))

    def __init__(
        self,
        database: Database,
        availability: HumanAvailability,
        transport: TransferTransport,
        briefing: BriefingBuilder,
        gateway: ToolGateway,
    ) -> None:
        self._database = database
        self._availability = availability
        self._transport = transport
        self._briefing = briefing
        self._gateway = gateway

    async def escalate(self, session: AgentSession, state: CallState, reason: str) -> Handoff:
        state.signals.escalation, state.signals.escalation_reason = Escalation.IN_PROGRESS, reason
        briefing = self._briefing.current(state, reason)
        level, result, reference = await self._handoff(session, state, briefing)
        handoff = Handoff(
            organization_id=state.organization_id,
            call_id=state.call_id,
            reason=reason,
            briefing=briefing,
            level=level,
            result=result,
            reference=reference,
        )
        async with self._database.session() as db:
            await HandoffService(db).record(handoff)
        state.signals.escalation = Escalation.DONE
        logger.info("handoff for call {}: {} → level {} {}", state.call_id, reason, level, result)
        return handoff

    async def _handoff(
        self, session: AgentSession, state: CallState, briefing: str
    ) -> tuple[int, str, str | None]:
        route = state.config.route(state.config.HANDOFF_ROUTE)
        if route and await self._availability.available(state.organization_id, route.queue):
            await session.say(self.HOLD, allow_interruptions=False).wait_for_playout()
            if await self._transport.warm_transfer(session, route.queue, briefing):
                return 0, "transferred", None
        for level, kind, text in self.FALLBACKS:
            if kind not in state.config.work_item_types:
                continue
            result = await self._gateway.invoke(
                state, self.WORK_TOOL, {"type": kind, "description": briefing}
            )
            if result.status == ToolStatus.ACCEPTED and result.reference:
                spoken = SpokenFormatter.reference(result.reference)
                await session.say(text.format(reference=spoken)).wait_for_playout()
                return level, kind, result.reference
            break
        phone = state.pack.organization.phone
        await session.say(
            self.FAILED_PHONE.format(phone=phone) if phone else self.FAILED
        ).wait_for_playout()
        return 3, "failed", None
