from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from livekit.agents import AgentSession


class HumanAvailability(ABC):
    """Whether a person is free on a queue. A staffing or contact-centre system
    implements this; the platform does not guess."""

    @abstractmethod
    async def available(self, organization_id: uuid.UUID, queue: str) -> bool: ...


class NoStaffSystem(HumanAvailability):
    """The default until a staffing system is connected: nobody is free, so the
    fallback levels run. It never promises a transfer that cannot happen."""

    async def available(self, organization_id: uuid.UUID, queue: str) -> bool:
        return False


class TransferTransport(ABC):
    """Moves the caller to the person: for phone calls, a SIP transfer; for web
    calls, a person joining the room."""

    @abstractmethod
    async def warm_transfer(self, session: AgentSession, queue: str, briefing: str) -> bool: ...


class NoTransport(TransferTransport):
    async def warm_transfer(self, session: AgentSession, queue: str, briefing: str) -> bool:
        return False
