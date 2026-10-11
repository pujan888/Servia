from __future__ import annotations

from datetime import timedelta

from livekit.api import (
    AccessToken,
    RoomAgentDispatch,
    RoomConfiguration,
    VideoGrants,
)

from app.core.config import LiveKitSettings
from app.modules.calls.models import Call
from app.modules.calls.schemas import CallCredentials
from app.modules.calls.utils import ParticipantIdentity
from app.voice.dispatch import CallDispatch


class TokenService:
    """Mints a join token that also tells LiveKit which agent to dispatch, and why."""

    def __init__(self, settings: LiveKitSettings) -> None:
        self._settings = settings

    def issue(self, call: Call) -> CallCredentials:
        # LiveKit JWT that lets this one caller join this one room
        identity = ParticipantIdentity.for_user(call.user_id)
        dispatch = CallDispatch(
            call_id=call.id, organization_id=call.organization_id, user_id=call.user_id
        )
        token = (
            AccessToken(self._settings.api_key, self._settings.api_secret)
            .with_identity(identity)
            .with_ttl(timedelta(seconds=self._settings.token_ttl_seconds))
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=call.room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name=self._settings.agent_name,
                            metadata=dispatch.encode(),
                        )
                    ]
                )
            )
            .to_jwt()
        )
        return CallCredentials(
            call_id=call.id,
            room_name=call.room_name,
            server_url=self._settings.url,
            token=token,
            identity=identity,
        )
