from __future__ import annotations

import uuid

import pytest

from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.greeting import Greeting
from app.agent.runtime import AgentRuntime
from app.core.config import PromptSettings, Settings, SettingsFactory
from app.core.database import Database
from app.core.enums import CallChannel, CallStatus, IdentityLevel
from app.core.tokens import TokenCounter
from app.modules.accounts.service import IdentityResolver
from app.modules.calls.service import CallService
from app.voice.utils import milliseconds
from tests.factories import ConfigFactory, PackFactory


class TestPromptSettings:
    def test_every_section_has_a_budget(self) -> None:
        settings = PromptSettings()
        for section in PromptCompiler.default_sections(TokenCounter()):
            assert hasattr(settings, section.budget_key), section.name

    def test_settings_carry_the_prompt_budget(self) -> None:
        assert isinstance(Settings.model_fields["prompt"].default, PromptSettings)


class TestIdentityResolver:
    @pytest.mark.parametrize(
        ("channel", "user_id", "level"),
        [
            (CallChannel.WEB, uuid.UUID(int=1), IdentityLevel.VERIFIED),
            (CallChannel.WEB, None, IdentityLevel.UNKNOWN),
            (CallChannel.PHONE, uuid.UUID(int=1), IdentityLevel.UNKNOWN),
        ],
    )
    def test_level(self, channel, user_id, level) -> None:
        assert IdentityResolver().resolve(channel, user_id) == level


class TestMilliseconds:
    @pytest.mark.parametrize(
        ("seconds", "expected"), [(0.18, 180), (1.2345, 1234), (0.0, 0), (-1.0, None), (None, None)]
    )
    def test_conversion(self, seconds, expected) -> None:
        assert milliseconds(seconds) == expected


class TestGreeting:
    def test_greets_by_first_name_and_asks_without_a_focus_service(self) -> None:
        text = Greeting(PackFactory.build()).instruction()
        assert "Greet Daniel by name" in text
        assert "ask how you can help" in text

    def test_mentions_the_focus_service(self) -> None:
        pack = PackFactory.build()
        pack.focus_service = pack.services[0]
        text = Greeting(pack).instruction()
        assert "about Service 0" in text
        assert "ask how you can help" in text


class TestCallLifecycle:
    async def test_active_then_completed(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        async with database.session() as session:
            await CallService(session).mark_active(
                call.id, IdentityLevel.VERIFIED, "eu-west", release_id=None
            )
        async with database.session() as session:
            active = await CallService(session).get(call.id)
            assert (active.status, active.identity_level) == (
                CallStatus.ACTIVE,
                IdentityLevel.VERIFIED,
            )
            await CallService(session).mark_ended(call.id)
        async with database.session() as session:
            assert (await CallService(session).get(call.id)).status == CallStatus.COMPLETED


class TestAgentRuntime:
    def test_state_compiles_the_prompt_for_the_worker(self) -> None:
        settings = SettingsFactory.load()
        runtime = AgentRuntime(settings, Database(settings.database))
        state = runtime.state(PackFactory.build(), ConfigFactory.build())
        assert state.compiled.prefix.startswith("How you speak:")
        assert state.identity_level == IdentityLevel.VERIFIED
