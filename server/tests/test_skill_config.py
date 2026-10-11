from __future__ import annotations

import pytest

from app.agent.runtime import EndpointingProfile
from app.core.enums import Eagerness
from app.core.errors import Conflict, Invalid
from app.modules.config.schemas import SkillWrite
from app.modules.config.service import ConfigVersionService, SkillService


def skill_write(key: str, trigger: str) -> SkillWrite:
    return SkillWrite(key=key, name=key, trigger=trigger, goal="Done.", steps="1. Do it.")


class TestSkillService:
    async def test_create_bumps_the_config_version(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        async with database.session() as session:
            before = await ConfigVersionService(session).current(call.organization_id)
            await SkillService(session).create(
                call.organization_id,
                skill_write(
                    "check_order", "When the caller asks where an order is or if it shipped"
                ),
            )
            after = await ConfigVersionService(session).current(call.organization_id)
        assert after == before + 1

    async def test_short_trigger_is_rejected(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        with pytest.raises(Invalid):
            async with database.session() as session:
                await SkillService(session).create(
                    call.organization_id, skill_write("x_skill", "bookings")
                )

    async def test_system_skill_cannot_be_disabled(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        data = skill_write(
            "general_inquiry", "When the caller has a general question about anything"
        )
        async with database.session() as session:
            await SkillService(session).create(call.organization_id, data, system=True)
        with pytest.raises(Conflict):
            async with database.session() as session:
                await SkillService(session).update(
                    call.organization_id,
                    "general_inquiry",
                    data.model_copy(update={"enabled": False}),
                )


class TestEndpointingProfile:
    def test_each_eagerness_maps_to_dynamic_endpointing(self) -> None:
        for eagerness in Eagerness:
            assert EndpointingProfile.for_eagerness(eagerness)["mode"] == "dynamic"

    def test_low_eagerness_waits_longer_than_high(self) -> None:
        low = EndpointingProfile.for_eagerness(Eagerness.LOW)
        high = EndpointingProfile.for_eagerness(Eagerness.HIGH)
        assert low["min_delay"] > high["min_delay"]
