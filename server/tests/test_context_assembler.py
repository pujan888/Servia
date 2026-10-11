from __future__ import annotations

import asyncio

from app.core.config import ContextSettings
from app.core.enums import IdentityLevel
from app.modules.context.assembler import ContextAssembler


class TestContextAssembler:
    async def test_pack_contains_only_its_own_tenant(self, database, tenants) -> None:
        call_a = await tenants.create("Tenant A", "Boiler for tenant A")
        await tenants.create("Tenant B", "Fridge for tenant B")

        pack = await ContextAssembler(database, ContextSettings()).assemble(
            call_a.id, IdentityLevel.VERIFIED
        )

        assert pack.organization.name == "Tenant A"
        assert [task.title for task in pack.tasks] == ["Boiler for tenant A"]

    async def test_slow_optional_loader_degrades_the_pack(self, database, tenants) -> None:
        call = await tenants.create("Tenant A", "Boiler")
        await database.warm(ContextAssembler.PARALLEL_LOADERS)
        assembler = ContextAssembler(database, ContextSettings())

        async def slow_tasks(user_id):
            await asyncio.sleep(2)
            return []

        assembler._tasks = slow_tasks
        pack = await assembler.assemble(call.id, IdentityLevel.VERIFIED)

        assert pack.degraded == ["tasks"]
        assert pack.tasks == []
