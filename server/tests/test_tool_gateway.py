from __future__ import annotations

import uuid

from app.agent.tools.contracts import Refusal
from app.agent.tools.gateway import ToolGateway
from app.agent.tools.registry import ToolRegistry
from app.core.enums import IdentityLevel, LatencyClass, SideEffect, ToolStatus
from tests.factories import AgentRuntimeFactory, ConfigFactory, PackFactory, ToolFactory
from tests.fakes import FakeCache, FakeJournal, FakeOutbox, FakeTool

REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {"reference": {"type": "string"}},
    "required": ["reference"],
}


class GatewayHarness:
    def __init__(self, *tools: FakeTool, bindings=(), identity=IdentityLevel.VERIFIED) -> None:
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)
        self.journal, self.outbox, self.cache = FakeJournal(), FakeOutbox(), FakeCache()
        self.gateway = ToolGateway(self.registry, self.journal, self.outbox, self.cache)
        pack = PackFactory.build()
        pack.identity_level = identity
        self.state = AgentRuntimeFactory.state(pack, ConfigFactory.build(tools=bindings))

    async def call(self, key: str, arguments: dict | None = None):
        return await self.gateway.invoke(self.state, key, arguments or {})


class TestGatewayRefusals:
    async def test_unbound_tool_is_refused(self) -> None:
        harness = GatewayHarness()
        result = await harness.call("refund_everything")
        assert (result.status, result.reason) == (ToolStatus.REFUSED, "not_bound")

    async def test_missing_argument_is_refused(self) -> None:
        tool = FakeTool("get_work_item")
        harness = GatewayHarness(
            tool, bindings=[ToolFactory.binding("get_work_item", parameters=REFERENCE_SCHEMA)]
        )
        result = await harness.call("get_work_item", {})
        assert result.reason == "invalid_arguments"
        assert tool.calls == []

    async def test_an_argument_the_schema_does_not_list_is_refused(self) -> None:
        harness = GatewayHarness(
            FakeTool("get_work_item"),
            bindings=[ToolFactory.binding("get_work_item", parameters=REFERENCE_SCHEMA)],
        )
        result = await harness.call("get_work_item", {"reference": "A1", "customer_id": "x"})
        assert result.reason == "invalid_arguments"

    async def test_identity_below_the_required_level_is_refused(self) -> None:
        harness = GatewayHarness(
            FakeTool("get_open_work"),
            bindings=[
                ToolFactory.binding("get_open_work", required_level=IdentityLevel.RECOGNISED)
            ],
            identity=IdentityLevel.UNKNOWN,
        )
        assert (await harness.call("get_open_work")).reason == "identity_too_low"

    async def test_missing_precondition_is_refused_with_its_instruction(self) -> None:
        refusal = Refusal("work_item_not_viewed", "Look it up first.")
        harness = GatewayHarness(
            FakeTool("add_work_note", precondition=refusal, side_effect=SideEffect.WRITE),
            bindings=[ToolFactory.binding("add_work_note", latency_class=LatencyClass.C)],
        )
        result = await harness.call("add_work_note")
        assert result.reason == "work_item_not_viewed"
        assert result.text == "NOT DONE. Look it up first. Do not tell the caller that it is done."
        assert harness.outbox.rows == {}

    async def test_calls_over_the_limit_are_refused(self) -> None:
        harness = GatewayHarness(
            FakeTool("get_open_work"), bindings=[ToolFactory.binding("get_open_work", max_calls=2)]
        )
        results = [await harness.call("get_open_work") for _ in range(3)]
        assert [r.status for r in results] == [ToolStatus.OK, ToolStatus.OK, ToolStatus.REFUSED]
        assert results[2].reason == "rate_exceeded"


class TestGatewayExecution:
    async def test_a_slow_tool_returns_the_fallback_on_timeout(self) -> None:
        harness = GatewayHarness(
            FakeTool("slow", delay=1.0), bindings=[ToolFactory.binding("slow", timeout_ms=100)]
        )
        result = await harness.call("slow")
        assert (result.status, result.reason) == (ToolStatus.FAILED, "timeout")
        assert "slow is not available. Offer a callback." in result.text
        assert result.text.startswith("NOT DONE.")

    async def test_successful_results_are_not_marked_not_done(self) -> None:
        harness = GatewayHarness(
            FakeTool("get_open_work"), bindings=[ToolFactory.binding("get_open_work")]
        )
        assert (await harness.call("get_open_work")).text == "done"

    async def test_a_crash_returns_the_fallback_never_the_error(self) -> None:
        harness = GatewayHarness(
            FakeTool("broken", error=RuntimeError("database password=hunter2")),
            bindings=[ToolFactory.binding("broken")],
        )
        result = await harness.call("broken")
        assert result.status == ToolStatus.FAILED
        assert "hunter2" not in result.text

    async def test_a_write_is_deferred_and_returns_a_spoken_receipt(self) -> None:
        tool = FakeTool("request_callback", side_effect=SideEffect.WRITE)
        harness = GatewayHarness(
            tool, bindings=[ToolFactory.binding("request_callback", latency_class=LatencyClass.C)]
        )
        result = await harness.call("request_callback")
        assert result.status == ToolStatus.ACCEPTED
        assert "R, E, F, zero, zero, zero" in result.text
        assert tool.calls == []

    async def test_the_same_write_twice_gives_one_action_and_one_reference(self) -> None:
        harness = GatewayHarness(
            FakeTool("request_callback", side_effect=SideEffect.WRITE),
            bindings=[ToolFactory.binding("request_callback", latency_class=LatencyClass.C)],
        )
        first = await harness.call("request_callback", {})
        second = await harness.call("request_callback", {})
        assert first.reference == second.reference
        assert len(harness.outbox.rows) == 1


class TestGatewayCache:
    def harness(self) -> GatewayHarness:
        self.read = FakeTool("get_work_item")
        return GatewayHarness(
            self.read,
            FakeTool("add_work_note", side_effect=SideEffect.WRITE),
            bindings=[
                ToolFactory.binding("get_work_item", cache_ttl_s=30),
                ToolFactory.binding(
                    "add_work_note", latency_class=LatencyClass.C, invalidates=("get_work_item",)
                ),
            ],
        )

    async def test_second_read_comes_from_the_cache(self) -> None:
        harness = self.harness()
        await harness.call("get_work_item")
        assert (await harness.call("get_work_item")).from_cache
        assert len(self.read.calls) == 1

    async def test_cache_is_per_caller(self) -> None:
        harness = self.harness()
        await harness.call("get_work_item")
        harness.state.pack.caller.id = uuid.UUID(int=999)
        assert not (await harness.call("get_work_item")).from_cache

    async def test_a_write_clears_the_reads_it_names(self) -> None:
        harness = self.harness()
        await harness.call("get_work_item")
        await harness.call("add_work_note")
        assert not (await harness.call("get_work_item")).from_cache


class TestGatewayJournal:
    async def test_every_call_is_journalled_including_refusals(self) -> None:
        harness = GatewayHarness(
            FakeTool("get_open_work"), bindings=[ToolFactory.binding("get_open_work")]
        )
        await harness.call("get_open_work")
        await harness.call("unknown_tool")
        assert [e.decision for e in harness.journal.entries] == ["permitted", "refused"]
        assert all(e.latency_ms is not None for e in harness.journal.entries)

    async def test_sensitive_arguments_are_masked(self) -> None:
        schema = {"type": "object", "properties": {"card_number": {"type": "string"}}}
        harness = GatewayHarness(
            FakeTool("pay"), bindings=[ToolFactory.binding("pay", parameters=schema)]
        )
        await harness.call("pay", {"card_number": "4111111111111111"})
        assert harness.journal.entries[0].arguments == {"card_number": "***"}

    async def test_writes_record_their_idempotency_key(self) -> None:
        harness = GatewayHarness(
            FakeTool("request_callback", side_effect=SideEffect.WRITE),
            bindings=[ToolFactory.binding("request_callback", latency_class=LatencyClass.C)],
        )
        await harness.call("request_callback")
        assert harness.journal.entries[0].idempotency_key
