from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.agent.tools.adapter import FrameworkToolAdapter
from app.agent.tools.exposure import ToolExposure
from app.agent.tools.utils import ArgumentNormaliser, IdempotencyKey, SpokenFormatter
from app.core.enums import LatencyClass, ToolStatus
from app.modules.config.utils import SchemaGuard
from app.voice.fillers import FillerLibrary
from tests.factories import ConfigFactory, ToolFactory


class TestSchemaGuard:
    def test_identity_fields_are_rejected(self) -> None:
        schema = {"type": "object", "properties": {"customer_id": {"type": "string"}}}
        assert any("identity" in error for error in SchemaGuard.schema_errors(schema))

    def test_a_non_object_schema_is_rejected(self) -> None:
        assert SchemaGuard.schema_errors({"type": "string"})

    def test_a_good_schema_passes(self) -> None:
        schema = {"type": "object", "properties": {"reference": {"type": "string"}}}
        assert SchemaGuard.schema_errors(schema) == []


class TestSmallParts:
    def test_references_are_spelled_for_speech(self) -> None:
        assert SpokenFormatter.reference("rf-8814") == "R, F, eight, eight, one, four"

    def test_equal_arguments_give_the_same_key(self) -> None:
        a = {"topic": " Boiler  leak ", "reason": "Wet floor"}
        b = {"reason": "wet floor", "topic": "boiler leak"}
        assert ArgumentNormaliser.digest(a) == ArgumentNormaliser.digest(b)
        assert IdempotencyKey.build("c", "t", a) == IdempotencyKey.build("c", "t", b)


class TestToolExposure:
    def test_ten_tools_or_fewer_are_all_exposed(self) -> None:
        tools = [ToolFactory.binding(f"tool_{i}") for i in range(10)]
        config = ConfigFactory.build(tools=tools)
        assert len(ToolExposure().keys(config, [])) == 10

    def test_more_than_ten_tools_follow_the_skills(self) -> None:
        tools = [ToolFactory.binding(f"tool_{i}") for i in range(11)] + [
            ToolFactory.binding("end_call")
        ]
        skill = ConfigFactory.skill("check_order").model_copy(update={"tool_keys": ("tool_3",)})
        config = ConfigFactory.build(skill, tools=tools)
        assert ToolExposure().keys(config, ["check_order"]) == ["tool_3", "end_call"]


class FakeGateway:
    def __init__(self) -> None:
        self.calls = []

    async def invoke(self, state, key, arguments):
        self.calls.append((key, arguments))
        return SimpleNamespace(text=f"{key} ok", control=None, status=ToolStatus.OK)


class TestFrameworkToolAdapter:
    def context(self):
        self.fillers_used = []

        @asynccontextmanager
        async def with_filler(source, delay):
            self.fillers_used.append(delay)
            yield

        return SimpleNamespace(userdata=object(), session=object(), with_filler=with_filler)

    def test_each_binding_becomes_a_tool_with_its_schema(self) -> None:
        schema = {"type": "object", "properties": {"reference": {"type": "string"}}}
        adapter = FrameworkToolAdapter(FakeGateway(), FillerLibrary(), "Ashley", supervisor=None)
        [tool] = adapter.tools([ToolFactory.binding("get_work_item", parameters=schema)])
        assert tool.info.raw_schema["name"] == "get_work_item"
        assert tool.info.raw_schema["parameters"] == schema

    def test_class_b_tools_use_a_filler_and_class_a_tools_do_not(self) -> None:
        gateway = FakeGateway()
        adapter = FrameworkToolAdapter(gateway, FillerLibrary(), "Ashley", supervisor=None)
        slow = ToolFactory.binding("slow", latency_class=LatencyClass.B)
        fast = ToolFactory.binding("fast")
        asyncio.run(adapter._handle(slow, self.context(), {}))
        assert self.fillers_used == [FrameworkToolAdapter.FILLER_DELAY_S]
        asyncio.run(adapter._handle(fast, self.context(), {}))
        assert self.fillers_used == []
        assert [key for key, _ in gateway.calls] == ["slow", "fast"]
