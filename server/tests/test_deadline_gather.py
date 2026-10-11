from __future__ import annotations

import asyncio
import time

import pytest

from app.modules.context.utils import DeadlineGather


class TestDeadlineGather:
    @staticmethod
    async def value(result: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return result

    async def test_slow_optional_loader_is_reported_missing_on_time(self) -> None:
        gather = DeadlineGather(deadline_ms=250, required_deadline_ms=1000)
        started = time.perf_counter()
        result = await gather.run(
            required={"organization": self.value("org", 0.01)},
            optional={"tasks": self.value("tasks", 2.0), "services": self.value("s", 0.01)},
        )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.26
        assert result.missing == ["tasks"]
        assert result.values == {"organization": "org", "services": "s"}

    async def test_loaders_run_in_parallel(self) -> None:
        gather = DeadlineGather(deadline_ms=250, required_deadline_ms=1000)
        started = time.perf_counter()
        await gather.run(
            required={"a": self.value("a", 0.1), "b": self.value("b", 0.1)},
            optional={"c": self.value("c", 0.1)},
        )
        assert time.perf_counter() - started < 0.2

    async def test_slow_required_loader_gets_the_longer_deadline(self) -> None:
        gather = DeadlineGather(deadline_ms=100, required_deadline_ms=1000)
        result = await gather.run(required={"caller": self.value("c", 0.3)}, optional={})
        assert result.values["caller"] == "c"

    async def test_required_loader_past_the_hard_deadline_fails(self) -> None:
        gather = DeadlineGather(deadline_ms=50, required_deadline_ms=100)
        with pytest.raises(TimeoutError):
            await gather.run(required={"caller": self.value("c", 1.0)}, optional={})

    async def test_optional_loader_that_raises_is_reported_missing(self) -> None:
        async def broken() -> None:
            raise RuntimeError("down")

        gather = DeadlineGather(deadline_ms=250, required_deadline_ms=1000)
        result = await gather.run(required={}, optional={"digest": broken()})
        assert result.missing == ["digest"]
