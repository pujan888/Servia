from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatherResult:
    values: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


class DeadlineGather:
    """Runs named loaders at the same time. Optional loaders get a short deadline;
    required ones get a longer one. Whatever is late is cancelled and reported."""

    def __init__(self, deadline_ms: int, required_deadline_ms: int) -> None:
        self._deadline = deadline_ms / 1000
        self._required_deadline = required_deadline_ms / 1000

    async def run(
        self, required: dict[str, Any], optional: dict[str, Awaitable[Any]]
    ) -> GatherResult:
        """Run the loaders and return the results."""
        tasks = {name: asyncio.ensure_future(job) for name, job in {**required, **optional}.items()}
        await asyncio.wait(tasks.values(), timeout=self._deadline)
        late_required = [tasks[name] for name in required if not tasks[name].done()]
        if late_required:
            await asyncio.wait(late_required, timeout=self._required_deadline - self._deadline)
        result = GatherResult()
        for name, task in tasks.items():
            if task.done() and not task.cancelled() and task.exception() is None:
                result.values[name] = task.result()
                continue
            if not task.done():
                task.cancel()
            if name in required:
                raise TimeoutError(f"required context loader '{name}' did not complete") from (
                    task.exception() if task.done() and not task.cancelled() else None
                )
            result.missing.append(name)
        return result


# Get the information needed to start the AI agent quickly,
# without waiting forever for slow data sources.


# Wait for async tasks to make progress/finish, but give me control back when my condition is met or
# the timeout expires
