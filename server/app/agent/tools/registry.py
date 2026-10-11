from __future__ import annotations

from app.agent.tools.base import ToolImplementation
from app.agent.tools.implementations.call import EndCall, EscalateToHuman
from app.agent.tools.implementations.knowledge import SearchKnowledge
from app.agent.tools.implementations.memory import GetCallHistory
from app.agent.tools.implementations.work import (
    AddWorkNote,
    CreateWorkItem,
    GetOpenWork,
    GetWorkItem,
    RequestCallback,
)
from app.core.config import MemorySettings
from app.core.database import Database
from app.modules.knowledge.retrieval import KnowledgeRouter
from app.modules.work.factory import WorkItemFactory


class ToolRegistry:
    """Maps an implementation name, as stored in a binding, to its code."""

    def __init__(self) -> None:
        self._implementations: dict[str, ToolImplementation] = {}

    def register(self, implementation: ToolImplementation) -> None:
        if implementation.name in self._implementations:
            raise ValueError(f"duplicate tool implementation '{implementation.name}'")
        self._implementations[implementation.name] = implementation

    def get(self, name: str) -> ToolImplementation | None:
        return self._implementations.get(name)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._implementations)


class PlatformTools:
    """Every implementation the platform provides."""

    @staticmethod
    def registry(
        database: Database,
        knowledge: KnowledgeRouter,
        memory: MemorySettings,
        work_items: WorkItemFactory,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        for implementation in (
            GetOpenWork(database),
            GetWorkItem(database),
            AddWorkNote(database),
            RequestCallback(work_items),
            CreateWorkItem(work_items),
            EndCall(),
            EscalateToHuman(),
            SearchKnowledge(knowledge),
            GetCallHistory(database, memory.history_limit),
        ):
            registry.register(implementation)
        return registry
