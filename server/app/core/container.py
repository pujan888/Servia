from __future__ import annotations

from app.agent.tools.registry import PlatformTools
from app.analysis.model import AnalysisModel
from app.core.cache import CacheLayer, MemoryCache, RedisStore
from app.core.config import Settings, SettingsFactory
from app.core.database import Database
from app.modules.knowledge.catalog import CardCatalog
from app.modules.knowledge.embedder import Embedder
from app.modules.knowledge.retrieval import KnowledgeRouter
from app.modules.work.factory import TableRouter, WorkItemFactory


class Container:
    """Process-scoped resources. Built once per process, disposed on shutdown."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SettingsFactory.load()
        self.database = Database(self.settings.database)
        self.cache = CacheLayer(
            MemoryCache(self.settings.cache.memory_entries), RedisStore(self.settings.cache.url)
        )
        self.embedder = Embedder(self.settings.knowledge)
        self.knowledge_router = KnowledgeRouter(
            self.database,
            self.embedder,
            CardCatalog(self.database),
            self.cache,
            self.settings.knowledge,
        )
        self.work_items = WorkItemFactory(
            self.database,
            self.embedder,
            AnalysisModel(self.settings.speech.llm_model, self.settings.memory.llm_timeout_s),
            TableRouter(self.database),
            self.settings.work,
            self.settings.detection,
        )
        self.registry = PlatformTools.registry(
            self.database, self.knowledge_router, self.settings.memory, self.work_items
        )

    async def aclose(self) -> None:
        await self.cache.aclose()
        await self.database.dispose()
