from __future__ import annotations

from fastapi import APIRouter

from app.modules.accounts.router import router as accounts_router
from app.modules.calls.router import router as calls_router
from app.modules.catalog.router import router as catalog_router
from app.modules.config.router import router as config_router
from app.modules.context.router import router as context_router
from app.modules.detection.router import router as detection_router
from app.modules.evaluation.router import router as evaluation_router
from app.modules.knowledge.router import router as knowledge_router
from app.modules.memory.router import router as memory_router
from app.modules.releases.router import router as releases_router
from app.modules.work.router import router as work_router


class ApiRouter:
    PREFIX = "/v1"
    MODULES = (
        accounts_router,
        catalog_router,
        work_router,
        calls_router,
        context_router,
        config_router,
        knowledge_router,
        memory_router,
        detection_router,
        releases_router,
        evaluation_router,
    )

    @classmethod
    def build(cls) -> APIRouter:
        router = APIRouter(prefix=cls.PREFIX)
        for module in cls.MODULES:
            router.include_router(module)
        return router
