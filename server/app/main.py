from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.modules  # noqa: F401  (registers the ORM metadata)
from app.api import ApiRouter
from app.core.config import SettingsFactory
from app.core.container import Container
from app.core.errors import ErrorHandlers
from app.core.logging import LogConfig


class Application:
    STATIC_DIR = Path(__file__).parent / "static"

    def __init__(self) -> None:
        self._settings = SettingsFactory.load()
        LogConfig.install(self._settings.log_level)
        self.api = FastAPI(title="Servia", lifespan=self._lifespan)
        self._configure()

    @asynccontextmanager
    async def _lifespan(self, api: FastAPI) -> AsyncIterator[None]:
        api.state.container = Container(self._settings)
        try:
            yield
        finally:
            await api.state.container.aclose()

    def _configure(self) -> None:
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=self._settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        ErrorHandlers.install(self.api)
        self.api.add_api_route("/health", self._health, methods=["GET"], tags=["ops"])
        self.api.include_router(ApiRouter.build())
        self.api.mount("/", StaticFiles(directory=self.STATIC_DIR, html=True), name="static")

    async def _health(self) -> dict[str, str]:
        return {"status": "ok", "env": self._settings.env}


app = Application().api
