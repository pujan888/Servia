from __future__ import annotations

import asyncio
import threading

from fastembed import TextEmbedding

from app.core.config import KnowledgeSettings


class Embedder:
    """A local embedding model, one per process. No network call per question.

    The model is loaded once. Embedding is CPU work, so it runs in a thread and
    never blocks the event loop that is carrying the call's audio."""

    def __init__(self, settings: KnowledgeSettings) -> None:
        self.model_name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                self._model = TextEmbedding(self.model_name)
                list(self._model.query_embed(["warm up"]))

    async def query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._query, text)

    async def passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._passages, texts)

    def _query(self, text: str) -> list[float]:
        self.load()
        return next(iter(self._model.query_embed([text]))).tolist()

    def _passages(self, texts: list[str]) -> list[list[float]]:
        self.load()
        return [vector.tolist() for vector in self._model.passage_embed(texts)]
