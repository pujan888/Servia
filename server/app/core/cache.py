from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import TypeAdapter, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError


class CacheKeys:
    """The only place that builds cache keys. The tenant is always first, then
    the key type from the register."""

    @staticmethod
    def build(organization_id: uuid.UUID, kind: str, *parts: object) -> str:
        CachePolicy.rule(kind)
        return ":".join(["t", str(organization_id), kind, *(str(part) for part in parts)])

    @staticmethod
    def kind(key: str) -> str:
        return key.split(":", 3)[2]


@dataclass(frozen=True, slots=True)
class CacheRule:
    kind: str
    ttl_s: int | None
    shared: bool
    local: bool
    invalidation: str


class CachePolicy:
    """The cache register. A key type that is not here cannot be built.

    `shared` puts it in the key-value store, seen by every process. `local` also
    keeps it in process memory; only a key with a version in it may be local,
    because another process cannot delete a local copy."""

    RULES = (
        CacheRule("release", 604_800, True, True, "none needed: a release never changes"),
        CacheRule("live", 60, True, False, "deleted on every promotion and rollback"),
        CacheRule("kb", 300, True, True, "collection versions in the key"),
        CacheRule("qvec", 3_600, False, True, "none needed: the same text gives the same vector"),
        CacheRule("toolres", None, True, False, "the binding's life; write tools delete it"),
        CacheRule("blocks", 300, True, False, "deleted on every change"),
        CacheRule("lock", 5, True, False, "released by its owner, or expires"),
    )

    @classmethod
    def rule(cls, kind: str) -> CacheRule:
        for rule in cls.RULES:
            if rule.kind == kind:
                return rule
        raise KeyError(f"cache key type '{kind}' is not in the register")


class MemoryCache:
    """A per-process cache with a life for each entry and a size limit."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self._max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_s: float) -> None:
        self._items[key] = (time.monotonic() + ttl_s, value)
        self._items.move_to_end(key)
        while len(self._items) > self._max_entries:
            self._items.popitem(last=False)

    def delete_prefix(self, prefix: str) -> int:
        doomed = [key for key in self._items if key.startswith(prefix)]
        for key in doomed:
            del self._items[key]
        return len(doomed)


class RedisStore:
    """The key-value level. Every call can fail; the cache layer treats a
    failure as a miss, so a Redis outage makes calls slower, never broken."""

    def __init__(self, url: str) -> None:
        self._redis = Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25)

    async def get(self, key: str) -> bytes | None:
        return await self._redis.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._redis.set(key, value, ex=ttl_s)

    async def delete_prefix(self, prefix: str) -> None:
        keys = [key async for key in self._redis.scan_iter(match=f"{prefix}*", count=500)]
        if keys:
            await self._redis.delete(*keys)

    async def acquire(self, key: str, ttl_s: int) -> bool:
        return bool(await self._redis.set(key, b"1", nx=True, ex=ttl_s))

    async def release(self, key: str) -> None:
        await self._redis.delete(key)

    async def aclose(self) -> None:
        await self._redis.aclose()


class CacheStats:
    """Hits and misses by key type, taken once per turn for the turn record."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def add(self, kind: str, outcome: str) -> None:
        self._counts[f"{kind}.{outcome}"] += 1

    def take(self) -> dict[str, int]:
        counts, self._counts = dict(self._counts), Counter()
        return counts


class SingleFlight:
    """Only one loader runs for a key, across every process. In a process, the
    others await the same task. Across processes, a Redis lock picks one; the
    rest wait for its value to appear, then read it."""

    POLL_S = 0.02

    def __init__(self, store: RedisStore) -> None:
        self._store = store
        self._running: dict[str, asyncio.Task] = {}

    async def run(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        read: Callable[[], Awaitable[Any | None]],
        across: bool,
    ) -> tuple[Any, bool]:
        """Returns the value and whether this process loaded it."""
        task = self._running.get(key)
        if task is None:
            work = self._across(key, loader, read) if across else self._here(loader)
            task = asyncio.ensure_future(work)
            self._running[key] = task
            task.add_done_callback(lambda _: self._running.pop(key, None))
        return await asyncio.shield(task)

    @staticmethod
    async def _here(loader) -> tuple[Any, bool]:
        return await loader(), True

    async def _across(self, key, loader, read) -> tuple[Any, bool]:
        lock = f"{key}:lock"
        rule = CachePolicy.rule("lock")
        try:
            owner = await self._store.acquire(lock, rule.ttl_s)
        except RedisError:
            return await loader(), True
        if owner:
            try:
                return await loader(), True
            finally:
                await self._release(lock)
        deadline = time.monotonic() + rule.ttl_s
        while time.monotonic() < deadline:
            await asyncio.sleep(self.POLL_S)
            value = await read()
            if value is not None:
                return value, False
        return await loader(), True

    async def _release(self, lock: str) -> None:
        try:
            await self._store.release(lock)
        except RedisError:
            logger.warning("cache lock {} was not released; it expires by itself", lock)


class CacheLayer:
    """Two levels: process memory, then the key-value store. The register
    decides which level a key type uses and how long it lives."""

    def __init__(self, memory: MemoryCache, store: RedisStore) -> None:
        self._memory = memory
        self._store = store
        self._flight = SingleFlight(store)
        self.stats = CacheStats()

    async def get[T](
        self,
        key: str,
        adapter: TypeAdapter[T],
        loader: Callable[[], Awaitable[T]],
        ttl_s: int | None = None,
    ) -> T:
        found = await self.peek(key, adapter)
        if found is not None:
            return found
        shared = CachePolicy.rule(CacheKeys.kind(key)).shared
        value, loaded = await self._flight.run(
            key, loader, lambda: self._read_store(key, adapter), across=shared
        )
        if loaded:
            await self.put(key, value, adapter, ttl_s)
        return value

    async def peek[T](self, key: str, adapter: TypeAdapter[T]) -> T | None:
        rule = CachePolicy.rule(CacheKeys.kind(key))
        if rule.local and (value := self._memory.get(key)) is not None:
            self.stats.add(rule.kind, "memory")
            return value
        value = await self._read_store(key, adapter) if rule.shared else None
        if value is None:
            self.stats.add(rule.kind, "miss")
            return None
        self.stats.add(rule.kind, "store")
        if rule.local:
            self._memory.set(key, value, rule.ttl_s)
        return value

    async def put[T](
        self, key: str, value: T, adapter: TypeAdapter[T], ttl_s: int | None = None
    ) -> None:
        rule = CachePolicy.rule(CacheKeys.kind(key))
        life = ttl_s or rule.ttl_s
        if rule.local:
            self._memory.set(key, value, life)
        if rule.shared:
            try:
                await self._store.set(key, adapter.dump_json(value), life)
            except RedisError as error:
                logger.warning("cache write failed for {}: {!r}", rule.kind, error)

    async def delete_prefix(self, prefix: str) -> None:
        self._memory.delete_prefix(prefix)
        try:
            await self._store.delete_prefix(prefix)
        except RedisError as error:
            logger.error("cache delete failed for {}; it expires by itself: {!r}", prefix, error)

    async def aclose(self) -> None:
        await self._store.aclose()

    async def _read_store[T](self, key: str, adapter: TypeAdapter[T]) -> T | None:
        try:
            raw = await self._store.get(key)
        except RedisError as error:
            logger.warning("cache read failed for {}: {!r}", CacheKeys.kind(key), error)
            return None
        if raw is None:
            return None
        try:
            return adapter.validate_json(raw)
        except ValidationError:
            logger.warning(
                "cached {} no longer fits its type; loading it again", CacheKeys.kind(key)
            )
            return None
