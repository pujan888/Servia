from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from app.agent.agent import ConversationAgent
from app.core.cache import CacheKeys, CacheLayer, CachePolicy, MemoryCache, RedisStore
from app.core.config import SettingsFactory
from app.core.enums import FlagType, Optimisation, Priority
from app.modules.config.schemas import DetectionSpec, RouteSpec
from app.modules.config.utils import OptimisationFlags
from app.modules.detection.guard import OutputGuard
from app.modules.releases.service import ReleaseCatalog
from app.voice.greeting import GreetingPreparer, PreparedGreeting
from tests.factories import AgentRuntimeFactory, ConfigFactory, PackFactory
from tests.fakes import FakeModel

TEXT = TypeAdapter(str)
URL = SettingsFactory.load().cache.url


def layer(url: str = URL) -> CacheLayer:
    """A separate layer is a separate process: its own memory and its own lock map."""
    return CacheLayer(MemoryCache(), RedisStore(url))


def key(kind: str = "kb", *parts) -> str:
    return CacheKeys.build(uuid.uuid4(), kind, *parts)


class TestRegister:
    def test_a_key_type_outside_the_register_cannot_be_built(self) -> None:
        with pytest.raises(KeyError):
            CacheKeys.build(uuid.uuid4(), "anything")

    def test_only_versioned_keys_are_kept_in_process_memory(self) -> None:
        local = {rule.kind for rule in CachePolicy.RULES if rule.local}
        assert local == {"release", "kb", "qvec"}


class TestCacheLayer:
    async def test_twenty_concurrent_reads_across_two_processes_load_once(self) -> None:
        loads = []

        async def loader() -> str:
            loads.append(1)
            await asyncio.sleep(0.05)
            return "value"

        first, second, shared = layer(), layer(), key("release", 7)
        results = await asyncio.gather(
            *(cache.get(shared, TEXT, loader) for cache in [first, second] * 10)
        )
        assert results == ["value"] * 20 and len(loads) == 1
        await first.aclose()
        await second.aclose()

    async def test_a_value_in_the_store_is_read_by_another_process(self) -> None:
        first, second, shared = layer(), layer(), key()
        await first.put(shared, "hello", TEXT)
        assert await second.peek(shared, TEXT) == "hello"
        assert second.stats.take() == {"kb.store": 1}
        assert await second.peek(shared, TEXT) == "hello"
        assert second.stats.take() == {"kb.memory": 1}
        await first.aclose()
        await second.aclose()

    async def test_a_delete_in_one_process_is_seen_by_every_process(self) -> None:
        first, second = layer(), layer()
        result = key("toolres", "get_work_item", "caller", "args")
        await first.put(result, "old status", TEXT, ttl_s=30)
        assert await second.peek(result, TEXT) == "old status"
        await first.delete_prefix(result.rsplit(":", 1)[0])
        assert await second.peek(result, TEXT) is None
        await first.aclose()
        await second.aclose()

    async def test_a_value_written_by_older_code_is_loaded_again(self) -> None:
        """After a deploy that changes a cached type, the old entries must not
        break calls. Found in testing: a renamed switch did exactly this."""
        cache, stale = layer(), key("release", 3)
        await cache.put(stale, {"greeting_audio": 100}, TypeAdapter(dict))
        fresh = await layer().get(stale, TypeAdapter(dict[Optimisation, int]), self._flags)
        assert fresh == {Optimisation.PREPARED_GREETING: 100}
        await cache.aclose()

    @staticmethod
    async def _flags() -> dict:
        return {Optimisation.PREPARED_GREETING: 100}

    async def test_without_redis_the_cache_is_slower_never_broken(self) -> None:
        down = layer("redis://localhost:1/0")
        assert await down.get(key("release", 1), TEXT, self._value) == "loaded"
        assert await down.peek(key(), TEXT) is None
        await down.put(key(), "x", TEXT)
        await down.aclose()

    @staticmethod
    async def _value() -> str:
        return "loaded"


class TestConfigCatalog:
    def test_the_whole_configuration_survives_the_store(self) -> None:
        config = ConfigFactory.build().model_copy(
            update={
                "routes": (
                    RouteSpec(
                        work_item_type="callback",
                        description="Call back.",
                        queue="q",
                        priority=Priority.HIGH,
                    ),
                ),
                "detection": DetectionSpec(
                    thresholds={FlagType.ABUSE: 0.7},
                    disabled=frozenset({FlagType.OFF_TOPIC}),
                    severe_abuse=0.95,
                ),
                "optimisations": {Optimisation.PREPARED_GREETING: 50},
            }
        )
        adapter = ReleaseCatalog.CONFIG
        assert adapter.validate_json(adapter.dump_json(config)) == config


class TestOptimisationFlags:
    def test_a_call_always_gets_the_same_answer(self) -> None:
        percentages = {Optimisation.PREPARED_GREETING: 50}
        call = uuid.uuid4()
        assert OptimisationFlags.for_call(percentages, call) == OptimisationFlags.for_call(
            percentages, call
        )

    def test_a_percentage_splits_the_calls(self) -> None:
        percentages = {Optimisation.PREPARED_GREETING: 50, Optimisation.PREEMPTIVE_GENERATION: 0}
        chosen = [OptimisationFlags.for_call(percentages, uuid.uuid4()) for _ in range(1000)]
        on = sum(Optimisation.PREPARED_GREETING in c for c in chosen)
        assert 420 < on < 580
        assert not any(Optimisation.PREEMPTIVE_GENERATION in c for c in chosen)


class FakeTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str):
        self.texts.append(text)
        for frame in ("frame-1", "frame-2"):
            yield SimpleNamespace(frame=frame)


class TestGreeting:
    async def test_the_greeting_is_written_guarded_then_synthesised(self) -> None:
        state = AgentRuntimeFactory.state(PackFactory.build(), ConfigFactory.build())
        tts = FakeTTS()
        model = FakeModel(texts=["Hello Daniel. Your repair costs £999. How can I help?"])
        greeting = GreetingPreparer(OutputGuard()).start(state, tts, model, "Greet them.")
        text = await greeting.text
        assert text == "Hello Daniel. I will confirm the exact price for you. How can I help?"
        assert await greeting.audio == ["frame-1", "frame-2"] and tts.texts == [text]
        assert model.prompts == ["Greet them."]

    async def run_enter(self, text_delay: float, audio_delay: float):
        state = AgentRuntimeFactory.state(PackFactory.build(), ConfigFactory.build())
        said, generated = [], []

        async def words():
            await asyncio.sleep(text_delay)
            return "Hello."

        async def audio():
            await asyncio.sleep(audio_delay)
            return ["frame"]

        state.greeting = PreparedGreeting(
            asyncio.create_task(words()), asyncio.create_task(audio())
        )
        await asyncio.sleep(0.01)
        session = SimpleNamespace(
            userdata=state,
            say=lambda text, audio=None: said.append((text, audio is not None)),
            generate_reply=lambda instructions: self._record(generated, instructions),
        )
        agent = SimpleNamespace(session=session, _greeting="Greet them.", _greeting_timeout_s=0.1)
        agent._prepared_text = lambda g: ConversationAgent._prepared_text(agent, g)
        await ConversationAgent.on_enter(agent)
        return said, generated, state.greeting

    async def test_ready_audio_plays_at_once(self) -> None:
        said, generated, _ = await self.run_enter(0, 0)
        assert said == [("Hello.", True)] and generated == []

    async def test_ready_words_are_spoken_live_when_the_audio_is_not_ready(self) -> None:
        said, _, greeting = await self.run_enter(0, 5)
        await asyncio.sleep(0)
        assert said == [("Hello.", False)] and greeting.audio.cancelled()

    async def test_late_words_fall_back_to_the_model(self) -> None:
        said, generated, _ = await self.run_enter(5, 5)
        assert said == [] and generated == ["Greet them."]

    @staticmethod
    async def _record(target: list, value: str) -> None:
        target.append(value)
