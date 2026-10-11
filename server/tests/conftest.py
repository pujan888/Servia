from __future__ import annotations

import uuid

import pytest

from app.agent.tools.registry import PlatformTools
from app.core.cache import CacheLayer, MemoryCache, RedisStore
from app.core.config import SettingsFactory
from app.core.database import Database
from app.core.enums import TaskStatus
from app.modules.accounts.models import Organization, User
from app.modules.calls.models import Call
from app.modules.calls.utils import RoomName
from app.modules.knowledge.catalog import CardCatalog
from app.modules.knowledge.embedder import Embedder
from app.modules.knowledge.retrieval import KnowledgeRouter
from app.modules.work.factory import TableRouter, WorkItemFactory
from app.modules.work.models import Task
from tests.fakes import FakeModel


class TenantFixture:
    """Two tenants with deliberately similar data: same customer name, same
    reference format. Isolation tests use it to catch cross-tenant reads."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.organizations: list[Organization] = []
        self.calls: dict[str, Call] = {}

    async def create(self, label: str, task_title: str) -> Call:
        async with self.database.session() as session:
            organization = Organization(slug=f"test-{uuid.uuid4().hex[:8]}", name=label)
            session.add(organization)
            await session.flush()
            user = User(organization_id=organization.id, full_name="Sam Taylor", profile={})
            session.add(user)
            await session.flush()
            session.add(
                Task(
                    organization_id=organization.id,
                    reference="KQ7M3A",
                    user_id=user.id,
                    title=task_title,
                    status=TaskStatus.OPEN,
                )
            )
            call = Call(
                organization_id=organization.id, user_id=user.id, room_name=RoomName.generate()
            )
            session.add(call)
            await session.flush()
        self.organizations.append(organization)
        self.calls[label] = call
        return call

    async def cleanup(self) -> None:
        async with self.database.session() as session:
            for organization in self.organizations:
                await session.delete(await session.get(Organization, organization.id))


@pytest.fixture
async def database():
    database = Database(SettingsFactory.load().database)
    try:
        await database.warm(1)
    except OSError as error:
        await database.dispose()
        pytest.skip(f"postgres is not reachable (run `make up && make migrate`): {error}")
    yield database
    await database.dispose()


@pytest.fixture
async def tenants(database):
    fixture = TenantFixture(database)
    yield fixture
    await fixture.cleanup()


@pytest.fixture(scope="session")
def embedder():
    embedder = Embedder(SettingsFactory.load().knowledge)
    embedder.load()
    return embedder


@pytest.fixture
async def cache():
    layer = CacheLayer(MemoryCache(), RedisStore(SettingsFactory.load().cache.url))
    yield layer
    await layer.aclose()


@pytest.fixture
def knowledge_router(database, embedder, cache):
    settings = SettingsFactory.load().knowledge
    return KnowledgeRouter(database, embedder, CardCatalog(database), cache, settings)


@pytest.fixture
def registry(database, knowledge_router, embedder):
    settings = SettingsFactory.load()
    summary = {"title": "Callback about the boiler", "summary": "Call back.", "confidence": 0.9}
    work_items = WorkItemFactory(
        database,
        embedder,
        FakeModel(structured=[summary], repeat=True),
        TableRouter(database),
        settings.work,
        settings.detection,
    )
    return PlatformTools.registry(database, knowledge_router, settings.memory, work_items)
