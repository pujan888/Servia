from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.core.dependencies import Resources, Session
from app.modules.config.service import AgentConfigService
from app.modules.knowledge.schemas import (
    CardRead,
    CardWrite,
    CollectionRead,
    CollectionWrite,
    DocumentRead,
    DocumentWrite,
    PublishRead,
    RetrievalResult,
    RetrievalTestRead,
    RetrievalTestWrite,
    SearchRequest,
)
from app.modules.knowledge.service import (
    CardService,
    CollectionService,
    DocumentService,
    PublishService,
    RetrievalTestService,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/collections", response_model=list[CollectionRead])
async def list_collections(organization_id: uuid.UUID, session: Session):
    return await CollectionService(session).list(organization_id)


@router.post("/collections", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def save_collection(organization_id: uuid.UUID, data: CollectionWrite, session: Session):
    return await CollectionService(session).save(organization_id, data)


@router.post("/collections/{key}/documents", response_model=DocumentRead, status_code=201)
async def save_document(
    organization_id: uuid.UUID, key: str, data: DocumentWrite, session: Session
):
    collection = await CollectionService(session).get(organization_id, key)
    return await DocumentService(session).save(collection, data)


@router.get("/collections/{key}/documents", response_model=list[DocumentRead])
async def list_documents(organization_id: uuid.UUID, key: str, session: Session):
    collection = await CollectionService(session).get(organization_id, key)
    return await DocumentService(session).list(collection)


@router.post("/collections/{key}/cards", response_model=CardRead, status_code=201)
async def save_card(organization_id: uuid.UUID, key: str, data: CardWrite, session: Session):
    collection = await CollectionService(session).get(organization_id, key)
    return await CardService(session).save(collection, data)


@router.get("/collections/{key}/cards", response_model=list[CardRead])
async def list_cards(organization_id: uuid.UUID, key: str, session: Session):
    collection = await CollectionService(session).get(organization_id, key)
    return await CardService(session).list(collection)


@router.post("/collections/{key}/tests", response_model=RetrievalTestRead, status_code=201)
async def add_test(
    organization_id: uuid.UUID, key: str, data: RetrievalTestWrite, session: Session
):
    collection = await CollectionService(session).get(organization_id, key)
    return await RetrievalTestService(session).add(collection, data)


@router.get("/collections/{key}/tests", response_model=list[RetrievalTestRead])
async def list_tests(organization_id: uuid.UUID, key: str, session: Session):
    collection = await CollectionService(session).get(organization_id, key)
    return await RetrievalTestService(session).list(collection)


@router.post("/collections/{key}/publish", response_model=PublishRead, status_code=202)
async def publish(organization_id: uuid.UUID, key: str, session: Session):
    collection = await CollectionService(session).get(organization_id, key)
    return await PublishService(session).enqueue(collection)


@router.get("/publishes/{publish_id}", response_model=PublishRead)
async def get_publish(organization_id: uuid.UUID, publish_id: uuid.UUID, session: Session):
    return await PublishService(session).get(organization_id, publish_id)


@router.post("/search", response_model=RetrievalResult)
async def search(data: SearchRequest, session: Session, resources: Resources):
    """A debug search: the same ladder the agent uses, with the route and scores."""
    config = await AgentConfigService(session).load(data.organization_id)
    chosen = config.collection(data.collection_key)
    scope = [chosen] if chosen else list(config.default_collections)
    others = [c for c in config.collections if c not in scope]
    return await resources.knowledge_router.search(
        data.organization_id, data.query, scope, others, data.filters, use_cache=False
    )
