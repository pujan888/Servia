from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.models import Base, Identified, OrganizationScoped, Timestamped

EMBEDDING_DIM = 384


class KnowledgeCollection(Base, Identified, Timestamped, OrganizationScoped):
    """A named group of documents and answer cards. Only one version is live."""

    __tablename__ = "knowledge_collections"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_collection_org_key"),)

    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    is_default: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    active_version: Mapped[int | None] = mapped_column(Integer())
    filter_keys: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default="{}"
    )


class KnowledgeDocument(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "knowledge_documents"
    __table_args__ = (UniqueConstraint("collection_id", "key", name="uq_document_collection_key"),)

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(300))
    source_type: Mapped[str] = mapped_column(String(16))
    source_format: Mapped[str] = mapped_column(String(16), default="markdown")
    source_url: Mapped[str | None] = mapped_column(Text())
    content: Mapped[str | None] = mapped_column(Text())
    kind: Mapped[str] = mapped_column(String(16), default="article", server_default="article")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )
    last_verified: Mapped[date | None] = mapped_column(Date())
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


class AnswerCard(Base, Identified, Timestamped, OrganizationScoped):
    """A curated answer to one frequent question, with the ways people ask it."""

    __tablename__ = "answer_cards"
    __table_args__ = (UniqueConstraint("collection_id", "key", name="uq_card_collection_key"),)

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(64))
    questions: Mapped[list[str]] = mapped_column(ARRAY(Text()))
    answer: Mapped[str] = mapped_column(Text())
    enabled: Mapped[bool] = mapped_column(Boolean(), default=True, server_default="true")


class KnowledgeChunk(Base, Identified, OrganizationScoped):
    """One retrievable unit of one published version."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("ix_chunks_scope", "organization_id", "collection_id", "version"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column(Integer())
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_documents.id", ondelete="CASCADE")
    )
    document_hash: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer())
    title: Mapped[str] = mapped_column(String(300))
    heading: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text())
    enriched_text: Mapped[str] = mapped_column(Text())
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(String(100))
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', enriched_text)", persisted=True)
    )


class CardVariant(Base, Identified, OrganizationScoped):
    """One way of asking one card's question, in one published version."""

    __tablename__ = "card_variants"
    __table_args__ = (
        Index("ix_card_variants_scope", "organization_id", "collection_id", "version"),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column(Integer())
    card_key: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text())
    answer: Mapped[str] = mapped_column(Text())
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class RetrievalTest(Base, Identified, Timestamped, OrganizationScoped):
    """A real caller question, and what retrieval must return first."""

    __tablename__ = "retrieval_tests"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text())
    expected_heading: Mapped[str | None] = mapped_column(String(300))
    expected_card_key: Mapped[str | None] = mapped_column(String(64))
    expect_refusal: Mapped[bool] = mapped_column(Boolean(), default=False, server_default="false")
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")


class KnowledgePublish(Base, Identified, Timestamped, OrganizationScoped):
    __tablename__ = "knowledge_publishes"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_collections.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer())
    status: Mapped[str] = mapped_column(String(16), default="pending")
    report: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
