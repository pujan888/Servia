from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CollectionWrite(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    name: str
    is_default: bool = False
    filter_keys: list[str] = Field(default_factory=list)


class CollectionRead(CollectionWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    active_version: int | None


class DocumentWrite(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    title: str
    source_type: str = Field(pattern="^(upload|url)$")
    source_format: str = Field(default="markdown", pattern="^(markdown|html|text)$")
    source_url: str | None = None
    content: str | None = None
    kind: str = Field(default="article", pattern="^(article|faq)$")
    metadata: dict[str, str] = Field(default_factory=dict)
    last_verified: date | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def source_matches(self) -> DocumentWrite:
        if self.source_type == "upload" and not self.content:
            raise ValueError("an upload needs content")
        if self.source_type == "url" and not self.source_url:
            raise ValueError("a URL document needs source_url")
        return self


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    key: str
    title: str
    source_type: str
    source_format: str
    source_url: str | None
    kind: str
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    last_verified: date | None
    enabled: bool


class CardWrite(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    questions: list[str] = Field(min_length=3, max_length=8)
    answer: str = Field(min_length=10)
    enabled: bool = True


class CardRead(CardWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID


class RetrievalTestWrite(BaseModel):
    question: str
    expected_heading: str | None = None
    expected_card_key: str | None = None
    expect_refusal: bool = False
    filters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_expectation(self) -> RetrievalTestWrite:
        given = [bool(self.expected_heading), bool(self.expected_card_key), self.expect_refusal]
        if sum(given) != 1:
            raise ValueError(
                "give exactly one of expected_heading, expected_card_key, expect_refusal"
            )
        return self


class RetrievalTestRead(RetrievalTestWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID


class PublishRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    status: str
    report: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None


class SearchRequest(BaseModel):
    organization_id: uuid.UUID
    query: str
    collection_key: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class Passage(BaseModel):
    title: str
    heading: str
    text: str
    similarity: float
    chunk_id: uuid.UUID


class RetrievalResult(BaseModel):
    route: str
    passages: list[Passage] = Field(default_factory=list)
    card_key: str | None = None
    card_answer: str | None = None
    best_similarity: float | None = None
    latency_ms: int = 0
    query: str = ""

    @property
    def top_heading(self) -> str | None:
        return self.passages[0].heading if self.passages else None
