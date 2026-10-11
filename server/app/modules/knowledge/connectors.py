from __future__ import annotations

import asyncio
import ipaddress
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from app.modules.knowledge.models import KnowledgeDocument


@dataclass(frozen=True)
class SourceContent:
    content: str
    source_format: str


class SourceError(Exception):
    pass


class SourceConnector(ABC):
    source_type: str

    @abstractmethod
    async def fetch(self, document: KnowledgeDocument) -> SourceContent: ...


class UploadConnector(SourceConnector):
    source_type = "upload"

    async def fetch(self, document: KnowledgeDocument) -> SourceContent:
        if not document.content:
            raise SourceError(f"document '{document.key}' has no content")
        return SourceContent(document.content, document.source_format)


class UrlGuard:
    """Stops a customer's URL from reaching the platform's own network. Every
    redirect is checked again, because a public page can redirect inward."""

    SCHEMES = frozenset({"http", "https"})

    @classmethod
    async def check(cls, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in cls.SCHEMES or not parsed.hostname:
            raise SourceError(f"not an http(s) URL: {url}")
        infos = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, None)
        for info in infos:
            address = ipaddress.ip_address(info[4][0])
            if not address.is_global or address.is_multicast:
                raise SourceError(f"{parsed.hostname} resolves to a non-public address")


class UrlConnector(SourceConnector):
    source_type = "url"
    MAX_REDIRECTS = 5
    TIMEOUT_S = 10.0

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes

    async def fetch(self, document: KnowledgeDocument) -> SourceContent:
        url = document.source_url or ""
        async with httpx.AsyncClient(follow_redirects=False, timeout=self.TIMEOUT_S) as client:
            for _ in range(self.MAX_REDIRECTS + 1):
                await UrlGuard.check(url)
                response = await client.get(url)
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                response.raise_for_status()
                if len(response.content) > self._max_bytes:
                    raise SourceError(f"{url} is larger than {self._max_bytes} bytes")
                kind = "html" if "html" in response.headers.get("content-type", "") else "markdown"
                return SourceContent(response.text, kind)
        raise SourceError(f"too many redirects from {document.source_url}")


class Connectors:
    def __init__(self, max_url_bytes: int) -> None:
        self._by_type = {c.source_type: c for c in (UploadConnector(), UrlConnector(max_url_bytes))}

    def for_document(self, document: KnowledgeDocument) -> SourceConnector:
        connector = self._by_type.get(document.source_type)
        if connector is None:
            raise SourceError(f"unknown source type '{document.source_type}'")
        return connector
