from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from markdownify import markdownify

from app.core.tokens import TokenCounter


class MarkdownNormaliser:
    """Every source becomes Markdown with ATX headings and clean spacing.

    Page furniture is removed with its contents before conversion. markdownify's
    own `strip` option keeps the text inside a removed tag, which would put a
    page's JavaScript into the knowledge base."""

    DROP_TAGS = ("script", "style", "nav", "footer", "header", "form", "noscript")

    @classmethod
    def normalise(cls, content: str, source_format: str) -> str:
        if source_format == "html":
            soup = BeautifulSoup(content, "html.parser")
            for element in soup(cls.DROP_TAGS):
                element.decompose()
            content = markdownify(str(soup), heading_style="ATX")
        text = content.replace("\r\n", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    @staticmethod
    def digest(markdown: str) -> str:
        return hashlib.sha256(markdown.encode()).hexdigest()


@dataclass(frozen=True)
class LintWarning:
    line: int
    kind: str
    text: str


class ContentLinter:
    """Finds facts that change from day to day. Those belong in a tool, not in
    the knowledge base, because a document cannot know about today."""

    RULES = (
        ("price", re.compile(r"[£$€]\s?\d")),
        ("time", re.compile(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b", re.I)),
        (
            "time",
            re.compile(r"\b(from|between)\s+\w+\s+(to|and|until)\s+\w+\s*(am|pm|o'clock)?\b", re.I),
        ),
        (
            "availability",
            re.compile(r"\b(in stock|out of stock|available today|availability)\b", re.I),
        ),
    )

    @classmethod
    def lint(cls, markdown: str) -> list[LintWarning]:
        warnings = []
        for number, line in enumerate(markdown.splitlines(), start=1):
            for kind, pattern in cls.RULES:
                if pattern.search(line):
                    warnings.append(LintWarning(number, kind, line.strip()[:160]))
                    break
        return warnings


@dataclass(frozen=True)
class ChunkPlan:
    position: int
    title: str
    heading: str
    body: str

    @property
    def text(self) -> str:
        return f"# {self.title}\n## {self.heading}\n\n{self.body}"


class HeadingSplitter:
    """Splits on H2 headings first, then paragraphs, then sentences. A numbered
    list is never split, and every chunk carries its H1 and H2."""

    SENTENCE = re.compile(r"(?<=[.!?])\s+")
    LIST_ITEM = re.compile(r"^\s*(\d+[.)]|[-*])\s+")

    def __init__(self, counter: TokenCounter, max_tokens: int, overlap_ratio: float) -> None:
        self._counter = counter
        self._max = max_tokens
        self._overlap = overlap_ratio

    def split(self, markdown: str, fallback_title: str) -> list[ChunkPlan]:
        title, sections = self._sections(markdown, fallback_title)
        plans: list[ChunkPlan] = []
        for heading, body in sections:
            for piece in self._pieces(body):
                plans.append(ChunkPlan(len(plans), title, heading, piece))
        return plans

    def _sections(self, markdown: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
        title, heading, lines, sections = fallback_title, None, [], []
        for line in markdown.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                continue
            if line.startswith("## "):
                if heading is not None or "".join(lines).strip():
                    sections.append((heading or title, "\n".join(lines).strip()))
                heading, lines = line[3:].strip(), []
                continue
            lines.append(line)
        if heading is not None or "".join(lines).strip():
            sections.append((heading or title, "\n".join(lines).strip()))
        return title, [(h, b) for h, b in sections if b]

    def _pieces(self, body: str) -> list[str]:
        if self._counter.count(body) <= self._max:
            return [body]
        pieces, current = [], []
        for unit in self._units(body):
            candidate = "\n\n".join([*current, unit])
            if current and self._counter.count(candidate) > self._max:
                pieces.append("\n\n".join(current))
                current = [self._tail(current[-1]), unit] if self._overlap else [unit]
            else:
                current.append(unit)
        if current:
            pieces.append("\n\n".join(current))
        return pieces

    def _units(self, body: str) -> list[str]:
        """Paragraphs, with each numbered or bulleted list kept whole. A plain
        paragraph that is still too big is split into sentences."""
        units, block = [], []
        for paragraph in re.split(r"\n\s*\n", body):
            is_list = all(
                self.LIST_ITEM.match(line) or not line.strip() for line in paragraph.splitlines()
            )
            if (
                is_list
                and block
                and all(self.LIST_ITEM.match(item) for item in block[-1].splitlines())
            ):
                block[-1] = f"{block[-1]}\n{paragraph}"
                continue
            block.append(paragraph)
        for unit in block:
            is_list = bool(self.LIST_ITEM.match(unit))
            if is_list or self._counter.count(unit) <= self._max:
                units.append(unit)
            else:
                units.extend(s for s in self.SENTENCE.split(unit) if s.strip())
        return units

    def _tail(self, text: str) -> str:
        budget = max(1, int(self._max * self._overlap))
        sentences = self.SENTENCE.split(text)
        tail = []
        for sentence in reversed(sentences):
            if self._counter.count(" ".join([sentence, *tail])) > budget:
                break
            tail.insert(0, sentence)
        return " ".join(tail) or sentences[-1]
