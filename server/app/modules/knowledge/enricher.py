from __future__ import annotations

import asyncio
import re

from livekit.agents import inference
from livekit.agents.llm import ChatContext
from loguru import logger

from app.modules.knowledge.utils import ChunkPlan


class ContextualEnricher:
    """Makes each chunk self-contained before it is embedded. It runs at publish
    time, never during a call.

    Every chunk gets a fixed prefix naming its document and section. Measured on
    the demo corpus, that prefix alone gave a 100% gate pass rate; an LLM rewrite
    on top gave 87%, because the model padded chunks with filler. So LLM
    enrichment is off by default. Turn it on for sources with many pronouns and
    no headings, and let the verification gate decide.

    A rewrite is rejected, and the prefixed original used, if it loses a number
    or a code, grows too much, or repeats the document or section name."""

    INSTRUCTIONS = (
        "Rewrite the passage so it can be understood on its own. Replace pronouns and "
        "phrases like 'this' or 'as above' with what they refer to. Do not mention the "
        "document or section names; they are added separately. Do not add facts. Do not "
        "remove facts. Keep every number, name and code exactly. If the passage is already "
        "clear on its own, return it unchanged. Return only the passage."
    )
    TOKEN = re.compile(r"\b[\w-]*\d[\w-]*\b")
    TIMEOUT_S = 30.0

    def __init__(self, llm_model: str | None, concurrency: int) -> None:
        self._llm_model = llm_model
        self._semaphore = asyncio.Semaphore(concurrency)

    async def enrich(self, plans: list[ChunkPlan]) -> list[str]:
        return list(await asyncio.gather(*(self._one(plan) for plan in plans)))

    async def _one(self, plan: ChunkPlan) -> str:
        base = self.prefix(plan) + plan.body
        if not self._llm_model:
            return base
        async with self._semaphore:
            rewritten = await self._rewrite(plan)
        if rewritten is None or not self.acceptable(plan, rewritten):
            return base
        return self.prefix(plan) + rewritten

    @staticmethod
    def prefix(plan: ChunkPlan) -> str:
        return f"{plan.title}, section '{plan.heading}': "

    @classmethod
    def acceptable(cls, plan: ChunkPlan, rewritten: str) -> bool:
        keeps_facts = set(cls.TOKEN.findall(plan.body)) <= set(cls.TOKEN.findall(rewritten))
        not_padded = len(rewritten) <= 1.3 * len(plan.body) + 80
        lowered = rewritten.lower()
        no_names = plan.title.lower() not in lowered and plan.heading.lower() not in lowered
        return keeps_facts and not_padded and no_names

    async def _rewrite(self, plan: ChunkPlan) -> str | None:
        chat = ChatContext.empty()
        chat.add_message(role="system", content=self.INSTRUCTIONS)
        chat.add_message(
            role="user",
            content=f"Document: {plan.title}\nSection: {plan.heading}\n\nPassage:\n{plan.body}",
        )
        try:
            async with asyncio.timeout(self.TIMEOUT_S):
                response = await inference.LLM(model=self._llm_model).chat(chat_ctx=chat).collect()
        except Exception as error:
            logger.warning("enrichment failed for '{}': {!r}", plan.heading, error)
            return None
        return response.text.strip() or None
