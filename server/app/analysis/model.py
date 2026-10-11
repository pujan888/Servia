from __future__ import annotations

import asyncio

from livekit.agents import inference
from livekit.agents.llm import ChatContext
from pydantic import BaseModel


class AnalysisModel:
    """The language model for work after a call. Nothing here is on a call's
    path, so a failure raises and the job tries again later."""

    def __init__(self, model: str, timeout_s: float) -> None:
        self.name = model
        self._timeout_s = timeout_s

    async def text(self, instructions: str, content: str) -> str:
        return (await self._complete(instructions, content, None)).strip()

    async def structured[T: BaseModel](self, instructions: str, content: str, schema: type[T]) -> T:
        return schema.model_validate_json(await self._complete(instructions, content, schema))

    async def _complete(
        self, instructions: str, content: str, schema: type[BaseModel] | None
    ) -> str:
        chat = ChatContext.empty()
        chat.add_message(role="system", content=instructions)
        chat.add_message(role="user", content=content)
        options = {"response_format": schema} if schema else {}
        async with asyncio.timeout(self._timeout_s):
            response = await inference.LLM(model=self.name).chat(chat_ctx=chat, **options).collect()
        return response.text
