from __future__ import annotations

from typing import Any

from app.agent.tools.base import ToolImplementation
from app.agent.tools.contracts import ToolContext, ToolOutcome
from app.core.enums import ToolStatus
from app.modules.knowledge.retrieval import KnowledgeRouter
from app.modules.knowledge.schemas import RetrievalResult


class KnowledgeState:
    """Retrievals of the current turn, for the turn record."""

    KEY = "knowledge_retrievals"

    @classmethod
    def add(cls, ctx: ToolContext, result: RetrievalResult) -> None:
        ctx.tool_state.setdefault(cls.KEY, []).append(
            {
                "query": result.query,
                "route": result.route,
                "card": result.card_key,
                "chunks": [str(p.chunk_id) for p in result.passages],
                "similarity": result.best_similarity,
                "ms": result.latency_ms,
            }
        )


class SearchKnowledge(ToolImplementation):
    """Answers policy and how-to questions from authored knowledge only."""

    name = "search_knowledge"
    answers_questions = True
    deterministic = True

    ONLY_THESE = (
        "Answer only from these passages. If they do not answer the question, say you "
        "do not have that information and offer a callback."
    )
    CARD = "Answer with this, in your own words, in at most two sentences: {answer}"
    NOTHING = (
        "The knowledge does not answer this. Say you do not have that information. "
        "Do not guess. Offer to have a person call back."
    )

    def __init__(self, router: KnowledgeRouter) -> None:
        self._router = router

    async def execute(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolOutcome:
        result = await self._router.search(
            ctx.organization_id,
            arguments["query"],
            ctx.knowledge_scope,
            ctx.knowledge_others,
            ctx.caller_values,
        )
        KnowledgeState.add(ctx, result)
        if result.route == "card":
            return ToolOutcome(ToolStatus.OK, self.CARD.format(answer=result.card_answer))
        if not result.passages:
            return ToolOutcome(ToolStatus.EMPTY, self.NOTHING)
        blocks = [
            f"[{i}] {p.title} — {p.heading}\n{p.text}" for i, p in enumerate(result.passages, 1)
        ]
        return ToolOutcome(ToolStatus.OK, self.ONLY_THESE + "\n\n" + "\n\n".join(blocks))
