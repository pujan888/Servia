from __future__ import annotations

from livekit.agents.llm import ChatContext

from app.agent.agent import TurnContextInserter


class TestTurnContextInserter:
    def test_turn_block_goes_just_before_the_latest_caller_message(self) -> None:
        context = ChatContext.empty()
        context.add_message(role="system", content="prefix")
        context.add_message(role="user", content="first")
        context.add_message(role="assistant", content="reply")
        context.add_message(role="user", content="second")

        TurnContextInserter.insert(context, "local time block")

        texts = [item.text_content for item in context.items]
        assert texts == ["prefix", "first", "reply", "local time block", "second"]
        assert context.items[3].role == "user"

    def test_system_prompt_is_unchanged(self) -> None:
        context = ChatContext.empty()
        context.add_message(role="system", content="prefix")
        context.add_message(role="user", content="hello")
        TurnContextInserter.insert(context, "block")
        assert context.items[0].text_content == "prefix"
