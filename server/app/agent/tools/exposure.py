from __future__ import annotations

from collections.abc import Iterable

from app.modules.config.schemas import AgentConfig


class ToolExposure:
    """Which tools the model sees. The tool definitions are part of the cached
    prefix, so the set should not change during a call.

    Up to STATIC_LIMIT tools: all of them, for the whole call.
    More than that: only the tools of the pinned and loaded skills, plus the
    tools every call needs. A skill load can then change the set once.

    When every collection is small enough to be in the prompt, search is not
    exposed at all: there is nothing to search that the model does not have."""

    STATIC_LIMIT = 10
    ALWAYS = frozenset({"end_call", "search_knowledge", "get_call_history"})
    SEARCH = "search_knowledge"

    def is_dynamic(self, config: AgentConfig) -> bool:
        return len(config.tools) > self.STATIC_LIMIT

    def keys(self, config: AgentConfig, skill_keys: Iterable[str]) -> list[str]:
        keys = self._keys(config, skill_keys)
        if config.collections and all(c.injectable for c in config.collections):
            keys = [key for key in keys if key != self.SEARCH]
        return keys

    def _keys(self, config: AgentConfig, skill_keys: Iterable[str]) -> list[str]:
        if not self.is_dynamic(config):
            return [binding.key for binding in config.tools]
        wanted = set(self.ALWAYS)
        skills = {skill.key: skill for skill in config.skills}
        for key in skill_keys:
            if key in skills:
                wanted.update(skills[key].tool_keys)
        return [binding.key for binding in config.tools if binding.key in wanted]
