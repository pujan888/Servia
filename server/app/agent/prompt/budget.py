from __future__ import annotations

from app.core.config import PromptSettings
from app.modules.context.schemas import ContextPack


class TokenBudget:
    """A budget for how many tokens a section is allowed to use."""

    def __init__(self, settings: PromptSettings):
        self._settings = settings

    def for_section(self, budget_key: str):
        # Return the token limit for a specific section.
        return getattr(self._settings, budget_key)

    @property
    def total(self):
        # Return the total token limit for the complete prompt.
        return self._settings.total_tokens


class ContextTruncation:
    """Shrinks a context pack one fixed step at a time,Each step is recorded"""

    PROFILE_FIELDS_KEPT = 3  # if the caller profile contains too many ,keep only three of them
    TASKS_KEPT = 5

    def __init__(self, pack: ContextPack):
        self.pack = pack.model_copy(deep=True)
        self.applied: list[str] = []
        self._steps = [self._profile, self._tasks, self._digest, self._services]

    def step(self) -> bool:
        # Apply the next available truncation step.
        while self._steps:
            if self._steps.pop(0)():
                return True
        return False

    def _profile(self) -> bool:
        # Reduce the number of profile fields kept in the context.
        details = self.pack.caller.details
        if len(details) <= self.PROFILE_FIELDS_KEPT:
            return False
        kept = sorted(details)[: self.PROFILE_FIELDS_KEPT]
        self.pack.caller.details = {key: details[key] for key in kept}
        self.applied.append(f"profile_fields>{self.PROFILE_FIELDS_KEPT}")
        return True

    def _tasks(self) -> bool:
        # Reduce the number of tasks kept in the context.
        if len(self.pack.tasks) <= self.TASKS_KEPT:
            return False
        self.pack.tasks = sorted(self.pack.tasks, key=lambda t: t.updated_at, reverse=True)[
            : self.TASKS_KEPT
        ]
        self.applied.append(f"tasks>{self.TASKS_KEPT}")
        return True

    def _digest(self) -> bool:
        # Remove the digest from the context.
        if not self.pack.digest:
            return False
        self.pack.digest = None
        self.applied.append("digest")
        return True

    def _services(self) -> bool:
        # Remove the services from the context.
        if len(self.pack.services) <= 1:
            return False
        self.pack.service_count = len(self.pack.services)
        self.pack.services = []
        self.applied.append("services_to_count")
        return True


# The goal of this code is to control the prompt’s token size by setti
# ng token limits and progressively removing less-important context (p
# rofile fields, old tasks, digest, and service details) when the cont
# ext becomes too large.
