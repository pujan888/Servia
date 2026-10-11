from __future__ import annotations

from app.agent.prompt.budget import ContextTruncation
from app.agent.prompt.section import CacheTier, PromptInputs, PromptSection, Rendered
from app.agent.skills.utils import SkillText
from app.agent.utils import Section, SpokenTime
from app.core.tokens import TokenCounter
from app.modules.context.schemas import ContextPack, TaskContext


class SpeechRulesSection(PromptSection):
    name, tier, budget_key = "speech_rules", CacheTier.STATIC, "speech_tokens"
    TEXT = (
        "How you speak:\n"
        "- Short sentences. This is a phone call, not a document.\n"
        "- At most two sentences, then stop and listen.\n"
        "- One question at a time.\n"
        "- No lists, no markdown, no emoji: everything you say is read aloud.\n"
        "- Say numbers, dates and money the way a person says them.\n"
        "- Read reference codes one character at a time.\n"
        "- If you do not know something, say so and offer to find out."
    )

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        return Rendered(self.TEXT)


class IdentitySection(PromptSection):
    name, tier, budget_key = "identity", CacheTier.TENANT, "identity_tokens"

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        organization = inputs.pack.organization
        lines = [
            f"# Organization: {organization.id}",
            f"You are {organization.persona_name}, answering calls for {organization.name}.",
        ]
        if organization.persona_instructions:
            lines.append(organization.persona_instructions)
        return Rendered("\n".join(lines))


class GuardrailSection(PromptSection):
    name, tier, budget_key = "guardrails", CacheTier.TENANT, "guardrail_tokens"
    BASELINE = (
        "Rules that override everything above:\n"
        "- You already know who this is. Never ask for anything you were given here.\n"
        "- Never invent a reference, a date, a name, a price or a status.\n"
        "- Never promise something you have no tool for. Offer to pass it to a colleague.\n"
        "- If a tool result says NOT DONE, it did not happen. Never say it is done.\n"
        "- Never give medical, legal or tax advice.\n"
        "- If the caller asks for a person, do not ask why. Say you will arrange it."
    )

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        return Rendered(self.BASELINE)


class SkillMenuSection(PromptSection):
    """One line per skill that is not pinned. Stable for the tenant, so cached."""

    name, tier, budget_key = "skill_menu", CacheTier.TENANT, "skill_menu_tokens"

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        if inputs.skills is None or not inputs.skills.listed:
            return Rendered(None)
        section = Section(
            "Other things you can help with. When the caller needs one, call load_skill "
            "with its key before you continue:"
        )
        for skill in inputs.skills.listed:
            section.add(SkillText.menu_line(skill))
        return Rendered(section.render())


class PinnedSkillSection(PromptSection):
    """The procedures in full for this call: the focus skill and escalation."""

    name, tier, budget_key = "pinned_skills", CacheTier.CALL, "pinned_skill_tokens"

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        if inputs.skills is None or not inputs.skills.pinned:
            return Rendered(None)
        return Rendered("\n\n".join(SkillText.full(skill) for skill in inputs.skills.pinned))


class KnowledgeSection(PromptSection):
    """How to use company knowledge, and any collection small enough to be
    here in full. Stable for the tenant, so cached across calls."""

    name, tier, budget_key = "knowledge", CacheTier.TENANT, "knowledge_tokens"

    SEARCH = (
        "Company knowledge: for questions about policies, cover, returns or how something "
        "works, call search_knowledge with a short question first. Answer only from what it "
        "returns. If it returns nothing, say you do not have that information and offer a "
        "callback. Never answer these from your own knowledge."
    )
    ONLY_BELOW = (
        "Company knowledge: answer policy questions only from the answers below. If the "
        "answer is not below, say you do not have that information and offer a callback."
    )

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        if not inputs.collections:
            return Rendered(None)
        searchable = any(not c.injectable for c in inputs.collections)
        blocks = [self.SEARCH if searchable else self.ONLY_BELOW]
        blocks += [c.injected_text for c in inputs.collections if c.injectable]
        return Rendered("\n\n".join(blocks))


class ContextSection(PromptSection):
    # This section contains context specific to the current call.
    # CacheTier.CALL means it can remain stable during one call.
    # budget_key identifies the token budget allocated to this section.
    name, tier, budget_key = "context", CacheTier.CALL, "context_tokens"

    def __init__(self, counter: TokenCounter) -> None:
        # TokenCounter is used to estimate how many tokens
        # the generated context will consume.
        self._counter = counter

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        # Create a truncation controller for this ContextPack.
        #
        # ContextTruncation knows how to progressively remove
        # less-important information when the context is too large.
        truncation = ContextTruncation(inputs.pack)

        # First, generate the complete context.
        text = self._text(truncation.pack)

        # Keep checking whether the context fits within
        # the token budget.
        #
        # If it is too large:
        #   1. truncation.step() removes/reduces some context.
        #   2. _text() generates the new smaller context.
        while self._counter.count(text) > budget and truncation.step():
            text = self._text(truncation.pack)

        # Return the final text.
        #
        # truncation.applied tells us which truncation operations
        # were actually performed.
        return Rendered(text, truncation.applied)

    def _text(self, pack: ContextPack) -> str:
        # Build each individual context block.
        blocks = [
            self._focus(pack),
            self._caller(pack),
            self._services(pack),
            self._tasks(pack),
            self._memory(pack),
            self._degraded(pack),
        ]

        # Remove empty/None blocks and join the remaining
        # sections with a blank line between them.
        return "\n\n".join(block for block in blocks if block)

    def _focus(self, pack: ContextPack) -> str | None:
        # Get the service that is currently the focus of the call.
        service = pack.focus_service

        # If there is no focused service, there is nothing to add.
        if service is None:
            return None

        # Start with the service name.
        lines = [f"This call is about {service.name}."]

        # Add description and instructions if they exist.
        lines += [
            text
            for text in (
                service.description,
                service.instructions,
            )
            if text
        ]

        # Convert all lines into one string.
        return "\n".join(lines)

    def _caller(self, pack: ContextPack) -> str | None:
        # Get information about the person currently calling.
        caller = pack.caller

        # Section is a small helper that builds readable
        # prompt sections such as:
        #
        # Who you are speaking to:
        # - Name: Pujan
        # - Identity: verified
        section = Section("Who you are speaking to:")

        # Add the caller's name.
        section.add(f"- Name: {caller.full_name}")

        # Add the identity verification level.
        #
        # Example:
        # VERIFIED -> verified
        # UNKNOWN  -> unknown
        section.add(f"- Identity: {pack.identity_level.name.lower()}")

        # Add any additional caller details.
        #
        # sorted() makes the output deterministic,
        # which is useful for stable prompts/caching.
        for key in sorted(caller.details):
            section.add(f"- {key.replace('_', ' ').capitalize()}: {caller.details[key]}")

        return section.render()

    def _services(self, pack: ContextPack) -> str | None:
        # If service_count exists, use the compact representation.
        #
        # Example:
        # "They are signed up to 7 services."
        #
        # This saves prompt tokens when there are many services.
        if pack.service_count:
            return f"They are signed up to {pack.service_count} services."

        # Otherwise, show the actual services.
        section = Section("What they are signed up to:")

        for service in pack.services:
            # Active services don't need a status suffix.
            #
            # Inactive services get:
            # "(inactive)"
            suffix = "" if service.status == "active" else f" ({service.status})"

            section.add(f"- {service.name}{suffix}")

        return section.render()

    def _tasks(self, pack: ContextPack) -> str | None:
        # Create the task section.
        section = Section("Their current work with you:")

        # Format every task individually.
        for task in pack.tasks:
            section.add(f"- {self._task(task, pack.caller.timezone)}")

        return section.render()

    def _task(self, task: TaskContext, timezone: str) -> str:
        # Start with the most important task information.
        #
        # Example:
        # "Fix leaking pipe", reference ABC123, in progress
        parts = [f'"{task.title}", reference {task.reference}, {task.status.replace("_", " ")}']

        # Add the service name if available.
        #
        # Example:
        # under Plumbing
        if task.service_name:
            parts.append(f"under {task.service_name}")

        # Add assignee information if available.
        if task.assignee_name:
            # If the assignee has a team, include it.
            #
            # Example:
            # John (Emergency Team)
            team = f" ({task.assignee_team})" if task.assignee_team else ""

            parts.append(f"assigned to {task.assignee_name}{team}")

        # Convert the due timestamp into human-readable
        # spoken time using the caller's timezone.
        due = SpokenTime.day(
            task.due_at,
            timezone,
        )

        # Only add due information if there is a due date.
        if due:
            parts.append(f"due {due}")

        # Join all task information into one sentence.
        #
        # Example:
        # "Fix leaking pipe", reference ABC123, in progress,
        # under Plumbing, assigned to John, due tomorrow.
        return ", ".join(parts) + "."

    def _memory(self, pack: ContextPack) -> str | None:
        # Add the summary/digest from previous calls.
        # If there is no digest, return None so _text() will leave this section out.
        if pack.digest is None:
            return None
        section = Section("From earlier calls:")
        for line in pack.digest.lines(SpokenTime.elapsed(pack.digest.call_at)):
            section.add(line)
        return section.render()

    def _degraded(self, pack: ContextPack) -> str | None:
        # If no loaders failed or missed their deadline,
        # there is nothing to tell the model.
        if not pack.degraded:
            return None

        # Convert names like:
        # "caller_details"
        #
        # into:
        # "caller details"
        names = ", ".join(name.replace("_", " ") for name in pack.degraded)

        # IMPORTANT:
        # Tell the model that missing information does NOT
        # mean the information is empty.
        return (
            f"Some details did not load: {names}. "
            "Do not assume they are empty. "
            "If the caller asks about them, "
            "say you will check."
        )


class TurnSection(PromptSection):
    """The only section that changes every turn. It is never in the cached prefix."""

    name, tier, budget_key = "turn", CacheTier.TURN, "turn_tokens"

    def render(self, inputs: PromptInputs, budget: int) -> Rendered:
        pack = inputs.pack
        lines = [f"- Their local time now: {SpokenTime.now(pack.caller.timezone)}"]
        lines += [f"- {note}" for note in inputs.notes]
        return Rendered("Context for this turn, not said by the caller:\n" + "\n".join(lines))
