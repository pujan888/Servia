from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from app.agent.prompt.budget import TokenBudget
from app.agent.prompt.section import CacheTier, PromptInputs, PromptSection
from app.agent.prompt.sections import (
    ContextSection,
    GuardrailSection,
    IdentitySection,
    KnowledgeSection,
    PinnedSkillSection,
    SkillMenuSection,
    SpeechRulesSection,
    TurnSection,
)
from app.core.config import PromptSettings
from app.core.tokens import TokenCounter


@dataclass(slots=True)
class SectionReport:
    name: str
    tier: str
    tokens: int
    budget: int

    @property
    def over_budget(self) -> bool:
        return self.tokens > self.budget

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "tokens": self.tokens,
            "budget": self.budget,
            "over_budget": self.over_budget,
        }


@dataclass(slots=True)
class CompiledPrompt:
    """The result of compiling a prompt from its sections"""

    prefix: str
    prefix_hash: str
    sections: list[SectionReport] = field(default_factory=list)
    truncations: list[str] = field(default_factory=list)
    budget_total: int = 0

    @property
    def over_total(self) -> bool:
        return self.total_tokens > self.budget_total

    @property
    def total_tokens(self) -> int:
        return sum(section.tokens for section in self.sections)


@dataclass(slots=True)
class TurnSuffix:
    text: str
    tokens: int
    elapsed_ms: int


class PromptCompiler:
    """Builds the prompt in tier order. Everything above the TURN tier is the
    prefix, which must stay byte-identical for the whole call."""

    SEPARATOR = "\n\n"

    def __init__(
        self, settings: PromptSettings, sections: list[PromptSection] | None = None
    ) -> None:
        self._counter = TokenCounter(settings.token_margin)
        self._budget = TokenBudget(settings)
        self._sections = sorted(
            sections or self.default_sections(self._counter), key=lambda s: s.tier
        )

    @staticmethod
    def default_sections(counter: TokenCounter) -> list[PromptSection]:
        return [
            SpeechRulesSection(),
            IdentitySection(),
            GuardrailSection(),
            SkillMenuSection(),
            KnowledgeSection(),
            ContextSection(counter),
            PinnedSkillSection(),
            TurnSection(),
        ]

    def fingerprint(self) -> str:
        """Changes when the platform's own prompt text changes, so a release
        records which prompt code its configuration was made for."""
        texts = [
            f"{type(section).__name__}.{name}={value}"
            for section in self._sections
            for name, value in sorted(vars(type(section)).items())
            if name.isupper() and isinstance(value, str)
        ]
        return hashlib.sha256("\n".join(texts).encode()).hexdigest()

    def compile(self, inputs: PromptInputs) -> CompiledPrompt:
        texts, reports, truncations = [], [], []
        for section in self._sections:
            if section.tier == CacheTier.TURN:
                continue
            budget = self._budget.for_section(section.budget_key)
            rendered = section.render(inputs, budget)
            truncations += rendered.truncations
            if rendered.text:
                texts.append(rendered.text)
            reports.append(
                SectionReport(
                    section.name, section.tier.name, self._counter.count(rendered.text), budget
                )
            )
        prefix = self.SEPARATOR.join(texts)
        return CompiledPrompt(
            prefix=prefix,
            prefix_hash=hashlib.sha256(
                prefix.encode()
            ).hexdigest(),  # The hash gives you a cheap way to identify whether the prefix changed.
            sections=reports,
            truncations=truncations,
            budget_total=self._budget.total,
        )

    def turn_suffix(self, inputs: PromptInputs) -> TurnSuffix:
        # It renders only the TURN sections, and measures how long that
        started = time.perf_counter()
        texts = []
        for section in self._sections:
            if section.tier != CacheTier.TURN:
                continue
            rendered = section.render(inputs, self._budget.for_section(section.budget_key))
            if rendered.text:
                texts.append(rendered.text)
        text = self.SEPARATOR.join(texts)
        return TurnSuffix(
            text=text,
            tokens=self._counter.count(text),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


# ---------------------------------------------------------
# BIG PICTURE:
#
# compile() and turn_suffix() divide the prompt into two
# parts:
#
# 1. compile()
#    → handles everything that is NOT specific to the
#      current conversation turn.
#    → skips sections whose tier is TURN.
#    → renders the other sections.
#    → combines their text into one stable "prefix".
#    → also records token usage, truncations, section reports,
#      and a hash of the final prefix.
#
# 2. turn_suffix()
#    → handles ONLY sections whose tier is TURN.
#    → skips all non-TURN sections.
#    → renders the current-turn information.
#    → combines it into one "suffix".
#    → returns the suffix text, token count, and build time.
#
# Example:
#
#    IdentitySection       → STATIC
#    ContextSection        → STATIC
#    GuardrailSection      → STATIC
#    SpeechRulesSection    → STATIC
#    TurnSection           → TURN
#
# compile() produces:
#
#    "You are Servia.
#     Caller is Pujan.
#     Follow these safety rules.
#     Speak naturally."
#
# turn_suffix() produces:
#
#    "Current request: I need a plumber.
#     Current conversation state: ..."
#
# The system can then conceptually give the LLM:
#
#    stable prefix
#          +
#    current-turn suffix
#          +
#    current user message
#
# The important condition is:
#
#    if section.tier == TURN:
#         compile() SKIPS it
#
#    if section.tier != TURN:
#         turn_suffix() SKIPS it
#
# Therefore the two functions split the prompt into:
#
#    STATIC / stable information
#              +
#    TURN / changing information
#
# This is useful because the stable part does not need to be
# rebuilt as the conversation changes, while the TURN part can
# be rebuilt for every new user turn.
# ---------------------------------------------------------

# I ran the real `PromptCompiler.compile()` on sample data from your seed: Northwind, with caller
# Daniel Whitfield. Here's what came back, field by field.

# ### `prefix`: the full system prompt text
# Sections 1 to 4, joined by blank lines:
# ```
# How you speak:
# - Short sentences. This is a phone call, not a document.
# - ...
# - If you do not know something, say so and offer to find out.

# # Organization: 11111111-1111-1111-1111-111111111111
# You are Alex, answering calls for Northwind.
# Northwind sells home appliances and runs the repair service for them. ...

# Rules that override everything above:
# - You already know who this is. Never ask for anything you were given here.
# - ...

# This call is about Appliance Repair.
# Engineer visits for appliances bought from Northwind.
# Visits are two-hour windows. ...

# Who you are speaking to:
# - Name: Daniel Whitfield
# - Identity: recognised
# - Address: 14 Rosemount Road, Manchester M20 3TF
# - Customer since: 2023
# - Preferred contact: phone

# What they are signed up to:
# - Appliance Repair
# - Northwind Care Plan

# Their current work with you:
# - "Washing machine leaking from the door seal", reference NW-4821, in progress, under Appliance
#   Repair, assigned to Sam Okoye (Field Engineers), due this Saturday.
# - "Care Plan claim for the oven fan", reference NW-4822, blocked, under Northwind Care Plan,
#   assigned to Priya Raman (Claims).
# ```

# ### `prefix_hash`: a fingerprint of that text
# ```
# a393d725f711dfef362a8789f2bd686226e99efc38b94b1b53be667e78f06967
# ```
# This is the SHA-256 of `prefix`. If even one character changes, the hash is completely different.
# Each turn's metrics record it, so you can confirm the prefix stayed the same during the call.

# ### `sections`: one report per section
# ```python
# {'name': 'speech_rules', 'tier': 'STATIC', 'tokens': 106, 'budget': 150, 'over_budget': False}
# {'name': 'identity',     'tier': 'TENANT', 'tokens': 75,  'budget': 150, 'over_budget': False}
# {'name': 'guardrails',   'tier': 'TENANT', 'tokens': 104, 'budget': 300, 'over_budget': False}
# {'name': 'context',      'tier': 'CALL',   'tokens': 214, 'budget': 600, 'over_budget': False}
# ```
# There's no `turn` entry because `compile()` skips the TURN section.

# ### `truncations`: what was cut to fit
# ```python
# []
# ```
# It's empty because everything fit. To show a non-empty result, I ran it again with the `context`
# budget lowered from 600 to 60 tokens:
# ```python
# truncations = ['services_to_count']
# # the services block became:  "They are signed up to 2 services."
# {'name': 'context', 'tokens': 205, 'budget': 60, 'over_budget': True}
# ```
# The possible truncation steps, tried in this order, are:
# - `profile_fields>3`: keep only 3 profile fields
# - `tasks>5`: keep only the 5 most recent tasks
# - `digest`: drop the memory from earlier calls
# - `services_to_count`: replace the services list with a count

# A step is skipped if it wouldn't change anything. In this run the first three had nothing to cut,
# so only the services step was applied.

# ### `budget_total`: the overall token limit
# ```
# 2750
# ```
# This comes from `PromptSettings.total_tokens`. The result also has two computed properties:
# - `total_tokens`: 499, the sum of the section token counts
# - `over_total`: `False`, because 499 is under 2750

# ### One thing the test showed
# Even after truncation, the context section was still 205 tokens against a budget of 60
# (`over_budget: True`). Once all the steps are used up, the loop stops and the too-long text is
# kept anyway. Tasks are only cut when there are **more than 5**, so the task list itself is never
# shortened below 5 tasks. Because of this, a budget is a target, not a hard limit. When it's
# exceeded, `_prepare` logs a warning ("prompt over budget...") and continues. That's the intended
# behavior as long as a warning is enough for you. If you need a hard limit, the truncation loop
# would need more steps.

# The demo script is at `scratchpad/demo_compile.py` if you want to change the data and run it
# yourself. Use `PYTHONPATH=. .venv/bin/python <path>` from `servia/`.
