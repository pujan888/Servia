from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.core.enums import EnrollmentStatus, IdentityLevel, TaskStatus
from app.modules.context.schemas import (
    CallerContext,
    ContextPack,
    OrganizationContext,
    ServiceContext,
    TaskContext,
)


class PackFactory:
    @staticmethod
    def build(tasks: int = 2, profile_fields: int = 2, services: int = 2) -> ContextPack:
        now = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
        return ContextPack(
            call_id=uuid.UUID(int=1),
            identity_level=IdentityLevel.VERIFIED,
            organization=OrganizationContext(
                id=uuid.UUID(int=2),
                name="Northwind",
                timezone="Europe/London",
                persona_name="Alex",
                persona_instructions="Northwind sells appliances and repairs them.",
            ),
            caller=CallerContext(
                id=uuid.UUID(int=3),
                full_name="Daniel Whitfield",
                locale="en-GB",
                timezone="Europe/London",
                details={f"field_{i:02d}": f"value {i}" for i in range(profile_fields)},
            ),
            services=[
                ServiceContext(
                    id=uuid.UUID(int=100 + i),
                    key=f"service-{i}",
                    name=f"Service {i}",
                    status=EnrollmentStatus.ACTIVE,
                )
                for i in range(services)
            ],
            tasks=[
                TaskContext(
                    reference=f"REF{i:03d}",
                    title=f"A fairly long title describing open job number {i} in detail",
                    status=TaskStatus.OPEN,
                    service_name="Appliance Repair",
                    assignee_name="Sam Okoye",
                    assignee_team="Field Engineers",
                    updated_at=now - timedelta(hours=i),
                )
                for i in range(tasks)
            ],
        )


class ConfigFactory:
    @staticmethod
    def skill(key: str, trigger: str | None = None, service_id=None, system: bool = False):
        from app.modules.config.schemas import SkillSpec

        return SkillSpec(
            key=key,
            name=key.replace("_", " ").title(),
            trigger=trigger or f"When the caller needs {key.replace('_', ' ')} and nothing else.",
            goal=f"The {key} is done.",
            steps="1. Do the first step.\n2. Do the second step.",
            guardrails=None,
            done_when=None,
            escalate_if=None,
            knowledge_key=None,
            tool_keys=(),
            service_id=service_id,
            system=system,
        )

    @classmethod
    def build(cls, *extra, version: int = 1, tools=()):
        from app.modules.config.schemas import AgentConfig, ProfileSpec

        skills = (
            cls.skill("escalate_to_human", system=True),
            cls.skill("general_inquiry", system=True),
            *extra,
        )
        return AgentConfig(
            organization_id=uuid.UUID(int=2),
            version=version,
            skills=skills,
            profile=ProfileSpec(),
            tools=tuple(tools),
        )


class AgentRuntimeFactory:
    @staticmethod
    def state(pack, config):
        from app.agent.prompt.compiler import PromptCompiler
        from app.agent.prompt.section import PromptInputs
        from app.agent.skills.selector import SkillSelector
        from app.agent.state import CallState
        from app.core.config import PromptSettings

        skills = SkillSelector().initial(pack, config)
        compiled = PromptCompiler(PromptSettings()).compile(PromptInputs(pack=pack, skills=skills))
        return CallState(
            pack=pack,
            config=config,
            skills=skills,
            compiled=compiled,
            active_skill=skills.active_key,
        )


class ToolFactory:
    @staticmethod
    def binding(key: str, implementation: str | None = None, **overrides):
        from app.core.enums import IdentityLevel, LatencyClass
        from app.modules.config.schemas import BindingSpec

        values = {
            "key": key,
            "implementation": implementation or key,
            "description": f"Use {key} when the caller needs it.",
            "parameters": {"type": "object", "properties": {}},
            "latency_class": LatencyClass.A,
            "required_level": IdentityLevel.UNKNOWN,
            "timeout_ms": 500,
            "max_calls": 5,
            "filler_category": None,
            "fallback_text": f"{key} is not available. Offer a callback.",
            "cache_ttl_s": 0,
            "invalidates": (),
        }
        values.update(overrides)
        return BindingSpec(**values)
