from __future__ import annotations

import uuid

from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.section import PromptInputs
from app.agent.skills.selector import SkillSelector
from app.core.config import PromptSettings
from app.core.enums import EnrollmentStatus
from app.modules.context.schemas import ServiceContext
from tests.factories import ConfigFactory, PackFactory


class TestSkillSelector:
    def setup_method(self) -> None:
        self.selector = SkillSelector()
        self.repairs = uuid.UUID(int=500)

    def pinned_keys(self, selection) -> list[str]:
        return [skill.key for skill in selection.pinned]

    def test_call_with_a_focus_service_pins_its_skill(self) -> None:
        pack = PackFactory.build()
        pack.focus_service = ServiceContext(
            id=self.repairs, key="repairs", name="Repairs", status=EnrollmentStatus.ACTIVE
        )
        config = ConfigFactory.build(ConfigFactory.skill("book_visit", service_id=self.repairs))
        selection = self.selector.initial(pack, config)
        assert self.pinned_keys(selection) == ["book_visit", "escalate_to_human"]
        assert selection.active_key == "book_visit"

    def test_call_without_a_focus_pins_general_inquiry(self) -> None:
        selection = self.selector.initial(PackFactory.build(), ConfigFactory.build())
        assert self.pinned_keys(selection) == ["general_inquiry", "escalate_to_human"]
        assert selection.active_key == "general_inquiry"

    def test_escalation_is_always_pinned(self) -> None:
        config = ConfigFactory.build(ConfigFactory.skill("check_order"))
        assert "escalate_to_human" in self.pinned_keys(
            self.selector.initial(PackFactory.build(), config)
        )

    def test_other_skills_are_listed_not_pinned(self) -> None:
        config = ConfigFactory.build(ConfigFactory.skill("check_order"))
        selection = self.selector.initial(PackFactory.build(), config)
        assert [skill.key for skill in selection.listed] == ["check_order"]

    def test_large_menus_show_only_the_callers_services(self) -> None:
        pack = PackFactory.build(services=1)
        mine = pack.services[0].id
        extra = [
            ConfigFactory.skill(f"theirs_{i}", service_id=uuid.UUID(int=900 + i)) for i in range(21)
        ]
        config = ConfigFactory.build(ConfigFactory.skill("mine", service_id=mine), *extra)
        listed = [skill.key for skill in self.selector.initial(pack, config).listed]
        assert listed == ["mine"]


class TestSkillSections:
    def test_twenty_skills_stay_inside_the_skill_budgets(self) -> None:
        skills = [ConfigFactory.skill(f"skill_{i:02d}") for i in range(20)]
        pack = PackFactory.build()
        selection = SkillSelector().initial(pack, ConfigFactory.build(*skills))
        compiled = PromptCompiler(PromptSettings()).compile(
            PromptInputs(pack=pack, skills=selection)
        )
        reports = {report.name: report for report in compiled.sections}
        assert not reports["skill_menu"].over_budget
        assert not reports["pinned_skills"].over_budget
        assert not compiled.over_total

    def test_menu_is_in_the_prefix_and_names_load_skill(self) -> None:
        pack = PackFactory.build()
        selection = SkillSelector().initial(
            pack, ConfigFactory.build(ConfigFactory.skill("check_order"))
        )
        prefix = (
            PromptCompiler(PromptSettings())
            .compile(PromptInputs(pack=pack, skills=selection))
            .prefix
        )
        assert "load_skill" in prefix
        assert "- check_order:" in prefix
        assert "Procedure: General Inquiry" in prefix
