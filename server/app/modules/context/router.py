from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.agent.prompt.compiler import PromptCompiler
from app.agent.prompt.section import PromptInputs
from app.agent.skills.selector import SkillSelector
from app.core.dependencies import Resources, Session
from app.modules.accounts.service import IdentityResolver
from app.modules.calls.schemas import CallContextRead
from app.modules.calls.service import CallContextService, CallService
from app.modules.context.assembler import ContextAssembler
from app.modules.releases.service import ReleaseCatalog

router = APIRouter(tags=["context"])


@router.get("/calls/{call_id}/context", response_model=CallContextRead)
async def preview_context(call_id: uuid.UUID, session: Session, resources: Resources):
    """Exactly what the agent will be given. Useful when a call goes wrong."""
    call = await CallService(session).get(call_id)
    saved = await CallContextService(session).get(call.organization_id, call_id)
    if saved is not None:
        return CallContextRead(
            source="saved",
            pack=saved.pack,
            prefix=saved.prefix,
            prefix_hash=saved.prefix_hash,
            sections=saved.sections,
            truncations=saved.truncations,
        )
    level = IdentityResolver().resolve(call.channel, call.user_id)
    settings = resources.settings
    pack = await ContextAssembler(resources.database, settings.context).assemble(call_id, level)
    catalog = ReleaseCatalog(resources.database, resources.cache, settings)
    _, config = await catalog.for_call(call.organization_id, call_id)
    skills = SkillSelector().initial(pack, config)
    compiled = PromptCompiler(settings.prompt).compile(
        PromptInputs(pack=pack, skills=skills, collections=config.collections)
    )
    return CallContextRead(
        source="live",
        pack=pack.model_dump(mode="json"),
        prefix=compiled.prefix,
        prefix_hash=compiled.prefix_hash,
        sections=[report.as_dict() for report in compiled.sections],
        truncations=compiled.truncations,
    )

