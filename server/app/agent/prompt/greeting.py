from __future__ import annotations

from app.modules.context.schemas import ContextPack


class Greeting:
    def __init__(self, pack: ContextPack):
        self._pack = pack

    def instruction(self) -> str:
        # Return the greeting instructions for the caller.
        parts = [f"Greet {self._pack.caller.full_name.split()[0]} by name in one short sentence."]
        if self._pack.focus_service:
            parts.append(f"Mention this is about {self._pack.focus_service.name}.")
        if self._pack.digest and self._pack.digest.summary:
            parts.append("If the last call left something open, mention it in a few words.")
        parts.append("Then ask how you can help. Do not list anything.")
        return " ".join(parts)
