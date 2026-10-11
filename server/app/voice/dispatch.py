from __future__ import annotations

import uuid

from pydantic import BaseModel


class CallDispatch(BaseModel):
    """The only thing the API hands the worker. Everything else is loaded from it."""

    call_id: uuid.UUID
    organization_id: uuid.UUID
    user_id: uuid.UUID

    def encode(self) -> str:
        return self.model_dump_json()

    @classmethod
    def decode(cls, raw: str | None) -> CallDispatch:
        if not raw:
            raise ValueError("job dispatched without call metadata")
        return cls.model_validate_json(raw)
