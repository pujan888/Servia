from __future__ import annotations

import hashlib
from typing import Any

from jsonschema import Draft202012Validator, SchemaError


class SchemaGuard:
    """Checks a binding's schema when it is saved, and the model's arguments
    when it calls the tool."""

    # Arguments that would let the model choose whose data a tool touches.
    # Identity comes from the call state, never from the model.
    IDENTITY_FIELDS = frozenset(
        {
            "customer_id",
            "caller_id",
            "user_id",
            "account_id",
            "organization_id",
            "org_id",
            "tenant_id",
            "call_id",
            "phone",
            "phone_number",
            "email",
        }
    )

    @classmethod
    def schema_errors(cls, schema: dict[str, Any]) -> list[str]:
        errors = []
        if schema.get("type") != "object":
            errors.append("The parameters schema must have type 'object'.")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            errors.append(f"The parameters schema is not valid: {error.message}")
        identity = cls.IDENTITY_FIELDS & set(schema.get("properties", {}))
        if identity:
            errors.append(
                f"The schema must not ask the model for identity: {sorted(identity)}. "
                "The gateway gives identity to the implementation."
            )
        return errors

    @staticmethod
    def argument_errors(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
        strict = {**schema, "additionalProperties": False}
        validator = Draft202012Validator(strict)
        return [error.message for error in validator.iter_errors(arguments)]


class OptimisationFlags:
    """Which switches are on for one call. The same call always gets the same
    answer, so a call's turns are never mixed, and a tenant at 50% splits its
    calls in half for the latency report."""

    @staticmethod
    def for_call(percentages: dict, call_id: object) -> frozenset:
        chosen = set()
        for name, percent in percentages.items():
            bucket = int(hashlib.sha256(f"{call_id}:{name}".encode()).hexdigest(), 16) % 100
            if bucket < percent:
                chosen.add(name)
        return frozenset(chosen)
