"""Shared finite JSON, immutable boundary objects and safe workflow errors."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, TypeVar

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from ..graphrag.field_contracts import FrozenModel, GraphRagError

Model = TypeVar("Model", bound=BaseModel)
BriefReason = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)
]


class DecisionExplanation(FrozenModel):
    """A concise evidence-based explanation, not hidden model reasoning or approval."""

    summary: BriefReason
    evidence_refs: tuple[str, ...] = Field(max_length=24)
    uncertainties: tuple[BriefReason, ...] = Field(max_length=8)


class WorkflowError(GraphRagError):
    """Expose a stable reason code without including source-bearing payloads."""

    def __init__(
        self,
        code: str,
        ids: tuple[str, ...] = (),
        *,
        feedback: tuple[dict[str, object], ...] = (),
    ) -> None:
        super().__init__(code, ids)
        # Feedback is a separate structured contract for model correction. It is
        # deliberately omitted from the exception string and public error code.
        self.feedback = feedback


def canonical(value: object) -> str:
    """Serialize finite JSON deterministically; unsupported inputs raise ValueError."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(data: bytes) -> str:
    """Return SHA-256 of the exact supplied bytes without external state."""
    return hashlib.sha256(data).hexdigest()


def parse(model: type[Model], text: str) -> Model:
    """Reject duplicate keys/nonfinite numbers before local schema validation."""

    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        """Preserve key uniqueness instead of silently overwriting source values."""
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        """Disallow JavaScript nonfinite extensions to JSON."""
        raise ValueError("nonfinite_json")

    try:
        data = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        return model.model_validate(data)
    except ValidationError as exc:
        feedback: tuple[dict[str, object], ...] = tuple(
            {
                "path": "/" + "/".join(str(part) for part in error["loc"]),
                "code": error["type"],
                "message": error["msg"],
            }
            for error in exc.errors(
                include_input=False, include_context=False, include_url=False
            )
        )
        raise WorkflowError("invalid_structured_output", feedback=feedback) from exc
    except (ValueError, TypeError) as exc:
        raise WorkflowError(
            "invalid_structured_output",
            feedback=(
                {
                    "path": "/",
                    "code": "invalid_json",
                    "message": "Return one JSON object with unique keys and finite values, matching the supplied schema.",
                },
            ),
        ) from exc


def unique(values: tuple[str, ...], code: str) -> None:
    """Reject ambiguous object identities; do not deduplicate business objects."""
    if len(values) != len(set(values)):
        raise WorkflowError(code)


class Issue(FrozenModel):
    """A machine-checkable problem; clinical approval remains separate."""

    code: str
    refs: tuple[str, ...] = ()
