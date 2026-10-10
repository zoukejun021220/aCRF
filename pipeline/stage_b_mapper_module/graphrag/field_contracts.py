"""Immutable runtime contracts for reviewed CRF fields and study profiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from .contracts import GraphError
from .contracts import StandardProfile as GraphStandardProfile

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
InputType = Literal["text", "number", "date", "radio", "checkbox", "mixed", "unknown"]
ANSWER_PATH_PARTS = frozenset(
    {
        "annotation",
        "annotations",
        "annotated_output",
        "answer",
        "answers",
        "gold",
        "ground_truth",
        "held_out_answers",
        "reference",
        "reference_with_sections",
    }
)


class FrozenModel(BaseModel):
    """Reject unknown properties and prevent mutation after boundary validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class GraphRagError(GraphError):
    """Carry a stable error code and safe record IDs without source text or paths."""

    def __init__(self, code: str, ids: tuple[str, ...] = ()) -> None:
        """Store a code and sanitized IDs; omit unsafe values from the exception text."""
        self.ids = tuple(
            identifier for identifier in ids if SAFE_ID_PATTERN.fullmatch(identifier)
        )
        super().__init__(code)


class SourceRef(FrozenModel):
    """Identify one exact source and a non-secret locator within it."""

    source_id: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    locator: str = Field(min_length=1)
    authority: Literal[
        "standard", "informative", "study_approved", "unverified", "generated"
    ]


class StandardProfile(FrozenModel):
    """Declare the exact SDTM knowledge snapshot selected for one runtime request."""

    sdtm: str = Field(min_length=1)
    sdtmig: str = Field(min_length=1)
    ct_package: str = Field(min_length=1)
    cosmos_commit: str = Field(pattern=COMMIT_PATTERN)
    include_drafts: StrictBool = False


class StudyProfile(FrozenModel):
    """Hold study identity and evidence for an explicit target standard declaration."""

    study_id: str = Field(min_length=1)
    standard: StandardProfile | None
    version_evidence: tuple[SourceRef, ...]


def _validate_relative_input_path(value: str) -> str:
    """Accept a normalized relative input path and reject answer-bearing namespaces."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(is_answer_path_part(part) for part in path.parts)
    ):
        raise ValueError("unsafe_input_path")
    return value


def is_answer_path_part(part: str) -> bool:
    """Detect a banned answer namespace in either a directory or file stem."""
    lowered = part.casefold()
    stem = PurePosixPath(lowered).stem
    return lowered in ANSWER_PATH_PARTS or stem in ANSWER_PATH_PARTS


class PageFileHash(FrozenModel):
    """Bind one reviewed page path and field allowlist to its exact serialized bytes."""

    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    field_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Reject absolute, traversing, and answer-bearing page paths."""
        return _validate_relative_input_path(value)

    @field_validator("field_ids")
    @classmethod
    def validate_field_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique safe source identifiers for every reviewed page field."""
        if len(set(value)) != len(value) or any(
            SAFE_ID_PATTERN.fullmatch(identifier) is None for identifier in value
        ):
            raise ValueError("invalid_field_allowlist")
        return value


class InputLock(FrozenModel):
    """Bind reviewed Stage A pages to original bytes and explicit field identities."""

    source_sha256: str = Field(pattern=SHA256_PATTERN)
    answer_free_reviewed: StrictBool
    review_record_id: str = Field(min_length=1)
    allowed_page_files: tuple[str, ...] = Field(min_length=1)
    page_hashes: tuple[PageFileHash, ...] = Field(min_length=1)

    @field_validator("allowed_page_files")
    @classmethod
    def validate_allowed_page_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique normalized paths outside answer-bearing directories."""
        paths = tuple(_validate_relative_input_path(path) for path in value)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate_page_path")
        return paths

    @model_validator(mode="after")
    def validate_page_hash_coverage(self) -> "InputLock":
        """Require one and only one byte hash record for every allowed page path."""
        locked_paths = tuple(page.path for page in self.page_hashes)
        if len(set(locked_paths)) != len(locked_paths) or set(locked_paths) != set(
            self.allowed_page_files
        ):
            raise ValueError("incomplete_page_hash_coverage")
        return self


class Option(FrozenModel):
    """Represent one explicit selectable control for a source field."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)


class RelatedField(FrozenModel):
    """Preserve one explicitly grouped qualifier and its source location."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    relation: Literal[
        "unit", "specimen", "method", "timing", "status", "navigation_context"
    ]
    source_ref: SourceRef


class SourceFieldContext(FrozenModel):
    """Expose reviewed source facts without claiming a target study standard."""

    study_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    field_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    form: str | None
    section_path: tuple[str, ...]
    input_type: InputType
    options: tuple[Option, ...]
    units: tuple[str, ...]
    related_fields: tuple[RelatedField, ...]
    source_refs: tuple[SourceRef, ...]


class FieldContext(SourceFieldContext):
    """Add an evidence-backed target standard to a reviewed source field."""

    target_standard: StandardProfile


def canonical_hash(value: BaseModel) -> str:
    """Hash a model's canonical finite JSON representation without mutating it."""
    payload = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_graph_profile(profile: StandardProfile) -> GraphStandardProfile:
    """Adapt the runtime Pydantic profile to the graph identity dataclass explicitly."""
    try:
        return GraphStandardProfile(**profile.model_dump(mode="python"))
    except GraphError as exc:
        raise GraphRagError("invalid_standard_profile") from exc
