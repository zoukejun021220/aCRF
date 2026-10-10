"""Branch-scoped decisions with independent source conditions and target selectors."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..graphrag.conditions import Identifier, RecordSelector, Scalar, SourceCondition
from ..graphrag.field_contracts import FrozenModel
from .common import BriefReason, DecisionExplanation, WorkflowError, unique
from .understanding import Unknown


class SourceSlot(FrozenModel):
    """Preserve unknown versus an explicit absence of source conditions."""

    status: Literal["unknown", "not_applicable", "specified"]
    value: SourceCondition | None

    @model_validator(mode="after")
    def consistent(self) -> SourceSlot:
        """Reject empty specified conditions and hidden predicates in other states."""
        if (self.status == "specified") != (self.value is not None):
            raise WorkflowError("source_condition_state_conflict")
        return self


class SelectorSlot(FrozenModel):
    """Retain target selection independently of source branch activation."""

    status: Literal["unknown", "not_applicable", "specified"]
    value: RecordSelector | None

    @model_validator(mode="after")
    def consistent(self) -> SelectorSlot:
        """Reject a selector whose explicit state disagrees with its payload."""
        if (self.status == "specified") != (self.value is not None):
            raise WorkflowError("selector_state_conflict")
        return self


class Observation(FrozenModel):
    """Propose a local target observation; the backend assigns persistent identity."""

    key: Identifier
    source_context_refs: tuple[Identifier, ...] = Field(min_length=1)
    candidate_ref: Identifier
    evidence_refs: tuple[Identifier, ...] = Field(min_length=1)


class Assignment(FrozenModel):
    """An actual fixed value, never a default fabricated from an allowed-value list."""

    variable: Identifier
    value: Scalar
    evidence_refs: tuple[Identifier, ...] = Field(min_length=1)


class Annotation(FrozenModel):
    """One source scope/branch outcome with complete machine-readable semantics."""

    key: Identifier
    source_scope: tuple[Identifier, ...] = Field(min_length=1)
    source_condition: SourceSlot
    disposition: Literal["mapped", "context_only", "not_submitted", "unresolved"]
    observation_key: str | None
    variable: str | None
    record_selector: SelectorSlot
    fixed_assignments: tuple[Assignment, ...]
    evidence_refs: tuple[Identifier, ...]
    reason: BriefReason
    context_targets: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def outcome_shape(self) -> Annotation:
        """Require mapping targets only on mapped branches and explicit other reasons."""
        if self.disposition == "mapped":
            if not self.observation_key or not self.variable or not self.evidence_refs:
                raise WorkflowError("incomplete_mapping_target")
        elif (
            self.observation_key is not None
            or self.variable is not None
            or self.fixed_assignments
            or self.record_selector.status != "not_applicable"
        ):
            raise WorkflowError("target_on_nonmapping_branch")
        if self.disposition != "mapped" and not self.reason.strip():
            raise WorkflowError("missing_disposition_reason")
        return self


class Selection(FrozenModel):
    """An explicit applicability judgment; rejection never means not submitted."""

    item_refs: tuple[Identifier, ...] = Field(min_length=1)
    candidate_refs: tuple[Identifier, ...]
    decision: Literal["apply", "adapt", "unresolved", "reject"]
    reason: BriefReason


class Supplemental(FrozenModel):
    """A supplemental proposal with visible missing slots; no guessed qualifier IDs."""

    source_scope: tuple[Identifier, ...] = Field(min_length=1)
    parent_observation_key: Identifier
    qnam: str | None
    qlabel: str | None
    evidence_refs: tuple[Identifier, ...]
    missing_slots: tuple[str, ...]


class Relationship(FrozenModel):
    """A proposed relationship between observed records, separate from knowledge edges."""

    from_observation_key: Identifier
    to_observation_key: Identifier
    relation: Identifier
    evidence_refs: tuple[Identifier, ...]
    missing_slots: tuple[str, ...]


class Retraction(FrozenModel):
    """Explicitly withdraw a previously saved annotation without hiding deletions."""

    annotation_key: Identifier
    reason: Identifier


class MappingDecision(FrozenModel):
    """One business response; all valid outputs are automatically saved by the host."""

    decisions: tuple[Selection, ...]
    target_observations: tuple[Observation, ...]
    annotations: tuple[Annotation, ...]
    material_checks: tuple[str, ...]
    evidence_requests: tuple[Unknown, ...]
    supplemental: tuple[Supplemental, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    retractions: tuple[Retraction, ...] = ()
    explanation: DecisionExplanation

    @model_validator(mode="after")
    def local_identity(self) -> MappingDecision:
        """Reject duplicates and dangling local observation references before saving."""
        keys = tuple(item.key for item in self.target_observations)
        unique(keys, "duplicate_observation_key")
        unique(tuple(item.key for item in self.annotations), "duplicate_annotation_key")
        refs = [
            item.observation_key
            for item in self.annotations
            if item.observation_key is not None
        ]
        refs += [item.parent_observation_key for item in self.supplemental]
        refs += [
            ref
            for item in self.relationships
            for ref in (item.from_observation_key, item.to_observation_key)
        ]
        if not set(refs) <= set(keys):
            raise WorkflowError("unknown_observation_key")
        return self
