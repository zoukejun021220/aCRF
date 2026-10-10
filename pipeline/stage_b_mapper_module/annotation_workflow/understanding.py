"""Source interpretation and the actual consumers of its four business outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..graphrag.conditions import Identifier, SourceCondition, leaves
from ..graphrag.field_contracts import FrozenModel
from .common import BriefReason, DecisionExplanation, WorkflowError, unique
from .source import SourcePacket


class Basis(FrozenModel):
    """Store inspectable source support, not hidden reasoning or target approval."""

    kind: Literal["explicit", "inferred", "unknown"]
    source_refs: tuple[Identifier, ...]
    reason: BriefReason


class Unknown(FrozenModel):
    """Route only an unknown with a stated effect on the current proposal."""

    missing: Identifier
    affected_refs: tuple[Identifier, ...] = Field(min_length=1)
    impact: Identifier
    source_kind: Literal["source", "standard", "study", "dependency"]
    material_ref: str | None = None


class ContextValue(FrozenModel):
    """A source-grounded dimension that can distinguish mapping candidates."""

    name: Identifier
    value: Identifier
    source_refs: tuple[Identifier, ...] = Field(min_length=1)


class CollectedItem(FrozenModel):
    """Interpret an item without any final SDTM variable or submission disposition."""

    source_refs: tuple[Identifier, ...] = Field(min_length=1)
    concept: str | None
    collected_property: str | None
    source_role: Literal[
        "measurement_result",
        "event_or_intervention_description",
        "occurrence",
        "performance_status",
        "time",
        "unit",
        "qualifier_context",
        "reference",
        "other_specification",
        "administrative",
        "unknown",
    ]
    value_mode: Literal[
        "entered", "printed_context", "prefilled", "instructional_example", "unknown"
    ]
    basis: Basis
    concept_context: tuple[ContextValue, ...] = ()


class SourceBinding(FrozenModel):
    """Connect a qualifier to its source subject without selecting a target record."""

    source_ref: Identifier
    relation: Literal["unit_for", "time_of", "qualifies", "references"]
    target_source_refs: tuple[Identifier, ...] = Field(min_length=1)
    source_context_ref: str | None = None
    basis: Basis


class CollectionCondition(FrozenModel):
    """Keep the activating source predicate separate from target record selection."""

    affected_refs: tuple[Identifier, ...] = Field(min_length=1)
    kind: Literal["collection", "visibility", "other_specify"]
    predicate: SourceCondition
    basis: Basis


class Conditions(FrozenModel):
    """An empty list never silently proves absence of source conditions."""

    status: Literal["unknown", "none_stated_in_supplied_source", "specified"]
    entries: tuple[CollectionCondition, ...]


class CollectionUnderstanding(FrozenModel):
    """The four semantic outputs; IDs and lifecycle bookkeeping stay on the backend."""

    items: tuple[CollectedItem, ...]
    bindings: tuple[SourceBinding, ...]
    conditions: Conditions
    material_unknowns: tuple[Unknown, ...]
    explanation: DecisionExplanation


class MappingQuery(FrozenModel):
    """Version-neutral semantic query with source-linked modifiers and questions."""

    item_ref: Identifier
    source_refs: tuple[Identifier, ...]
    concept: str
    collected_property: str
    context: tuple[ContextValue, ...]
    conditions: tuple[CollectionCondition, ...]


def validate_understanding(
    value: CollectionUnderstanding, packet: SourcePacket
) -> None:
    """Reject fabricated source references/modes; keep partial semantic uncertainty."""
    visible = {item.ref for item in packet.objects}
    refs: list[str] = []
    for item in value.items:
        unique(item.source_refs, "duplicate_item_source")
        refs.extend((*item.source_refs, *item.basis.source_refs))
        if item.basis.kind != "unknown" and not item.basis.source_refs:
            raise WorkflowError("missing_source_basis")
        for context in item.concept_context:
            refs.extend(context.source_refs)
        for ref in item.source_refs:
            mode = packet.object(ref).entry_mode
            if mode != "unknown" and mode != item.value_mode:
                raise WorkflowError("source_value_mode_conflict")
    for binding in value.bindings:
        refs.extend(
            (
                binding.source_ref,
                *binding.target_source_refs,
                *binding.basis.source_refs,
            )
        )
        if binding.source_context_ref is not None:
            refs.append(binding.source_context_ref)
    if (value.conditions.status == "specified") != bool(value.conditions.entries):
        raise WorkflowError("condition_status_conflict")
    for condition in value.conditions.entries:
        refs.extend((*condition.affected_refs, *condition.basis.source_refs))
        refs.extend(
            leaf.source_id
            for leaf in leaves(condition.predicate)
            if hasattr(leaf, "source_id")
        )
    for gap in value.material_unknowns:
        refs.extend(gap.affected_refs)
    if not set(refs) <= visible:
        raise WorkflowError("unknown_source_ref")
    if not value.items and not value.material_unknowns:
        raise WorkflowError("empty_understanding")


def queries(
    value: CollectionUnderstanding,
    packet: SourcePacket,
    *,
    include_qualifiers: bool = False,
) -> tuple[MappingQuery, ...]:
    """Consume concepts/properties and attach only qualifiers belonging to this item."""
    validate_understanding(value, packet)
    output = []
    for index, item in enumerate(value.items):
        if (
            not item.concept
            or not item.collected_property
            or (
                not include_qualifiers
                and item.source_role
                in {"unit", "time", "administrative", "qualifier_context", "unknown"}
            )
        ):
            continue
        context = list(item.concept_context)
        # Source labels retain critical qualifiers omitted by a model's generic
        # concept (e.g. "distance"). They are lexical evidence, not target labels.
        for ref in item.source_refs:
            source = packet.object(ref)
            if source.text:
                context.append(
                    ContextValue(
                        name="source_label", value=source.text, source_refs=(ref,)
                    )
                )
        for binding in value.bindings:
            if set(binding.target_source_refs) & set(item.source_refs):
                source = packet.object(binding.source_ref)
                context.append(
                    ContextValue(
                        name=binding.relation,
                        value=source.text,
                        source_refs=(source.ref,),
                    )
                )
        output.append(
            MappingQuery(
                item_ref=f"item:{index+1}",
                source_refs=item.source_refs,
                concept=item.concept,
                collected_property=item.collected_property,
                context=tuple(context),
                conditions=tuple(
                    condition
                    for condition in value.conditions.entries
                    if set(condition.affected_refs) & set(item.source_refs)
                ),
            )
        )
    return tuple(output)
