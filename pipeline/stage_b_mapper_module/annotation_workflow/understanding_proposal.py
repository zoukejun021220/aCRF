"""Source judgments compiled without asking the model to restate known facts.

An interpreted subject is one writable source object. Its supporting evidence may
include other visible objects, but those references never become extra subjects.
Entry mode remains a source fact, including when the source says it is unknown.
"""

from __future__ import annotations

from typing import Literal

from ..graphrag.conditions import Identifier, SourceCondition
from ..graphrag.field_contracts import FrozenModel
from .common import DecisionExplanation, WorkflowError
from .source import SourcePacket
from .understanding import (
    Basis,
    CollectedItem,
    CollectionUnderstanding,
    Conditions,
    ContextValue,
    SourceBinding,
    Unknown,
    validate_understanding,
)


class InterpretedItem(FrozenModel):
    """Interpret one scoped subject; provenance and related context stay in basis."""

    source_ref: Identifier
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
    basis: Basis
    concept_context: tuple[ContextValue, ...] = ()


class UnderstandingProposal(FrozenModel):
    """Only meaning, source relationships, conditions and consequential unknowns."""

    items: tuple[InterpretedItem, ...]
    bindings: tuple[SourceBinding, ...]
    conditions: Conditions
    material_unknowns: tuple[Unknown, ...]
    explanation: DecisionExplanation


def _required_subjects(packet: SourcePacket) -> tuple[str, ...]:
    # A choice list may be interpreted through its parent question. An option
    # scoped on its own still needs a judgment; a merely visible parent cannot
    # authorize omitting the only writable source.
    return tuple(
        ref
        for ref in packet.output_scope
        if not (
            packet.object(ref).kind == "option"
            and packet.object(ref).parent_ref in packet.output_scope
        )
    )


def understanding_context(packet: SourcePacket) -> dict[str, object]:
    """Expose precise subject choices separately from visible supporting facts."""
    facts = [
        {
            "source_ref": item.ref,
            "text": item.text,
            "kind": item.kind,
            "entry_mode": item.entry_mode,
            "datatype": item.datatype,
            "parent_ref": item.parent_ref,
            "context_ref": item.context_ref,
        }
        for item in packet.objects
    ]
    return {
        "subject_choices": [
            fact for fact in facts if fact["source_ref"] in packet.output_scope
        ],
        "required_subject_refs": list(_required_subjects(packet)),
        "visible_source_refs": [item.ref for item in packet.objects],
        "known_source_facts": facts,
        "contract": {
            "subject": "One item per interpreted source_ref from subject_choices; never merge evidence references into its subject.",
            "coverage": "Include every required_subject_ref exactly once. Scoped options may rely on their included parent question, or be interpreted separately.",
            "evidence": "Use basis.source_refs and concept_context.source_refs for visible labels, instructions or related context supporting the subject interpretation.",
            "source_facts": "The backend retains entry_mode and datatype as supplied. Do not output value_mode or reinterpret a blank input as an instructional example.",
            "uncertainty": "Keep uncertain concepts or properties null and source_role unknown; state the evidence gap. Do not omit a required subject to express uncertainty.",
        },
    }


def compile_understanding(
    value: UnderstandingProposal,
    packet: SourcePacket,
    *,
    delivered_material_refs: tuple[str, ...] = (),
) -> CollectionUnderstanding:
    """Validate source decisions, then add only deterministic source metadata.

    Validation returns paths and allowed handles for a bounded correction turn.
    It never repairs a model judgment, drops an item, or invents source evidence.
    """
    visible = tuple(item.ref for item in packet.objects)
    feedback: list[dict[str, object]] = []

    def reject(
        code: str,
        path: str,
        message: str,
        refs: tuple[str, ...] = (),
        allowed: tuple[str, ...] = visible,
    ) -> None:
        feedback.append(
            {
                "path": path,
                "code": code,
                "message": message,
                "affected_refs": list(refs),
                "allowed_refs": list(allowed),
            }
        )

    def check_refs(
        refs: tuple[str, ...],
        path: str,
        affected: tuple[str, ...],
        *,
        allowed: tuple[str, ...] = visible,
    ) -> None:
        for index, ref in enumerate(refs):
            if ref not in allowed:
                reject(
                    "understanding_source_ref_invalid",
                    f"{path}/{index}",
                    (
                        "Use an exact visible source or already delivered material handle; an indexed but unread material is not evidence."
                        if path == "/explanation/evidence_refs"
                        else "Use an exact visible source handle; a source reference is not a material or target-standard handle."
                    ),
                    affected,
                    allowed,
                )

    def check_ref(ref: str, path: str, affected: tuple[str, ...]) -> None:
        if ref not in visible:
            reject(
                "understanding_source_ref_invalid",
                path,
                "Use an exact visible source handle.",
                affected,
            )

    def check_basis(basis: Basis, path: str, affected: tuple[str, ...]) -> None:
        check_refs(basis.source_refs, f"{path}/source_refs", affected)
        if basis.kind != "unknown" and not basis.source_refs:
            reject(
                "missing_source_basis",
                f"{path}/source_refs",
                "Explicit or inferred judgments need visible source support. If that support is unavailable, retain an unknown judgment and explain the gap.",
                affected,
            )

    def check_predicate(
        predicate: SourceCondition, path: str, affected: tuple[str, ...]
    ) -> None:
        if predicate.op == "eq" or predicate.op == "in":
            check_ref(predicate.source_id, f"{path}/source_id", affected)
        elif predicate.op == "not":
            check_predicate(predicate.arg, f"{path}/arg", affected)
        else:
            for index, arg in enumerate(predicate.args):
                check_predicate(arg, f"{path}/args/{index}", affected)

    seen: set[str] = set()
    for index, item in enumerate(value.items):
        path = f"/items/{index}"
        affected: tuple[str, ...] = (item.source_ref,)
        if item.source_ref not in packet.output_scope:
            reject(
                "understanding_subject_scope_invalid",
                f"{path}/source_ref",
                "Interpret only a scoped subject. Put visible supporting titles or instructions in basis.source_refs instead.",
                affected,
                packet.output_scope,
            )
        if item.source_ref in seen:
            reject(
                "understanding_duplicate_subject",
                f"{path}/source_ref",
                "Return one interpretation for this subject. Express relationships to other sources in bindings, not duplicate subject items.",
                affected,
                packet.output_scope,
            )
        seen.add(item.source_ref)
        check_basis(item.basis, f"{path}/basis", affected)
        for context_index, context in enumerate(item.concept_context):
            check_refs(
                context.source_refs,
                f"{path}/concept_context/{context_index}/source_refs",
                affected,
            )

    missing = tuple(ref for ref in _required_subjects(packet) if ref not in seen)
    if missing:
        reject(
            "understanding_missing_subject",
            "/items",
            "Add an interpretation for every missing required subject. A null concept/property and unknown role are valid when the source is insufficient; do not omit the subject.",
            missing,
            packet.output_scope,
        )

    for index, binding in enumerate(value.bindings):
        path = f"/bindings/{index}"
        affected = (binding.source_ref, *binding.target_source_refs)
        check_ref(binding.source_ref, f"{path}/source_ref", affected)
        check_refs(binding.target_source_refs, f"{path}/target_source_refs", affected)
        if binding.source_context_ref is not None:
            check_ref(
                binding.source_context_ref, f"{path}/source_context_ref", affected
            )
        check_basis(binding.basis, f"{path}/basis", affected)

    if (value.conditions.status == "specified") != bool(value.conditions.entries):
        reject(
            "condition_status_conflict",
            "/conditions/status",
            "Use specified only with actual condition entries. With no entries, retain unknown or none_stated_in_supplied_source according to the supplied source evidence.",
            tuple(
                dict.fromkeys(
                    ref
                    for condition in value.conditions.entries
                    for ref in condition.affected_refs
                )
            ),
            (),
        )
    for index, condition in enumerate(value.conditions.entries):
        path = f"/conditions/entries/{index}"
        affected = condition.affected_refs
        check_refs(affected, f"{path}/affected_refs", affected)
        check_basis(condition.basis, f"{path}/basis", affected)
        check_predicate(condition.predicate, f"{path}/predicate", affected)

    for index, gap in enumerate(value.material_unknowns):
        check_refs(
            gap.affected_refs,
            f"/material_unknowns/{index}/affected_refs",
            gap.affected_refs,
        )
    # Explanation may cite documents actually read in this session. Typed source
    # predicates and bases retain their source-only identity; admitting a document
    # does not fabricate a new CRF control or let an unread index entry be evidence.
    check_refs(
        value.explanation.evidence_refs,
        "/explanation/evidence_refs",
        (),
        allowed=tuple(dict.fromkeys((*visible, *delivered_material_refs))),
    )

    if feedback:
        raise WorkflowError(str(feedback[0]["code"]), feedback=tuple(feedback))

    compiled = CollectionUnderstanding(
        items=tuple(
            CollectedItem(
                source_refs=(item.source_ref,),
                concept=item.concept,
                collected_property=item.collected_property,
                source_role=item.source_role,
                value_mode=packet.object(item.source_ref).entry_mode,
                basis=item.basis,
                concept_context=item.concept_context,
            )
            for item in value.items
        ),
        bindings=value.bindings,
        conditions=value.conditions,
        material_unknowns=value.material_unknowns,
        explanation=value.explanation,
    )
    validate_understanding(compiled, packet)
    return compiled
