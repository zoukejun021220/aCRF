"""Small model judgments compiled into evidence-linked internal mapping objects.

Ordinary bindings contain only source handles and a target. Special variants keep
real branch/value/relationship decisions expressible without nullable bookkeeping
on every ordinary binding. Aliases are ordered session choices, never model IDs.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, TypeAlias

from pydantic import Field

from ..graphrag.conditions import (
    Identifier,
    RecordAll,
    RecordEq,
    Scalar,
    SourceAll,
    SourceCondition,
    TargetRef,
    leaves,
)
from ..graphrag.field_contracts import FrozenModel
from .common import BriefReason, DecisionExplanation, WorkflowError, canonical, digest
from .decision import (
    Annotation,
    Assignment,
    MappingDecision,
    Observation,
    Relationship,
    Selection,
    SelectorSlot,
    SourceSlot,
    Supplemental,
)
from .knowledge import Candidate
from .source import Material, SourcePacket
from .understanding import CollectionUnderstanding, Unknown

When: TypeAlias = Identifier | SourceCondition


class Binding(FrozenModel):
    source: tuple[Identifier, ...] = Field(min_length=1)
    target: Identifier


class ConditionalBinding(Binding):
    when: When


class SupportedValue(FrozenModel):
    """A model-selected value needs delivered support beyond an allowed-value list."""

    target: Identifier
    value: Scalar
    support: tuple[Identifier, ...] = Field(min_length=1)


class BindingDetails(FrozenModel):
    when: When = "inherit"
    selectors: tuple[SupportedValue, ...] = ()
    constants: tuple[SupportedValue, ...] = ()
    support: tuple[Identifier, ...] = ()


class QualifiedBinding(Binding):
    details: BindingDetails


class RecordLink(FrozenModel):
    record: int = Field(ge=1)
    relation: Identifier
    support: tuple[Identifier, ...] = Field(min_length=1)
    reason: BriefReason


class ProposedRecord(FrozenModel):
    template: Identifier
    bindings: tuple[Binding | ConditionalBinding | QualifiedBinding, ...] = Field(
        min_length=1
    )
    reason: BriefReason


class LinkedRecord(ProposedRecord):
    links: tuple[RecordLink, ...] = Field(min_length=1)


class OtherOutcome(FrozenModel):
    source: tuple[Identifier, ...] = Field(min_length=1)
    outcome: Literal["context_only", "not_submitted", "unresolved"]
    reason: BriefReason


class OutcomeDetails(FrozenModel):
    when: When = "inherit"
    support: tuple[Identifier, ...] = ()
    context_targets: tuple[Identifier, ...] = ()


class QualifiedOutcome(OtherOutcome):
    details: OutcomeDetails


class SupplementalOutcome(FrozenModel):
    outcome: Literal["supplemental"]
    source: tuple[Identifier, ...] = Field(min_length=1)
    parent_record: int = Field(ge=1)
    qnam: Identifier | None
    qlabel: Identifier | None
    support: tuple[Identifier, ...] = Field(min_length=1)
    reason: BriefReason


class Proposal(FrozenModel):
    """Only grouping, applicability, binding, disposition and material questions."""

    records: tuple[ProposedRecord | LinkedRecord, ...]
    other_outcomes: tuple[OtherOutcome | QualifiedOutcome | SupplementalOutcome, ...]
    open_questions: tuple[Unknown, ...]


def restrict_proposal_sources(
    schema: dict[str, Any], output_sources: tuple[str, ...]
) -> dict[str, Any]:
    """Constrain writable handles in an already prepared provider schema.

    Support and source predicates retain read visibility. This is an input-bound
    grammar aid; the compiler still enforces scope on every submitted proposal.
    The caller's generic schema is never mutated or reused with another scope.
    """
    result = copy.deepcopy(schema)
    definitions = result.get("$defs", {})
    for name in (
        "Binding",
        "ConditionalBinding",
        "QualifiedBinding",
        "OtherOutcome",
        "QualifiedOutcome",
        "SupplementalOutcome",
    ):
        if name in definitions:
            definitions[name]["properties"]["source"]["items"]["enum"] = list(
                output_sources
            )
    return result


def _ordered(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _identity(kind: str, value: object) -> str:
    return f"{kind}:{digest(canonical(value).encode())[:20]}"


def _topic(candidate: Candidate) -> tuple[str, str] | None:
    rows = [row for row in candidate.rows if row.values().get("role") == "Topic"]
    if len(rows) == 1:
        value = rows[0].values().get("assigned_value", "")
        if value:
            return rows[0].variable, value
    return None


def _candidate_aliases(
    packet: SourcePacket,
    candidates: tuple[Candidate, ...],
    materials: tuple[Material, ...],
    reserved_refs: tuple[str, ...],
) -> dict[str, Candidate]:
    # Reserve even admitted, unread material handles at the session boundary. A
    # later material read must not rename template choices in a saved proposal.
    occupied = (
        {item.ref for item in packet.objects}
        | {item.ref for item in materials}
        | set(reserved_refs)
    )
    aliases: dict[str, Candidate] = {}
    number = 1
    for candidate in candidates:
        while f"T{number}" in occupied:
            number += 1
        aliases[f"T{number}"] = candidate
        number += 1
    return aliases


def proposal_context(
    packet: SourcePacket,
    understanding: CollectionUnderstanding,
    candidates: tuple[Candidate, ...],
    *,
    materials: tuple[Material, ...] = (),
    reserved_refs: tuple[str, ...] = (),
) -> dict[str, object]:
    """Give exact choices and useful knowledge, keeping hashes/internal IDs backend-owned."""
    return {
        "output_sources": list(packet.output_scope),
        "support_sources": [item.ref for item in packet.objects],
        "readonly_support_sources": [
            item.ref for item in packet.objects if item.ref not in packet.output_scope
        ],
        "choice_groups": [
            {
                "parent": parent.ref,
                "options": [
                    item.ref
                    for item in packet.objects
                    if item.kind == "option" and item.parent_ref == parent.ref
                ],
                "selection": (
                    "single"
                    if parent.datatype in {"single_choice", "radio"}
                    else "multiple" if parent.datatype == "checkbox" else "unknown"
                ),
            }
            for parent in packet.objects
            if any(
                item.kind == "option" and item.parent_ref == parent.ref
                for item in packet.objects
            )
        ],
        "support_materials": [
            item.ref for item in materials if item.study_id == packet.study_id
        ],
        "templates": [
            {
                "alias": alias,
                "title": candidate.title,
                "domain": candidate.domain,
                "kind": candidate.kind,
                "sdtmig": candidate.sdtmig,
                "source_version": candidate.source_version,
                "coverage_limits": list(candidate.coverage_limits),
                "excerpt": candidate.excerpt,
                "targets": [
                    row.variable
                    for row in candidate.rows
                    if row.values().get("origin_type") != "Derived"
                ],
                "automatic_topic_selector": (
                    {"target": topic[0], "value": topic[1]}
                    if (topic := _topic(candidate)) is not None
                    else None
                ),
                "rows": [row.values() for row in candidate.rows],
            }
            for alias, candidate in _candidate_aliases(
                packet, candidates, materials, reserved_refs
            ).items()
        ],
        "source_conditions": {
            "status": understanding.conditions.status,
            "choices": [
                {
                    "handle": f"C{index + 1}",
                    "affected_sources": list(condition.affected_refs),
                    "predicate": condition.predicate.model_dump(mode="json"),
                }
                for index, condition in enumerate(understanding.conditions.entries)
            ],
            "default": "inherit each bound source's conditions; unknown stays unknown",
        },
    }


class _Compiler:
    def __init__(
        self,
        packet: SourcePacket,
        understanding: CollectionUnderstanding,
        candidates: tuple[Candidate, ...],
        materials: tuple[Material, ...],
        reserved_refs: tuple[str, ...],
    ) -> None:
        self.packet = packet
        self.understanding = understanding
        self.candidates = _candidate_aliases(
            packet, candidates, materials, reserved_refs
        )
        self.visible = {item.ref for item in packet.objects}
        self.materials = {
            item.ref for item in materials if item.study_id == packet.study_id
        }

    def fail(
        self,
        code: str,
        path: str,
        source: tuple[str, ...],
        message: str,
        allowed: tuple[str, ...] = (),
    ) -> None:
        raise WorkflowError(
            code,
            feedback=(
                {
                    "code": code,
                    "path": path,
                    "affected_refs": list(source),
                    "message": message,
                    "allowed": list(allowed),
                },
            ),
        )

    def scope(self, source: tuple[str, ...], path: str) -> None:
        if len(source) != len(set(source)) or not set(source) <= set(
            self.packet.output_scope
        ):
            self.fail(
                "proposal_source_scope_invalid",
                path,
                source,
                "Use each output_sources handle once within this binding or outcome. "
                "Readonly support sources may be cited as evidence, never given an outcome.",
                self.packet.output_scope,
            )

    def source_identity(self, ref: str) -> set[str]:
        """An option belongs to its explicit parent; nearby context is not identity.

        The understanding contract permits options to rely on a scoped parent
        item. Preserve the original binding while resolving that interpretation.
        This association alone never asserts single-choice coverage.
        """
        item = self.packet.object(ref)
        return {ref} | (
            {item.parent_ref}
            if item.kind == "option" and item.parent_ref in self.packet.output_scope
            else set()
        )

    def support(self, refs: tuple[str, ...], path: str) -> tuple[str, ...]:
        output = []
        for ref in refs:
            if ref in self.candidates:
                output.append(self.candidates[ref].ref)
            elif ref in self.visible | self.materials:
                output.append(ref)
            else:
                self.fail(
                    "proposal_support_unknown",
                    path,
                    (),
                    "Select delivered source, template alias, or material support.",
                    tuple(sorted(self.visible | self.materials | set(self.candidates))),
                )
        return _ordered(output)

    def condition(self, source: tuple[str, ...], when: When, path: str) -> SourceSlot:
        conditions = self.understanding.conditions
        by_source = [
            tuple(
                entry.predicate
                for entry in conditions.entries
                if self.source_identity(ref) & set(entry.affected_refs)
            )
            for ref in source
        ]
        fingerprints = {
            canonical([entry.model_dump(mode="json") for entry in predicates])
            for predicates in by_source
        }
        if len(fingerprints) > 1:
            self.fail(
                "proposal_mixed_source_conditions",
                path,
                source,
                "Split sources with different inherited conditions into separate bindings.",
            )
        inherited = by_source[0]
        if when == "unknown":
            if inherited:
                self.fail(
                    "proposal_condition_weakened",
                    path,
                    source,
                    "Retain the known inherited condition; request missing evidence separately.",
                )
            return SourceSlot(status="unknown", value=None)
        if when == "inherit":
            if conditions.status == "unknown":
                return SourceSlot(status="unknown", value=None)
            selected = inherited
        elif isinstance(when, str):
            handles = {
                f"C{index + 1}": entry for index, entry in enumerate(conditions.entries)
            }
            if when not in handles or not all(
                self.source_identity(ref) & set(handles[when].affected_refs)
                for ref in source
            ):
                self.fail(
                    "proposal_condition_handle_invalid",
                    path,
                    source,
                    "Use an applicable condition handle or an explicit source predicate.",
                    ("inherit", "unknown", *handles),
                )
            selected = (*inherited, handles[when].predicate)
        else:
            if (
                not {
                    leaf.source_id
                    for leaf in leaves(when)
                    if hasattr(leaf, "source_id")
                }
                <= self.visible
            ):
                self.fail(
                    "proposal_condition_source_unknown",
                    path,
                    source,
                    "Source predicates may reference only visible source handles.",
                    tuple(sorted(self.visible)),
                )
            selected = (*inherited, when)
        unique = {canonical(entry.model_dump(mode="json")): entry for entry in selected}
        predicates = tuple(unique.values())
        if not predicates:
            return SourceSlot(status="not_applicable", value=None)
        return SourceSlot(
            status="specified",
            value=(
                predicates[0]
                if len(predicates) == 1
                else SourceAll(op="all", args=predicates)
            ),
        )

    def row(
        self, candidate: Candidate, target: str, path: str, source: tuple[str, ...]
    ) -> dict[str, str]:
        allowed = tuple(
            row.variable
            for row in candidate.rows
            if row.values().get("origin_type") != "Derived"
        )
        if target not in allowed:
            self.fail(
                "proposal_target_invalid",
                path,
                source,
                "Choose a collected or assigned target present in this template.",
                allowed,
            )
        return candidate.row(target).values()

    def literal(
        self,
        candidate: Candidate,
        value: SupportedValue,
        path: str,
        source: tuple[str, ...],
    ) -> tuple[str, ...]:
        row = self.row(candidate, value.target, path, source)
        literal = str(value.value)
        if (row.get("assigned_value") and row["assigned_value"] != literal) or (
            row.get("value_list") and literal not in row["value_list"].split(";")
        ):
            self.fail(
                "template_value_conflict",
                path,
                source,
                "The chosen value contradicts this template row.",
            )
        refs = self.support(value.support, path + "/support")
        if not set(refs) & (self.visible | self.materials) and not (
            row.get("assigned_value") == literal
        ):
            self.fail(
                "proposal_value_support_missing",
                path,
                source,
                "An allowed-value list is not evidence that this value applies; select source or material support.",
            )
        return _ordered((*refs, candidate.ref))

    def annotation(
        self,
        binding: Binding,
        candidate: Candidate,
        observation: str,
        reason: str,
        path: str,
    ) -> Annotation:
        self.scope(binding.source, path + "/source")
        row = self.row(candidate, binding.target, path + "/target", binding.source)
        details = (
            binding.details
            if isinstance(binding, QualifiedBinding)
            else BindingDetails()
        )
        when = binding.when if isinstance(binding, ConditionalBinding) else details.when
        condition = self.condition(binding.source, when, path + "/when")
        refs = _ordered(
            (
                *binding.source,
                candidate.ref,
                *self.support(details.support, path + "/details/support"),
            )
        )
        selectors = []
        topic = _topic(candidate)
        if topic is not None:
            selectors.append(
                RecordEq(
                    op="eq",
                    variable_ref=TargetRef(kind="variable", id=topic[0]),
                    value=topic[1],
                    evidence_ids=(candidate.ref,),
                )
            )
        for index, value in enumerate(details.selectors):
            support = self.literal(
                candidate, value, f"{path}/details/selectors/{index}", binding.source
            )
            selectors.append(
                RecordEq(
                    op="eq",
                    variable_ref=TargetRef(kind="variable", id=value.target),
                    value=value.value,
                    evidence_ids=support,
                )
            )
        by_target: dict[str, object] = {}
        for selector in selectors:
            target = selector.variable_ref.id
            if target in by_target and by_target[target] != selector.value:
                self.fail(
                    "proposal_selector_conflict",
                    path,
                    binding.source,
                    "Two selector values contradict each other.",
                )
            by_target[target] = selector.value
        selectors = list(
            {
                canonical(item.model_dump(mode="json")): item for item in selectors
            }.values()
        )
        assignments = []
        for index, value in enumerate(details.constants):
            support = self.literal(
                candidate, value, f"{path}/details/constants/{index}", binding.source
            )
            assignments.append(
                Assignment(
                    variable=value.target, value=value.value, evidence_refs=support
                )
            )
        # An allowed list is consulted only after a source literal exists; it never
        # supplies a patient value or causes additional template rows to be mapped.
        if (
            not any(item.variable == binding.target for item in assignments)
            and row.get("origin_type") == "Collected"
        ):
            printed = {
                self.packet.object(ref).text.strip()
                for ref in binding.source
                if self.packet.object(ref).entry_mode == "printed_context"
                and any(
                    ref in item.source_refs and item.value_mode == "printed_context"
                    for item in self.understanding.items
                )
            }
            if len(printed) == 1:
                literal = next(iter(printed))
                if literal and (
                    literal == row.get("assigned_value")
                    or literal in row.get("value_list", "").split(";")
                ):
                    assignments.append(
                        Assignment(
                            variable=binding.target, value=literal, evidence_refs=refs
                        )
                    )
        variables = [item.variable for item in assignments]
        if len(variables) != len(set(variables)):
            self.fail(
                "proposal_duplicate_constant",
                path,
                binding.source,
                "Provide one fixed value per target variable.",
            )
        selector_slot = SelectorSlot(
            status="specified" if selectors else "not_applicable",
            value=(
                selectors[0]
                if len(selectors) == 1
                else RecordAll(op="all", args=tuple(selectors)) if selectors else None
            ),
        )
        return Annotation(
            key=_identity(
                "annotation",
                [
                    observation,
                    sorted(binding.source),
                    condition.model_dump(mode="json"),
                    binding.target,
                ],
            ),
            source_scope=binding.source,
            source_condition=condition,
            disposition="mapped",
            observation_key=observation,
            variable=binding.target,
            record_selector=selector_slot,
            fixed_assignments=tuple(assignments),
            evidence_refs=refs,
            reason=reason,
        )

    def compile(self, value: Proposal) -> MappingDecision:
        observations: list[Observation] = []
        annotations: list[Annotation] = []
        selections: list[Selection] = []
        supplemental: list[Supplemental] = []
        relationships: list[Relationship] = []
        for index, record in enumerate(value.records):
            path = f"/records/{index}"
            source = _ordered(
                [ref for binding in record.bindings for ref in binding.source]
            )
            if record.template not in self.candidates:
                self.fail(
                    "proposal_template_unknown",
                    path + "/template",
                    source,
                    "Choose one of the delivered template aliases.",
                    tuple(self.candidates),
                )
            candidate = self.candidates[record.template]
            for binding in record.bindings:
                self.scope(binding.source, path + "/bindings")
            binding_keys = []
            for binding in record.bindings:
                binding_details = (
                    binding.details
                    if isinstance(binding, QualifiedBinding)
                    else BindingDetails()
                )
                when = (
                    binding.when
                    if isinstance(binding, ConditionalBinding)
                    else binding_details.when
                )
                binding_keys.append(
                    {
                        "source": sorted(binding.source),
                        "target": binding.target,
                        "condition": self.condition(
                            binding.source, when, path + "/bindings"
                        ).model_dump(mode="json"),
                        "selectors": sorted(
                            canonical(item.model_dump(mode="json", exclude={"support"}))
                            for item in binding_details.selectors
                        ),
                        "constants": sorted(
                            canonical(item.model_dump(mode="json", exclude={"support"}))
                            for item in binding_details.constants
                        ),
                    }
                )
            key = _identity(
                "observation",
                [
                    candidate.ref,
                    sorted(source),
                    sorted(canonical(item) for item in binding_keys),
                ],
            )
            contexts = _ordered(
                [
                    context
                    for ref in source
                    if (context := self.packet.object(ref).context_ref) is not None
                ]
            )
            contexts = contexts or source
            observations.append(
                Observation(
                    key=key,
                    source_context_refs=contexts,
                    candidate_ref=candidate.ref,
                    evidence_refs=_ordered((*contexts, candidate.ref)),
                )
            )
            items = tuple(
                f"item:{i + 1}"
                for i, item in enumerate(self.understanding.items)
                if any(
                    set(item.source_refs) & self.source_identity(ref) for ref in source
                )
            )
            if not items:
                self.fail(
                    "proposal_source_not_understood",
                    path,
                    source,
                    "The record must bind at least one interpreted source item.",
                )
            selections.append(
                Selection(
                    item_refs=items,
                    candidate_refs=(candidate.ref,),
                    decision="apply",
                    reason=record.reason,
                )
            )
            for binding_index, binding in enumerate(record.bindings):
                annotations.append(
                    self.annotation(
                        binding,
                        candidate,
                        key,
                        record.reason,
                        f"{path}/bindings/{binding_index}",
                    )
                )
        keys = [item.key for item in observations]
        if len(keys) != len(set(keys)):
            self.fail(
                "proposal_duplicate_record",
                "/records",
                (),
                "Merge duplicate records; express distinct branches in their bindings.",
            )

        def record_key(index: int, path: str, source: tuple[str, ...]) -> str:
            if index > len(keys):
                self.fail(
                    "proposal_record_reference_invalid",
                    path,
                    source,
                    "Use a 1-based record position from this proposal.",
                    tuple(str(i + 1) for i in range(len(keys))),
                )
            return keys[index - 1]

        for index, outcome in enumerate(value.other_outcomes):
            path = f"/other_outcomes/{index}"
            self.scope(outcome.source, path + "/source")
            if isinstance(outcome, SupplementalOutcome):
                parent = record_key(
                    outcome.parent_record, path + "/parent_record", outcome.source
                )
                support = _ordered(
                    (*outcome.source, *self.support(outcome.support, path + "/support"))
                )
                missing = tuple(
                    name
                    for name in ("qnam", "qlabel")
                    if getattr(outcome, name) is None
                )
                supplemental.append(
                    Supplemental(
                        source_scope=outcome.source,
                        parent_observation_key=parent,
                        qnam=outcome.qnam,
                        qlabel=outcome.qlabel,
                        evidence_refs=support,
                        missing_slots=missing,
                    )
                )
                disposition: Literal["context_only", "not_submitted", "unresolved"] = (
                    "unresolved"
                )
                details = OutcomeDetails(support=outcome.support)
            else:
                disposition = outcome.outcome
                details = (
                    outcome.details
                    if isinstance(outcome, QualifiedOutcome)
                    else OutcomeDetails()
                )
            support = _ordered(
                (
                    *outcome.source,
                    *self.support(details.support, path + "/details/support"),
                )
            )
            if not set(details.context_targets) <= self.visible:
                self.fail(
                    "proposal_context_target_invalid",
                    path,
                    outcome.source,
                    "Context targets must be visible source handles.",
                    tuple(sorted(self.visible)),
                )
            condition = self.condition(outcome.source, details.when, path + "/when")
            annotations.append(
                Annotation(
                    key=_identity(
                        "annotation",
                        [
                            sorted(outcome.source),
                            condition.model_dump(mode="json"),
                            disposition,
                        ],
                    ),
                    source_scope=outcome.source,
                    source_condition=condition,
                    disposition=disposition,
                    observation_key=None,
                    variable=None,
                    record_selector=SelectorSlot(status="not_applicable", value=None),
                    fixed_assignments=(),
                    evidence_refs=support,
                    reason=outcome.reason,
                    context_targets=details.context_targets,
                )
            )
        for index, record in enumerate(value.records):
            if not isinstance(record, LinkedRecord):
                continue
            for link_index, link in enumerate(record.links):
                path = f"/records/{index}/links/{link_index}"
                target = record_key(link.record, path + "/record", ())
                if target == keys[index]:
                    self.fail(
                        "proposal_self_relationship",
                        path,
                        (),
                        "Choose another proposed record.",
                    )
                support = self.support(link.support, path + "/support")
                relationships.append(
                    Relationship(
                        from_observation_key=keys[index],
                        to_observation_key=target,
                        relation=link.relation,
                        evidence_refs=support,
                        missing_slots=(),
                    )
                )
        annotation_keys = [annotation.key for annotation in annotations]
        if len(annotation_keys) != len(set(annotation_keys)):
            self.fail(
                "proposal_duplicate_outcome",
                "/",
                _ordered(
                    [
                        ref
                        for annotation in annotations
                        for ref in annotation.source_scope
                    ]
                ),
                "Remove duplicate outcomes; keep distinct targets and source branches explicit.",
            )
        explanation_refs = _ordered(
            [ref for annotation in annotations for ref in annotation.evidence_refs]
        )[:24]
        uncertainties = tuple(
            question.missing[:600] for question in value.open_questions
        )[:8]
        reasons = [item.reason for item in (*value.records, *value.other_outcomes)] or [
            question.impact for question in value.open_questions
        ]
        if not reasons:
            self.fail(
                "proposal_empty",
                "/",
                self.packet.output_scope,
                "Give every authorized source a mapping or an explicit other outcome with a reason.",
            )
        questions = tuple(
            (
                question.model_copy(
                    update={"material_ref": self.candidates[question.material_ref].ref}
                )
                if question.source_kind == "standard"
                and question.material_ref in self.candidates
                else question
            )
            for question in value.open_questions
        )
        return MappingDecision(
            decisions=tuple(selections),
            target_observations=tuple(observations),
            annotations=tuple(annotations),
            material_checks=(),
            evidence_requests=questions,
            supplemental=tuple(supplemental),
            relationships=tuple(relationships),
            explanation=DecisionExplanation(
                summary=" ".join(reasons)[:600],
                evidence_refs=explanation_refs,
                uncertainties=uncertainties,
            ),
        )


def compile_proposal(
    value: Proposal,
    packet: SourcePacket,
    understanding: CollectionUnderstanding,
    candidates: tuple[Candidate, ...],
    *,
    materials: tuple[Material, ...] = (),
    reserved_refs: tuple[str, ...] = (),
) -> MappingDecision:
    """Expand only deterministic consequences; caller retains full internal validation."""
    return _Compiler(
        packet, understanding, candidates, materials, reserved_refs
    ).compile(value)
