"""Mechanical evidence/scope checks; clinical semantics remain reviewable."""

from __future__ import annotations

from ..graphrag.conditions import RecordEq, RecordIn, SourceEq, SourceIn, leaves
from .common import Issue, WorkflowError, canonical
from .coverage import applies, option_coverage
from .decision import Annotation, MappingDecision
from .knowledge import Candidate
from .source import Material, SourceObject, SourcePacket
from .understanding import CollectionUnderstanding, queries


def _option_active(annotation: Annotation, option: SourceObject) -> bool | None:
    """Evaluate only an explicitly related option and its parent's finite branch."""
    if not {option.ref, option.parent_ref} & set(annotation.source_scope):
        return False
    slot = annotation.source_condition
    if slot.status == "unknown":
        return None
    if slot.value is None:
        return True
    assert option.parent_ref is not None
    return applies(
        slot.value,
        option.parent_ref,
        (
            option.text,
            *((option.stored_code,) if option.stored_code is not None else ()),
        ),
    )


def covered_source_refs(
    decision: MappingDecision, packet: SourcePacket
) -> frozenset[str]:
    """Count explicit dispositions and proven single-choice coverage symmetrically.

    Binding a parent covers its applicable radio branches. Conversely, all known
    branches must have an applicable disposition before child bindings cover the
    parent. This does not create annotations, map values, merge records, or expand
    write permission. Multiple selections and unknown predicates are not inferred.
    """
    covered = {
        ref for annotation in decision.annotations for ref in annotation.source_scope
    }
    parents: dict[str, list[bool]] = {}
    for option in packet.objects:
        if option.kind != "option" or option.parent_ref is None:
            continue
        parent = packet.object(option.parent_ref)
        if parent.datatype not in {"single_choice", "radio"}:
            continue
        active = any(
            _option_active(annotation, option) is True
            for annotation in decision.annotations
        )
        parents.setdefault(parent.ref, []).append(active)
        if active:
            covered.add(option.ref)
    covered.update(parent for parent, branches in parents.items() if all(branches))
    return frozenset(covered & set(packet.output_scope))


def _source_value_mapping_issue(
    annotation: Annotation, candidate: Candidate, packet: SourcePacket
) -> Issue | None:
    """Keep finite source-to-codelist compatibility open unless literals match.

    A named codelist is not its term inventory. Labels/codes are used verbatim;
    this check neither parses score prefixes nor invents a value conversion.
    """
    assert annotation.variable is not None
    row = candidate.row(annotation.variable).values()
    if row.get("origin_type") != "Collected" or not any(
        row.get(name)
        for name in ("value_list", "codelist", "codelist_submission_value")
    ):
        return None
    options = [
        option
        for option in packet.objects
        if option.kind == "option"
        and option.parent_ref is not None
        and _option_active(annotation, option) is not False
    ]
    if not options:
        return None
    allowed = set(row.get("value_list", "").split(";")) - {""}
    if allowed and all(
        option.text in allowed
        or (option.stored_code is not None and option.stored_code in allowed)
        for option in options
    ):
        return None
    return Issue(
        code="source_value_mapping_unresolved",
        refs=(
            annotation.key,
            candidate.ref,
            annotation.variable,
            *(option.ref for option in options),
        ),
    )


def check_decision(
    decision: MappingDecision,
    packet: SourcePacket,
    understanding: CollectionUnderstanding,
    delivered: tuple[Candidate, ...],
    materials: tuple[Material, ...] = (),
) -> tuple[Issue, ...]:
    """Reject proven contradictions; return review issues for uncertainty/coverage.

    This validates concrete evidence existence and scoped template occurrences,
    not arbitrary clinical entailment. Unresolved facts are retained for review.
    """
    visible = {item.ref for item in packet.objects}
    candidates = {item.ref: item for item in delivered}
    allowed = (
        visible
        | set(candidates)
        | {item.ref for item in materials if item.study_id == packet.study_id}
    )
    observations = {item.key: item for item in decision.target_observations}
    item_refs = {f"item:{index+1}" for index in range(len(understanding.items))}
    issues: list[Issue] = []
    for unknown in understanding.material_unknowns:
        issues.append(
            Issue(code="source_understanding_gap", refs=unknown.affected_refs)
        )
    if understanding.conditions.status == "unknown":
        issues.append(Issue(code="source_conditions_unknown"))

    def evidence(refs: tuple[str, ...]) -> None:
        """Enforce delivery and same-study evidence visibility at every claim."""
        if not set(refs) <= allowed:
            raise WorkflowError("evidence_not_delivered")

    def literal(candidate: Candidate, variable: str, value: object) -> None:
        """Check exact assigned/allowed source values in the chosen template only."""
        row = candidate.row(variable).values()
        fixed = row.get("assigned_value", "")
        allowed_values = row.get("value_list", "")
        if fixed and str(value) != fixed:
            raise WorkflowError("template_value_conflict")
        if allowed_values and str(value) not in allowed_values.split(";"):
            raise WorkflowError("template_value_conflict")
        if not fixed and not allowed_values:
            issues.append(
                Issue(code="value_support_unresolved", refs=(candidate.ref, variable))
            )

    evidence(decision.explanation.evidence_refs)
    for selection in decision.decisions:
        if not set(selection.item_refs) <= item_refs or not set(
            selection.candidate_refs
        ) <= set(candidates):
            raise WorkflowError("unknown_selection_ref")
        if selection.decision in {"apply", "adapt"} and not selection.candidate_refs:
            raise WorkflowError("missing_candidate")
    for observation in decision.target_observations:
        if (
            observation.candidate_ref not in candidates
            or not set(observation.source_context_refs) <= visible
        ):
            raise WorkflowError("unknown_observation_support")
        evidence(observation.evidence_refs)
        if observation.candidate_ref not in observation.evidence_refs:
            raise WorkflowError("missing_candidate_support")
        if not any(
            selection.decision in {"apply", "adapt"}
            and observation.candidate_ref in selection.candidate_refs
            for selection in decision.decisions
        ):
            raise WorkflowError("observation_without_selection")
    seen: dict[tuple[str, str], str] = {}
    covered: set[str] = set()
    for annotation in decision.annotations:
        if not set(annotation.source_scope) <= set(packet.output_scope):
            raise WorkflowError("output_scope_denied")
        evidence(annotation.evidence_refs)
        for leaf in leaves(annotation.source_condition.value):
            if isinstance(leaf, (SourceEq, SourceIn)) and leaf.source_id not in visible:
                raise WorkflowError("unknown_source_condition_ref")
        for ref in annotation.context_targets:
            if ref not in visible:
                raise WorkflowError("unknown_context_target")
        for ref in annotation.source_scope:
            key = (ref, canonical(annotation.source_condition.model_dump(mode="json")))
            if key in seen and seen[key] != annotation.disposition:
                raise WorkflowError("branch_disposition_conflict")
            seen[key] = annotation.disposition
            covered.add(ref)
        if (
            annotation.source_condition.status == "unknown"
            or annotation.disposition == "unresolved"
        ):
            issues.append(Issue(code="annotation_unresolved", refs=(annotation.key,)))
        if annotation.disposition == "not_submitted":
            study = any(
                item.ref in annotation.evidence_refs
                and item.kind == "study"
                and item.approved
                and item.study_id == packet.study_id
                for item in materials
            )
            standard = any(
                ref in candidates
                and candidates[ref].kind == "rule_evidence_bundle"
                and "NOT SUBMITTED" in candidates[ref].excerpt
                for ref in annotation.evidence_refs
            )
            if not study and not standard:
                issues.append(
                    Issue(code="submission_basis_unresolved", refs=(annotation.key,))
                )
        if annotation.disposition != "mapped":
            continue
        assert (
            annotation.observation_key is not None and annotation.variable is not None
        )
        observation = observations[annotation.observation_key]
        candidate = candidates[observation.candidate_ref]
        if (
            candidate.sdtmig != packet.standard.sdtmig
            or candidate.ref not in annotation.evidence_refs
        ):
            raise WorkflowError("mapping_evidence_mismatch")
        row = candidate.row(annotation.variable).values()
        if row.get("origin_type") == "Derived":
            raise WorkflowError("derived_as_collected")
        if value_issue := _source_value_mapping_issue(annotation, candidate, packet):
            issues.append(value_issue)
        if candidate.coverage_limits:
            issues.append(
                Issue(code="candidate_coverage_incomplete", refs=(candidate.ref,))
            )
        for selector in leaves(annotation.record_selector.value):
            if isinstance(selector, (RecordEq, RecordIn)):
                evidence(selector.evidence_ids)
                if candidate.ref not in selector.evidence_ids:
                    raise WorkflowError("selector_evidence_mismatch")
                values = (
                    (selector.value,)
                    if isinstance(selector, RecordEq)
                    else selector.values
                )
                for value in values:
                    literal(candidate, selector.variable_ref.id, value)
        if annotation.record_selector.status == "unknown":
            issues.append(Issue(code="selector_unresolved", refs=(annotation.key,)))
        # Topic/value-level constraints must not disappear just to pass the checker.
        topic = [
            r
            for r in candidate.rows
            if r.values().get("role") == "Topic" and r.values().get("assigned_value")
        ]
        if (
            topic
            and annotation.record_selector.status == "not_applicable"
            and annotation.variable not in {r.variable for r in topic}
        ):
            issues.append(Issue(code="topic_selector_missing", refs=(annotation.key,)))
        for assignment in annotation.fixed_assignments:
            evidence(assignment.evidence_refs)
            if candidate.ref not in assignment.evidence_refs:
                raise WorkflowError("assignment_evidence_mismatch")
            literal(candidate, assignment.variable, assignment.value)
            target_row = candidate.row(assignment.variable).values()
            if target_row.get("origin_type") == "Collected":
                source_values = [
                    packet.object(ref)
                    for ref in assignment.evidence_refs
                    if ref in visible
                ]
                if not any(
                    source.entry_mode == "printed_context"
                    and source.text.strip() == str(assignment.value)
                    for source in source_values
                ):
                    raise WorkflowError("collected_constant_without_source")
    for supplemental in decision.supplemental:
        evidence(supplemental.evidence_refs)
        if not set(supplemental.source_scope) <= set(packet.output_scope):
            raise WorkflowError("output_scope_denied")
        issues.append(
            Issue(code="supplemental_review_required", refs=supplemental.source_scope)
        )
    for relationship in decision.relationships:
        evidence(relationship.evidence_refs)
        issues.append(
            Issue(
                code="relationship_review_required",
                refs=(
                    relationship.from_observation_key,
                    relationship.to_observation_key,
                ),
            )
        )
    for request in decision.evidence_requests:
        if not set(request.affected_refs) <= visible:
            raise WorkflowError("unknown_gap_scope")
        issues.append(Issue(code="evidence_request_open", refs=request.affected_refs))
    covered = set(covered_source_refs(decision, packet))
    for ref in sorted(set(packet.output_scope) - covered):
        issues.append(Issue(code="source_not_covered", refs=(ref,)))
    for ref in sorted(covered):
        outcomes = [
            annotation
            for annotation in decision.annotations
            if ref in annotation.source_scope
        ]
        if (
            any(
                annotation.source_condition.status == "not_applicable"
                for annotation in outcomes
            )
            and len({annotation.disposition for annotation in outcomes}) > 1
        ):
            issues.append(Issue(code="overlapping_branch_dispositions", refs=(ref,)))
    for condition in understanding.conditions.entries:
        for ref in condition.affected_refs:
            source = packet.object(ref)
            mapped = [
                annotation
                for annotation in decision.annotations
                if annotation.disposition == "mapped"
                and (
                    ref in annotation.source_scope
                    or (
                        source.kind == "option"
                        and source.parent_ref in annotation.source_scope
                    )
                )
            ]
            if any(
                annotation.source_condition.status != "specified"
                for annotation in mapped
            ):
                issues.append(Issue(code="source_condition_not_preserved", refs=(ref,)))
    for binding in understanding.bindings:
        # References can intentionally connect different observations; equality
        # is an invariant only for unit/time qualification of the same subject.
        if binding.relation not in {"unit_for", "time_of"}:
            continue
        source_obs = {
            a.observation_key
            for a in decision.annotations
            if binding.source_ref in a.source_scope and a.observation_key
        }
        subject_obs = {
            a.observation_key
            for a in decision.annotations
            if set(binding.target_source_refs) & set(a.source_scope)
            and a.observation_key
        }
        if source_obs and subject_obs and not source_obs <= subject_obs:
            raise WorkflowError("qualifier_observation_conflict")
    if not packet.study_standard_confirmed:
        issues.append(Issue(code="study_standard_unconfirmed"))
    issues.extend(option_coverage(decision, packet))
    return tuple(issues)
