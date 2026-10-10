"""Deterministic complete review display and dependency-group review triggers."""

from __future__ import annotations

from .common import DecisionExplanation, canonical
from .decision import MappingDecision
from .knowledge import Candidate
from .source import SourcePacket
from .understanding import CollectionUnderstanding


def needs_consistency(
    value: CollectionUnderstanding, packet: SourcePacket, decision: MappingDecision
) -> bool:
    """Trigger semantic review for shared or cross-context grouping, even with valid IDs."""
    # Count actual source grouping boundaries. Compiler provenance may contain
    # several supporting refs for one observation without creating a dependency.
    contexts = {
        context
        for annotation in decision.annotations
        if annotation.disposition == "mapped"
        for ref in annotation.source_scope
        if (context := packet.object(ref).context_ref) is not None
    }
    return bool(
        decision.relationships
        or len(decision.target_observations) > 1
        or len(contexts) > 1
        or any(
            len(binding.target_source_refs) > 1 or binding.relation == "references"
            for binding in value.bindings
        )
        or any(
            item.parent_ref and item.parent_ref not in packet.output_scope
            for item in packet.objects
            if item.ref in packet.output_scope
        )
    )


def render(decision: MappingDecision, candidates: tuple[Candidate, ...]) -> str:
    """Display every disposition and partial proposal; never declare study approval.

    This is a review artifact, not final PDF placement. Canonical predicates are
    retained next to readable labels, so no condition is inferred from text.
    """
    observations = {item.key: item for item in decision.target_observations}
    by_ref = {item.ref: item for item in candidates}
    lines = ["# Annotation proposal — review required", ""]
    lines.extend(explanation_lines(decision.explanation))
    for selection in decision.decisions:
        lines.append(
            f"- Candidate {selection.decision} [{', '.join(selection.candidate_refs)}]: {selection.reason}"
        )
    for annotation in sorted(decision.annotations, key=lambda item: item.key):
        scope = ", ".join(annotation.source_scope)
        display = {
            "mapped": "",
            "not_submitted": "[NOT SUBMITTED]",
            "context_only": "Context only",
            "unresolved": "Unresolved",
        }[annotation.disposition]
        if annotation.disposition == "mapped":
            assert annotation.observation_key is not None
            observation = observations[annotation.observation_key]
            display = (
                by_ref[observation.candidate_ref].domain
                + "."
                + str(annotation.variable)
            )
            display += " [observation=" + annotation.observation_key + "]"
        if annotation.source_condition.status == "specified":
            assert annotation.source_condition.value is not None
            display += " source_if=" + canonical(
                annotation.source_condition.value.model_dump(mode="json")
            )
        elif annotation.source_condition.status == "unknown":
            display += " [source condition unresolved]"
        if annotation.record_selector.status == "specified":
            assert annotation.record_selector.value is not None
            display += " when=" + canonical(
                annotation.record_selector.value.model_dump(mode="json")
            )
        elif annotation.record_selector.status == "unknown":
            display += " [record selector unresolved]"
        for assignment in annotation.fixed_assignments:
            display += f" {assignment.variable}={canonical(assignment.value)}"
        lines.append(f"- {annotation.key} [{scope}]: {display}")
        if annotation.reason:
            lines.append("  " + annotation.reason)
    for supplemental in decision.supplemental:
        lines.append(
            "- SUPP proposal: " + canonical(supplemental.model_dump(mode="json"))
        )
    for relationship in decision.relationships:
        lines.append(
            "- Relationship proposal: "
            + canonical(relationship.model_dump(mode="json"))
        )
    for request in decision.evidence_requests:
        lines.append(
            "- Open evidence request: " + canonical(request.model_dump(mode="json"))
        )
    return "\n".join(lines) + "\n"


def explanation_lines(explanation: DecisionExplanation) -> list[str]:
    """Render supplied brief explanations and explicit evidence without inferring reasons."""
    return [
        "Decision explanation: " + explanation.summary,
        "Evidence: " + (", ".join(explanation.evidence_refs) or "None cited"),
        "Uncertainties: " + ("; ".join(explanation.uncertainties) or "None stated"),
        "",
    ]
