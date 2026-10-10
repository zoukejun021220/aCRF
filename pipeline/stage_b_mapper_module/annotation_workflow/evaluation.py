"""Lossless, separately invoked evaluation; the inference path never imports answers."""

from __future__ import annotations

from collections import Counter

from ..graphrag.field_contracts import FrozenModel
from .common import canonical
from .decision import MappingDecision


class Comparison(FrozenModel):
    """Exact engineering comparison is not independent clinical adjudication."""

    exact: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    label_authority: str = "engineering_expectation"
    expert_adjudication: str = "pending"


def semantic_rows(value: MappingDecision) -> tuple[str, ...]:
    """Normalize only local observation aliases and ordering, never repair facts.

    Reference provenance, conditions, extra facts and duplicate occurrences remain
    significant. Matching expert-approved alternatives is intentionally separate.
    """
    observations = {
        item.key: canonical(item.model_dump(mode="json", exclude={"key"}))
        for item in value.target_observations
    }
    rows = ["observation:" + item for item in observations.values()]
    for annotation in value.annotations:
        row = annotation.model_dump(mode="json", exclude={"key", "reason"})
        if annotation.observation_key:
            row["observation_key"] = observations[annotation.observation_key]
        rows.append("annotation:" + canonical(row))
    for supplemental in value.supplemental:
        row = supplemental.model_dump(mode="json")
        row["parent_observation_key"] = observations[
            supplemental.parent_observation_key
        ]
        rows.append("supplemental:" + canonical(row))
    for relationship in value.relationships:
        row = relationship.model_dump(mode="json")
        row["from_observation_key"] = observations[relationship.from_observation_key]
        row["to_observation_key"] = observations[relationship.to_observation_key]
        rows.append("relationship:" + canonical(row))
    rows.extend(
        "request:" + canonical(item.model_dump(mode="json"))
        for item in value.evidence_requests
    )
    return tuple(sorted(rows))


def compare(actual: MappingDecision, expected: MappingDecision) -> Comparison:
    """Count missing and additional full facts; duplicate errors cannot disappear."""
    actual_rows, expected_rows = Counter(semantic_rows(actual)), Counter(
        semantic_rows(expected)
    )
    missing = tuple(sorted((expected_rows - actual_rows).elements()))
    extra = tuple(sorted((actual_rows - expected_rows).elements()))
    return Comparison(exact=not missing and not extra, missing=missing, extra=extra)
