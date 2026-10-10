"""Finite option coverage without guessing missing values or cross-field conditions."""

from __future__ import annotations

from ..graphrag.conditions import (
    SourceAll,
    SourceAny,
    SourceCondition,
    SourceEq,
    SourceIn,
    SourceNot,
)
from .common import Issue
from .decision import MappingDecision
from .source import SourcePacket


def applies(
    condition: SourceCondition, source_id: str, values: tuple[str, ...]
) -> bool | None:
    """Evaluate one known option; foreign source predicates remain unknown."""
    if isinstance(condition, (SourceEq, SourceIn)):
        if condition.source_id != source_id:
            return None
        literals = (
            (condition.value,) if isinstance(condition, SourceEq) else condition.values
        )
        return any(value in literals for value in values)
    if isinstance(condition, SourceNot):
        result = applies(condition.arg, source_id, values)
        return None if result is None else not result
    results = tuple(applies(arg, source_id, values) for arg in condition.args)
    if isinstance(condition, SourceAll):
        return False if False in results else None if None in results else True
    assert isinstance(condition, SourceAny)
    return True if True in results else None if None in results else False


def option_coverage(
    decision: MappingDecision, packet: SourcePacket
) -> tuple[Issue, ...]:
    """Report missing and contradictory single-choice branches, keeping other scopes open.

    Multi-select combinations and conditions involving other fields require a
    separate semantic review; this finite check never asserts their completeness.
    """
    issues = []
    for option in packet.objects:
        if (
            option.kind != "option"
            or option.ref not in packet.output_scope
            or option.parent_ref is None
        ):
            continue
        parent = packet.object(option.parent_ref)
        if parent.datatype not in {"single_choice", "radio"}:
            issues.append(
                Issue(code="option_coverage_requires_review", refs=(option.ref,))
            )
            continue
        outcomes: set[str] = set()
        uncertain = False
        for annotation in decision.annotations:
            if not {option.ref, parent.ref} & set(annotation.source_scope):
                continue
            slot = annotation.source_condition
            active = (
                None
                if slot.status == "unknown"
                else (
                    True
                    if slot.value is None
                    else applies(
                        slot.value,
                        parent.ref,
                        (
                            option.text,
                            *(
                                [option.stored_code]
                                if option.stored_code is not None
                                else []
                            ),
                        ),
                    )
                )
            )
            if active is True:
                outcomes.add(annotation.disposition)
            elif active is None:
                uncertain = True
        if uncertain:
            issues.append(
                Issue(code="option_condition_requires_review", refs=(option.ref,))
            )
        if not outcomes:
            issues.append(Issue(code="option_branch_not_covered", refs=(option.ref,)))
        elif len(outcomes) > 1:
            issues.append(Issue(code="option_branch_conflict", refs=(option.ref,)))
    return tuple(issues)
