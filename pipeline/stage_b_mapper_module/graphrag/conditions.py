"""Immutable, disjoint source-control and versioned-record predicate ASTs."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from .field_contracts import FrozenModel

Identifier = Annotated[StrictStr, Field(min_length=1)]
Scalar: TypeAlias = (
    StrictStr
    | StrictBool
    | StrictInt
    | Annotated[StrictFloat, Field(allow_inf_nan=False)]
)


class TargetRef(FrozenModel):
    """Select a full actual binding or standard-variable occurrence ID."""

    kind: Literal["binding", "variable"]
    id: Identifier


class Predicate(FrozenModel):
    """Bound every parsed tree independently, including direct class construction."""

    @model_validator(mode="after")
    def bounded_tree(self) -> Predicate:
        """Reject depth over eight or more than 64 nodes; do not rewrite predicates."""
        pending: list[tuple[Predicate, int]] = [(self, 1)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 64 or depth > 8:
                raise ValueError("condition_limit_exceeded")
            args = getattr(node, "args", ())
            pending.extend((arg, depth + 1) for arg in args)
            arg = getattr(node, "arg", None)
            if arg is not None:
                pending.append((arg, depth + 1))
        return self


class SourceEq(Predicate):
    """Compare a reviewed control with one literal value."""

    op: Literal["eq"]
    source_id: Identifier
    value: Scalar


class SourceIn(Predicate):
    """Compare a reviewed control with an explicit nonempty set of literals."""

    op: Literal["in"]
    source_id: Identifier
    values: tuple[Scalar, ...] = Field(min_length=1)


class SourceAll(Predicate):
    """Conjoin source conditions without losing scope."""

    op: Literal["all"]
    args: tuple[SourceCondition, ...] = Field(min_length=1)


class SourceAny(Predicate):
    """Disjoin explicit source conditions, never text split on English words."""

    op: Literal["any"]
    args: tuple[SourceCondition, ...] = Field(min_length=1)


class SourceNot(Predicate):
    """Negate one source predicate while preserving its exact leaf identity."""

    op: Literal["not"]
    arg: SourceCondition


SourceCondition: TypeAlias = Annotated[
    SourceEq | SourceIn | SourceAll | SourceAny | SourceNot, Field(discriminator="op")
]


class RecordEq(Predicate):
    """Select one target-variable value with independent semantic evidence IDs."""

    op: Literal["eq"]
    variable_ref: TargetRef
    value: Scalar
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class RecordIn(Predicate):
    """Select an explicit target-variable value set, distinct from source controls."""

    op: Literal["in"]
    variable_ref: TargetRef
    values: tuple[Scalar, ...] = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class RecordAll(Predicate):
    """Conjoin target predicates while retaining every evidence-backed leaf."""

    op: Literal["all"]
    args: tuple[RecordSelector, ...] = Field(min_length=1)


class RecordAny(Predicate):
    """Disjoin target predicates; no implicit assignment is introduced."""

    op: Literal["any"]
    args: tuple[RecordSelector, ...] = Field(min_length=1)


class RecordNot(Predicate):
    """Negate one target predicate explicitly."""

    op: Literal["not"]
    arg: RecordSelector


RecordSelector: TypeAlias = Annotated[
    RecordEq | RecordIn | RecordAll | RecordAny | RecordNot, Field(discriminator="op")
]
for _model in (SourceAll, SourceAny, SourceNot, RecordAll, RecordAny, RecordNot):
    _model.model_rebuild()


def parse_source_condition(value: object) -> SourceCondition:
    """Parse a source AST at a boundary, raising ValidationError on malformed input."""
    return TypeAdapter(SourceCondition).validate_python(value)


def leaves(
    condition: SourceCondition | RecordSelector | None,
) -> tuple[SourceEq | SourceIn | RecordEq | RecordIn, ...]:
    """Return original leaves in order without interpreting boolean relationships."""
    if condition is None:
        return ()
    if isinstance(condition, (SourceEq, SourceIn, RecordEq, RecordIn)):
        return (condition,)
    if isinstance(condition, (SourceNot, RecordNot)):
        return leaves(condition.arg)
    return tuple(leaf for arg in condition.args for leaf in leaves(arg))
