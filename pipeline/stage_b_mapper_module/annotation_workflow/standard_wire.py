"""Flat native wire shapes compiled into the full annotation semantics.

Avoid overlapping object unions in local grammar generation. Kind tags precede
payloads under canonical sorting; unused slots must be null/empty. Compilation
only renames represented slots, never supplies targets, facts or assignments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..graphrag.field_contracts import FrozenModel
from .common import BriefReason
from .standard_annotation import Condition, Missing, StandardAnnotation


class WireValue(FrozenModel):
    kind: Literal[
        "source", "ct_lookup", "constant", "iso8601_datetime", "iso8601_duration"
    ]
    literal: str | None
    source_field: str | None
    source_fields: tuple[str, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def slots(self):
        """Reject inactive payloads instead of guessing the intended variant."""
        if self.kind in {"source", "ct_lookup"}:
            valid = (
                self.source_field is not None
                and self.literal is None
                and not self.source_fields
            )
        elif self.kind == "constant":
            valid = (
                self.literal is not None
                and self.source_field is None
                and not self.source_fields
            )
        else:
            valid = (
                bool(self.source_fields)
                and self.literal is None
                and self.source_field is None
            )
        if not valid:
            # Expose variant-specific slot constraints to bounded correction;
            # this reports representation errors, never clinical answers.
            constraint = (
                f"{self.kind} requires source_field, literal=null, source_fields=[]"
                if self.kind in {"source", "ct_lookup"}
                else (
                    "constant requires literal, source_field=null, source_fields=[]"
                    if self.kind == "constant"
                    else f"{self.kind} requires source_fields, literal/source_field=null"
                )
            )
            raise ValueError("wire_value_payload_does_not_match_kind: " + constraint)
        return self

    def represented(self) -> dict:
        """Preserve every represented value with its semantic field names."""
        if self.kind in {"source", "ct_lookup"}:
            return {"kind": self.kind, "field_id": self.source_field}
        if self.kind == "constant":
            return {"kind": self.kind, "value": self.literal}
        return {"kind": self.kind, "field_ids": list(self.source_fields)}


class WireRecord(FrozenModel):
    kind: Literal["collection_record", "test", "instrument"]
    name: str | None
    test_code: str | None
    test_source_field: str | None

    @model_validator(mode="after")
    def slots(self):
        """Keep fixed, collected-test and assessment-wide bindings distinct."""
        valid = (
            (
                self.kind == "collection_record"
                and self.name is None
                and self.test_code is None
            )
            or (
                self.kind == "test"
                and self.test_code is not None
                and self.name is None
                and self.test_source_field is None
            )
            or (
                self.kind == "instrument"
                and self.name is not None
                and self.test_code is None
                and self.test_source_field is None
            )
        )
        if not valid:
            raise ValueError("wire_record_payload_does_not_match_kind")
        return self

    def represented(self) -> dict:
        """Rename slots only, without adding an implicit study convention."""
        if self.kind == "test":
            return {"kind": self.kind, "test_code": self.test_code}
        if self.kind == "instrument":
            return {"kind": self.kind, "instrument": self.name}
        return {
            "kind": self.kind,
            **(
                {"test_source_field": self.test_source_field}
                if self.test_source_field
                else {}
            ),
        }


class WireFact(FrozenModel):
    domain: str
    variable: str
    value: WireValue
    when: Condition | None
    record: WireRecord


class WireAnnotation(FrozenModel):
    decision: Literal[
        "mapped", "not_submitted", "partial", "needs_convention", "abstain"
    ]
    facts: tuple[WireFact, ...] = Field(max_length=32)
    missing: tuple[Missing, ...] = Field(max_length=8)
    not_submitted_when: tuple[Condition, ...] = Field(max_length=8)
    reason: BriefReason
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def semantic_shape(self):
        """Report impossible business states through the normal correction parser."""
        self.represented()
        return self

    def represented(self) -> StandardAnnotation:
        """Compile one-for-one; business and source validation stay authoritative."""
        return StandardAnnotation.model_validate(
            {
                "status": self.decision,
                "facts": [
                    {
                        "domain": f.domain,
                        "variable": f.variable,
                        "value": f.value.represented(),
                        "when": f.when.model_dump(mode="json") if f.when else None,
                        "record": f.record.represented(),
                    }
                    for f in self.facts
                ],
                "missing": self.missing,
                "not_submitted_when": [
                    c.model_dump(mode="json") for c in self.not_submitted_when
                ],
                "explanation": self.reason,
                "evidence_refs": self.evidence_refs,
            }
        )


WIRE_INSTRUCTIONS = """Wire format: decision is the annotation status; reason is its
brief evidence justification. A value uses exactly one kind and these slots:
source/ct_lookup: source_field = focused ID, literal = null, source_fields = [].
constant: literal = assigned string, source_field = null, source_fields = [].
An explicitly assigned fixed descriptor uses constant, even when its value is
corroborated by CT. ct_lookup requires a collected source field to look up.
ISO transforms: source_fields = component IDs, literal/source_field = null.
A record uses exactly one kind: test uses test_code; instrument uses name;
collection_record binds a neighboring collected TEST via test_source_field when
annotating its result or qualifier. When the focused field IS the collected TEST
name, use {"kind":"collection_record","name":null,"test_code":null,
"test_source_field":null} for both TEST/source and TESTCD/ct_lookup facts.
Unresolved panel performance also uses this all-null collection_record payload.
The test kind requires a non-null fixed public TESTCD; a lookup input is represented
by collection_record. Choose one value binding per target/record/condition, and
choose one of uncoded result mapping or partial missing coding_dictionary.
All unused record slots are null. when = null unless a delivered public mapping
explicitly activates a source condition. Arrays contain actual distinct facts,
never padding to maximum length. Do not repeat facts or fill neighboring fields.
Keep reason short; supply it after the decision, not as a reasoning preamble.
"""
