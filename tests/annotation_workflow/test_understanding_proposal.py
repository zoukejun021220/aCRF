"""Regressions for source interpretation versus backend-owned source facts."""

from __future__ import annotations

import unittest
from typing import Any

from pydantic import ValidationError
from support import packet

from pipeline.stage_b_mapper_module.annotation_workflow.common import \
    WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.source import \
    SourcePacket
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import \
    queries
from pipeline.stage_b_mapper_module.annotation_workflow.understanding_proposal import (
    UnderstandingProposal, compile_understanding, understanding_context)


def ordinary() -> dict[str, Any]:
    """Author a new wire answer independently of the backend's stored objects."""
    return {
        "items": [
            {
                "source_ref": ref,
                "concept": concept,
                "collected_property": prop,
                "source_role": role,
                "basis": {
                    "kind": "explicit",
                    "source_refs": [ref, "measurement"],
                    "reason": reason,
                },
                "concept_context": [],
            }
            for ref, concept, prop, role, reason in (
                (
                    "bp",
                    "Systolic blood pressure",
                    "measurement result",
                    "measurement_result",
                    "The label identifies a systolic result within one measurement entry.",
                ),
                (
                    "unit",
                    "Measurement unit",
                    "unit",
                    "unit",
                    "The mmHg label supplies the unit of that measurement.",
                ),
                (
                    "date",
                    "Measurement date",
                    "date",
                    "time",
                    "The date label specifies measurement time.",
                ),
            )
        ],
        "bindings": [
            {
                "source_ref": ref,
                "relation": relation,
                "target_source_refs": ["bp"],
                "source_context_ref": "measurement",
                "basis": {
                    "kind": "explicit",
                    "source_refs": ["measurement"],
                    "reason": "The source links the qualifier to this measurement.",
                },
            }
            for ref, relation in (("unit", "unit_for"), ("date", "time_of"))
        ],
        "conditions": {"status": "none_stated_in_supplied_source", "entries": []},
        "material_unknowns": [],
        "explanation": {
            "summary": "The form collects a systolic result with a unit and date.",
            "evidence_refs": ["measurement", "bp", "unit", "date"],
            "uncertainties": [],
        },
    }


class UnderstandingProposalTests(unittest.TestCase):
    def compile(self, raw: dict[str, Any], source: SourcePacket | None = None):
        return compile_understanding(
            UnderstandingProposal.model_validate(raw), source or packet()
        )

    def test_model_wire_cannot_restate_mode_or_merge_subjects(self):
        properties = UnderstandingProposal.model_json_schema()["$defs"][
            "InterpretedItem"
        ]["properties"]
        self.assertIn("source_ref", properties)
        self.assertNotIn("source_refs", properties)
        self.assertNotIn("value_mode", properties)
        for obsolete in (
            {"value_mode": "entered"},
            {"source_refs": ["bp", "measurement"]},
        ):
            raw = ordinary()
            raw["items"][0].update(obsolete)
            with self.subTest(obsolete=obsolete), self.assertRaises(ValidationError):
                UnderstandingProposal.model_validate(raw)
        raw = ordinary()
        raw["items"][0]["source_ref"] = ["bp", "measurement"]
        with self.assertRaises(ValidationError):
            UnderstandingProposal.model_validate(raw)

    def test_backend_preserves_every_source_mode_including_unknown(self):
        for mode in (
            "entered",
            "printed_context",
            "prefilled",
            "instructional_example",
            "unknown",
        ):
            source_raw = packet().model_dump(mode="json")
            next(o for o in source_raw["objects"] if o["ref"] == "unit")[
                "entry_mode"
            ] = mode
            source = SourcePacket.model_validate(source_raw)
            with self.subTest(mode=mode):
                result = self.compile(ordinary(), source)
                self.assertEqual(result.items[1].value_mode, mode)
                self.assertEqual(result.items[0].value_mode, "entered")

    def test_printed_support_stays_evidence_not_a_second_entered_subject(self):
        result = self.compile(ordinary())
        self.assertEqual(
            [i.source_refs for i in result.items], [("bp",), ("unit",), ("date",)]
        )
        self.assertIn("measurement", result.items[0].basis.source_refs)
        self.assertEqual(result.items[1].value_mode, "printed_context")
        retrieval = queries(result, packet())
        self.assertEqual(len(retrieval), 1)
        self.assertEqual(retrieval[0].source_refs, ("bp",))
        self.assertEqual(
            {(context.name, context.value) for context in retrieval[0].context},
            # Source-label evidence retains qualifiers without adding another
            # entered subject or expanding the annotation scope.
            {
                ("unit_for", "mmHg"),
                ("time_of", "Date of measurement"),
                ("source_label", "Systolic blood pressure"),
            },
        )

    def test_context_subject_is_rejected_unless_explicitly_scoped(self):
        raw = ordinary()
        raw["items"].append(
            {
                "source_ref": "measurement",
                "concept": "Measurement context",
                "collected_property": "context",
                "source_role": "qualifier_context",
                "basis": {
                    "kind": "explicit",
                    "source_refs": ["measurement"],
                    "reason": "The instruction identifies one measurement entry.",
                },
            }
        )
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        feedback = raised.exception.feedback[0]
        self.assertEqual(raised.exception.code, "understanding_subject_scope_invalid")
        self.assertEqual(feedback["path"], "/items/3/source_ref")
        self.assertEqual(feedback["affected_refs"], ["measurement"])
        self.assertEqual(feedback["allowed_refs"], ["bp", "unit", "date"])
        expanded = packet().model_copy(
            update={"output_scope": (*packet().output_scope, "measurement")}
        )
        self.assertEqual(
            self.compile(raw, expanded).items[-1].source_refs, ("measurement",)
        )

    def test_missing_subject_cannot_be_hidden_in_a_material_question(self):
        raw = ordinary()
        raw["items"].pop(1)
        raw["material_unknowns"] = [
            {
                "missing": "unit meaning",
                "affected_refs": ["unit"],
                "impact": "May change the unit interpretation.",
                "source_kind": "source",
                "material_ref": None,
            }
        ]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "understanding_missing_subject")
        self.assertEqual(raised.exception.feedback[0]["path"], "/items")
        self.assertEqual(raised.exception.feedback[0]["affected_refs"], ["unit"])

    def test_explicit_unknown_judgment_is_retained_without_inventing_meaning(self):
        raw = ordinary()
        raw["items"][0].update(
            {
                "concept": None,
                "collected_property": None,
                "source_role": "unknown",
                "basis": {
                    "kind": "unknown",
                    "source_refs": [],
                    "reason": "The supplied label does not establish the concept.",
                },
            }
        )
        raw["conditions"] = {"status": "unknown", "entries": []}
        result = self.compile(raw)
        self.assertEqual(result.items[0].source_refs, ("bp",))
        self.assertIsNone(result.items[0].concept)
        self.assertEqual(result.items[0].source_role, "unknown")
        self.assertEqual(result.items[0].value_mode, "entered")
        self.assertEqual(result.conditions.status, "unknown")
        self.assertEqual(queries(result, packet()), ())

    def test_duplicate_subjects_require_correction_instead_of_deduplication(self):
        raw = ordinary()
        raw["items"].append({**raw["items"][0], "source_role": "administrative"})
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "understanding_duplicate_subject")
        self.assertEqual(raised.exception.feedback[0]["path"], "/items/3/source_ref")

    def test_options_can_rely_on_scoped_parent_but_not_on_visible_parent_only(self):
        source_raw = packet().model_dump(mode="json")
        source_raw["objects"].append(
            {
                **source_raw["objects"][1],
                "ref": "bp:option",
                "kind": "option",
                "text": "Not assessed",
                "parent_ref": "bp",
            }
        )
        source_raw["output_scope"].append("bp:option")
        source = SourcePacket.model_validate(source_raw)
        self.assertEqual(len(self.compile(ordinary(), source).items), 3)
        self.assertEqual(
            understanding_context(source)["required_subject_refs"],
            ["bp", "unit", "date"],
        )
        isolated = source.model_copy(update={"output_scope": ("bp:option",)})
        raw = ordinary()
        raw["items"] = []
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw, isolated)
        self.assertEqual(raised.exception.code, "understanding_missing_subject")
        self.assertEqual(raised.exception.feedback[0]["affected_refs"], ["bp:option"])

    def test_context_exposes_scope_separately_from_visible_source_facts(self):
        context = understanding_context(packet())
        self.assertEqual(
            [choice["source_ref"] for choice in context["subject_choices"]],
            ["bp", "unit", "date"],
        )
        self.assertIn("measurement", context["visible_source_refs"])
        unit = next(
            choice
            for choice in context["known_source_facts"]
            if choice["source_ref"] == "unit"
        )
        self.assertEqual(unit["entry_mode"], "printed_context")
        self.assertEqual(unit["datatype"], "unknown")

    def test_all_judgment_bases_require_visible_support_when_not_unknown(self):
        for location, expected_path in (
            ("item", "/items/0/basis/source_refs"),
            ("binding", "/bindings/0/basis/source_refs"),
        ):
            raw = ordinary()
            parent = raw["items"][0] if location == "item" else raw["bindings"][0]
            parent["basis"]["source_refs"] = []
            with self.subTest(location=location), self.assertRaises(
                WorkflowError
            ) as raised:
                self.compile(raw)
            self.assertEqual(raised.exception.code, "missing_source_basis")
            self.assertEqual(raised.exception.feedback[0]["path"], expected_path)
        raw = ordinary()
        raw["items"][0]["basis"]["source_refs"] = ["not-visible"]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(
            raised.exception.feedback[0]["path"], "/items/0/basis/source_refs/0"
        )

    def test_binding_handle_errors_identify_actual_field_and_allowed_sources(self):
        for field, replacement, path in (
            ("source_ref", "not-visible", "/bindings/0/source_ref"),
            ("target_source_refs", ["not-visible"], "/bindings/0/target_source_refs/0"),
            ("source_context_ref", "not-visible", "/bindings/0/source_context_ref"),
        ):
            raw = ordinary()
            raw["bindings"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(WorkflowError) as raised:
                self.compile(raw)
            self.assertEqual(raised.exception.code, "understanding_source_ref_invalid")
            self.assertEqual(raised.exception.feedback[0]["path"], path)
            self.assertIn("measurement", raised.exception.feedback[0]["allowed_refs"])

    def test_nested_conditions_preserve_branch_and_report_exact_invalid_leaf(self):
        raw = ordinary()
        raw["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": ["bp"],
                    "kind": "collection",
                    "predicate": {
                        "op": "not",
                        "arg": {
                            "op": "any",
                            "args": [
                                {
                                    "op": "eq",
                                    "source_id": "bp",
                                    "value": "Not assessed",
                                },
                                {
                                    "op": "in",
                                    "source_id": "date",
                                    "values": ["Unknown", "Missing"],
                                },
                            ],
                        },
                    },
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["measurement"],
                        "reason": "Synthetic collection instruction states this branch.",
                    },
                }
            ],
        }
        result = self.compile(raw)
        self.assertEqual(result.conditions.model_dump(mode="json"), raw["conditions"])
        raw["conditions"]["entries"][0]["predicate"]["arg"]["args"][1][
            "source_id"
        ] = "not-visible"
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(
            raised.exception.feedback[0]["path"],
            "/conditions/entries/0/predicate/arg/args/1/source_id",
        )
        self.assertEqual(raised.exception.feedback[0]["affected_refs"], ["bp"])

    def test_condition_status_and_condition_basis_are_not_silently_normalized(self):
        raw = ordinary()
        raw["conditions"]["status"] = "specified"
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "condition_status_conflict")
        raw["conditions"]["entries"] = [
            {
                "affected_refs": ["bp"],
                "kind": "visibility",
                "predicate": {"op": "eq", "source_id": "bp", "value": "Yes"},
                "basis": {
                    "kind": "explicit",
                    "source_refs": [],
                    "reason": "A branch is proposed.",
                },
            }
        ]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "missing_source_basis")
        self.assertEqual(
            raised.exception.feedback[0]["path"],
            "/conditions/entries/0/basis/source_refs",
        )

    def test_context_explanation_and_unknown_references_all_stay_source_bound(self):
        for location, expected in (
            ("context", "/items/0/concept_context/0/source_refs/0"),
            ("explanation", "/explanation/evidence_refs/0"),
            ("gap", "/material_unknowns/0/affected_refs/0"),
        ):
            raw = ordinary()
            if location == "context":
                raw["items"][0]["concept_context"] = [
                    {
                        "name": "position",
                        "value": "supine",
                        "source_refs": ["not-visible"],
                    }
                ]
            elif location == "explanation":
                raw["explanation"]["evidence_refs"] = ["not-visible"]
            else:
                raw["material_unknowns"] = [
                    {
                        "missing": "source detail",
                        "affected_refs": ["not-visible"],
                        "impact": "Changes interpretation.",
                        "source_kind": "source",
                        "material_ref": None,
                    }
                ]
            with self.subTest(location=location), self.assertRaises(
                WorkflowError
            ) as raised:
                self.compile(raw)
            self.assertEqual(raised.exception.code, "understanding_source_ref_invalid")
            self.assertEqual(raised.exception.feedback[0]["path"], expected)

    def test_explanation_material_support_requires_actual_delivery(self):
        raw = ordinary()
        raw["explanation"]["evidence_refs"].append("completion-guidance")
        proposed = UnderstandingProposal.model_validate(raw)
        with self.assertRaises(WorkflowError) as raised:
            compile_understanding(proposed, packet())
        self.assertEqual(
            raised.exception.feedback[0]["path"], "/explanation/evidence_refs/4"
        )
        self.assertNotIn(
            "completion-guidance", raised.exception.feedback[0]["allowed_refs"]
        )
        result = compile_understanding(
            proposed,
            packet(),
            delivered_material_refs=("completion-guidance",),
        )
        self.assertIn("completion-guidance", result.explanation.evidence_refs)

        # Delivered guidance can support the overall explanation, but is not a
        # source-control handle in a typed source basis or branch predicate.
        raw["items"][0]["basis"]["source_refs"] = ["completion-guidance"]
        with self.assertRaises(WorkflowError) as raised:
            compile_understanding(
                UnderstandingProposal.model_validate(raw),
                packet(),
                delivered_material_refs=("completion-guidance",),
            )
        self.assertEqual(
            raised.exception.feedback[0]["path"], "/items/0/basis/source_refs/0"
        )


if __name__ == "__main__":
    unittest.main()
