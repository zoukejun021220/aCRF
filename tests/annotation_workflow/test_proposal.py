"""Compiler regressions use real pinned templates and deliberately wrong judgments."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
from support import RULES, SNAPSHOT, SYSBP, decision, packet, understanding

from pipeline.stage_b_mapper_module.annotation_workflow.checks import check_decision
from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.evaluation import compare
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.proposal import (
    Proposal,
    compile_proposal,
    proposal_context,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import wire_schema
from pipeline.stage_b_mapper_module.annotation_workflow.source import Material
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
)


def ordinary() -> dict:
    return {
        "records": [
            {
                "template": "T1",
                "bindings": [
                    {"source": ["bp"], "target": "VSORRES"},
                    {"source": ["unit"], "target": "VSORRESU"},
                    {"source": ["date"], "target": "VSDTC"},
                ],
                "reason": "The result, printed unit and date belong to one systolic measurement.",
            }
        ],
        "other_outcomes": [],
        "open_questions": [],
    }


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = Knowledge(SNAPSHOT, packet().standard, RULES)
        cls.candidates = (cls.kb.resolve(SYSBP),)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.kb.close()

    def compile(self, raw: dict):
        return compile_proposal(
            Proposal.model_validate(raw), packet(), understanding(), self.candidates
        )

    def test_simple_model_answer_compiles_to_exact_existing_engineering_facts(self):
        result = self.compile(ordinary())
        self.assertTrue(compare(result, decision()).exact)
        self.assertEqual(len(result.annotations), 3)
        self.assertEqual(
            {a.variable for a in result.annotations}, {"VSORRES", "VSORRESU", "VSDTC"}
        )
        self.assertEqual(
            [
                issue.code
                for issue in check_decision(
                    result, packet(), understanding(), self.candidates
                )
            ],
            ["study_standard_unconfirmed"],
        )
        self.assertEqual(result.decisions[0].item_refs, ("item:1", "item:2", "item:3"))
        self.assertIn(SYSBP, result.target_observations[0].evidence_refs)
        self.assertFalse(any(a.variable == "VSTESTCD" for a in result.annotations))

    def test_null_mapping_and_backend_item_ids_are_not_model_contract(self):
        raw = ordinary()
        raw["records"][0]["bindings"][0]["target"] = None
        with self.assertRaises(ValidationError):
            Proposal.model_validate(raw)
        raw = ordinary()
        raw["records"][0]["item_refs"] = ["backend item:1"]
        with self.assertRaises(ValidationError):
            Proposal.model_validate(raw)

    def test_actionable_alias_scope_and_target_errors(self):
        for field, replacement, code, path in (
            (
                "template",
                "backend T1",
                "proposal_template_unknown",
                "/records/0/template",
            ),
            (
                "target",
                "VSSTRESN",
                "proposal_target_invalid",
                "/records/0/bindings/0/target",
            ),
            (
                "source",
                ["not-delivered"],
                "proposal_source_scope_invalid",
                "/records/0/bindings",
            ),
        ):
            raw = ordinary()
            target = (
                raw["records"][0]
                if field == "template"
                else raw["records"][0]["bindings"][0]
            )
            target[field] = replacement
            with self.subTest(field=field), self.assertRaises(WorkflowError) as raised:
                self.compile(raw)
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.feedback[0]["path"], path)
            self.assertTrue(raised.exception.feedback[0]["allowed"])

    def test_source_constants_require_actual_printed_value(self):
        for mode in ("entered", "prefilled", "unknown", "instructional_example"):
            source = packet()
            source = source.model_copy(
                update={
                    "objects": tuple(
                        (
                            item.model_copy(update={"entry_mode": mode})
                            if item.ref == "unit"
                            else item
                        )
                        for item in source.objects
                    )
                }
            )
            result = compile_proposal(
                Proposal.model_validate(ordinary()),
                source,
                understanding(),
                self.candidates,
            )
            self.assertFalse(any(a.fixed_assignments for a in result.annotations))
        self.assertEqual(
            self.compile(ordinary()).annotations[1].fixed_assignments[0].value, "mmHg"
        )

    def test_model_still_chooses_grouping_checker_rejects_wrong_unit_group(self):
        raw = ordinary()
        unit = raw["records"][0]["bindings"].pop(1)
        raw["records"].append(
            {
                "template": "T1",
                "bindings": [unit],
                "reason": "A separate observation is proposed.",
            }
        )
        result = self.compile(raw)
        self.assertEqual(len(result.target_observations), 2)
        with self.assertRaises(WorkflowError) as raised:
            check_decision(result, packet(), understanding(), self.candidates)
        self.assertEqual(raised.exception.code, "qualifier_observation_conflict")

    def test_inherited_unknown_and_source_conditions_never_disappear(self):
        raw_understanding = understanding().model_dump(mode="json")
        raw_understanding["conditions"] = {"status": "unknown", "entries": []}
        unknown = CollectionUnderstanding.model_validate(raw_understanding)
        result = compile_proposal(
            Proposal.model_validate(ordinary()), packet(), unknown, self.candidates
        )
        self.assertTrue(
            all(a.source_condition.status == "unknown" for a in result.annotations)
        )
        raw_understanding["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": ["bp", "unit", "date"],
                    "kind": "collection",
                    "predicate": {"op": "eq", "source_id": "bp", "value": "Yes"},
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["bp"],
                        "reason": "Synthetic source branch.",
                    },
                }
            ],
        }
        understood = CollectionUnderstanding.model_validate(raw_understanding)
        result = compile_proposal(
            Proposal.model_validate(ordinary()), packet(), understood, self.candidates
        )
        self.assertTrue(
            all(
                a.source_condition.value == understood.conditions.entries[0].predicate
                for a in result.annotations
            )
        )
        proposal = ordinary()
        proposal["records"][0]["bindings"][0]["when"] = "unknown"
        with self.assertRaises(WorkflowError) as raised:
            compile_proposal(
                Proposal.model_validate(proposal), packet(), understood, self.candidates
            )
        self.assertEqual(raised.exception.code, "proposal_condition_weakened")

    def test_explicit_branches_and_nonmapping_dispositions_remain_expressible(self):
        raw = ordinary()
        raw["records"][0]["bindings"] = [
            {
                "source": ["bp"],
                "target": "VSORRES",
                "when": {"op": "eq", "source_id": "bp", "value": "Yes"},
            }
        ]
        raw["other_outcomes"] = [
            {
                "source": ["bp"],
                "outcome": "not_submitted",
                "reason": "No branch requires study confirmation.",
                "details": {"when": {"op": "eq", "source_id": "bp", "value": "No"}},
            },
            {
                "source": ["unit"],
                "outcome": "context_only",
                "reason": "Unit is retained as source context.",
                "details": {"context_targets": ["bp"]},
            },
            {
                "source": ["date"],
                "outcome": "unresolved",
                "reason": "The date's submission use is uncertain.",
            },
        ]
        result = self.compile(raw)
        self.assertEqual(
            [a.disposition for a in result.annotations],
            ["mapped", "not_submitted", "context_only", "unresolved"],
        )
        self.assertNotEqual(
            result.annotations[0].source_condition,
            result.annotations[1].source_condition,
        )
        self.assertEqual(result.annotations[2].context_targets, ("bp",))
        issues = check_decision(
            result,
            packet(),
            understanding().model_copy(update={"bindings": ()}),
            self.candidates,
        )
        self.assertIn("submission_basis_unresolved", {item.code for item in issues})

    def test_selected_extra_values_require_support_and_remain_separate_from_source_conditions(
        self,
    ):
        raw = ordinary()
        raw["records"][0]["bindings"][0]["details"] = {
            "selectors": [{"target": "VSPOS", "value": "SITTING", "support": ["bp"]}]
        }
        result = self.compile(raw)
        self.assertEqual(
            result.annotations[0].source_condition.status, "not_applicable"
        )
        self.assertEqual(result.annotations[0].record_selector.value.op, "all")
        raw["records"][0]["bindings"][0]["details"]["selectors"][0]["support"] = ["T1"]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "proposal_value_support_missing")

    def test_template_allowed_value_never_becomes_a_constant_without_source_choice(
        self,
    ):
        result = self.compile(ordinary())
        self.assertEqual(
            [
                (x.variable, x.value)
                for a in result.annotations
                for x in a.fixed_assignments
            ],
            [("VSORRESU", "mmHg")],
        )
        raw = ordinary()
        raw["records"][0]["bindings"][0]["details"] = {
            "constants": [{"target": "VSORRES", "value": "120", "support": ["bp"]}]
        }
        result = self.compile(raw)
        with self.assertRaises(WorkflowError) as raised:
            check_decision(result, packet(), understanding(), self.candidates)
        self.assertEqual(raised.exception.code, "collected_constant_without_source")

    def test_material_support_is_limited_to_delivered_same_study(self):
        material = Material(
            ref="study-policy",
            kind="study",
            study_id=packet().study_id,
            text="Synthetic policy",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        raw = {
            "records": [],
            "other_outcomes": [
                {
                    "source": ["bp", "unit", "date"],
                    "outcome": "not_submitted",
                    "reason": "Study policy confirms non-submission.",
                    "details": {"support": [material.ref]},
                }
            ],
            "open_questions": [],
        }
        with self.assertRaises(WorkflowError):
            self.compile(raw)
        result = compile_proposal(
            Proposal.model_validate(raw),
            packet(),
            understanding(),
            self.candidates,
            materials=(material,),
        )
        self.assertNotIn(
            "submission_basis_unresolved",
            {
                item.code
                for item in check_decision(
                    result, packet(), understanding(), self.candidates, (material,)
                )
            },
        )
        with self.assertRaises(WorkflowError):
            compile_proposal(
                Proposal.model_validate(raw),
                packet(),
                understanding(),
                self.candidates,
                materials=(material.model_copy(update={"study_id": "foreign"}),),
            )

    def test_supplemental_and_links_do_not_fabricate_qualifier_names_or_record_ids(
        self,
    ):
        raw = ordinary()
        raw["records"][0]["bindings"] = [{"source": ["bp"], "target": "VSORRES"}]
        raw["records"][0]["links"] = [
            {
                "record": 2,
                "relation": "related_measurement",
                "support": ["measurement"],
                "reason": "Synthetic cross-record relation for review.",
            }
        ]
        raw["records"].append(
            {
                "template": "T1",
                "bindings": [{"source": ["date"], "target": "VSDTC"}],
                "reason": "A separate source event is proposed for review.",
            }
        )
        raw["other_outcomes"] = [
            {
                "outcome": "supplemental",
                "source": ["unit"],
                "parent_record": 1,
                "qnam": None,
                "qlabel": None,
                "support": ["measurement"],
                "reason": "The qualifier name requires study evidence.",
            }
        ]
        result = self.compile(raw)
        self.assertEqual(result.supplemental[0].missing_slots, ("qnam", "qlabel"))
        self.assertEqual(
            result.supplemental[0].parent_observation_key,
            result.target_observations[0].key,
        )
        self.assertEqual(
            result.relationships[0].to_observation_key,
            result.target_observations[1].key,
        )
        self.assertEqual(result.annotations[-1].disposition, "unresolved")
        raw["records"][0]["links"][0]["record"] = 3
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "proposal_record_reference_invalid")

    def test_choices_are_explicit_and_wire_schema_has_no_nullable_ordinary_targets(
        self,
    ):
        context = proposal_context(packet(), understanding(), self.candidates)
        template = context["templates"][0]
        self.assertEqual(template["alias"], "T1")
        self.assertIn("VSORRES", template["targets"])
        self.assertNotIn("VSSTRESN", template["targets"])
        self.assertNotIn("source_sha256", template)
        schema = wire_schema(Proposal)
        self.assertEqual(
            schema["$defs"]["Binding"]["properties"]["target"]["type"], "string"
        )
        self.assertEqual(
            set(schema["$defs"]["Binding"]["properties"]), {"source", "target"}
        )
        self.assertEqual(
            Proposal.model_validate(ordinary()).model_dump(
                mode="json", exclude_defaults=True
            ),
            ordinary(),
        )

    def test_distinct_record_selectors_preserve_model_grouping(self):
        raw = ordinary()
        raw["records"] = [
            {
                "template": "T1",
                "bindings": [
                    {
                        "source": ["bp"],
                        "target": "VSORRES",
                        "details": {
                            "selectors": [
                                {
                                    "target": "VSPOS",
                                    "value": position,
                                    "support": ["measurement"],
                                }
                            ]
                        },
                    }
                ],
                "reason": f"The source measurement applies to the {position} record.",
            }
            for position in ("SITTING", "STANDING")
        ]
        result = self.compile(raw)
        self.assertEqual(len(set(item.key for item in result.target_observations)), 2)
        raw["records"][1] = raw["records"][0]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "proposal_duplicate_record")

    def test_only_model_reasons_are_attributed_and_empty_proposal_is_actionable(self):
        raw = ordinary()
        self.assertEqual(
            self.compile(raw).explanation.summary, raw["records"][0]["reason"]
        )
        with self.assertRaises(WorkflowError) as raised:
            self.compile({"records": [], "other_outcomes": [], "open_questions": []})
        self.assertEqual(raised.exception.code, "proposal_empty")
        self.assertEqual(
            raised.exception.feedback[0]["affected_refs"], list(packet().output_scope)
        )

    def test_knowledge_question_alias_is_resolved_without_mutating_model_reply(self):
        raw = ordinary()
        raw["open_questions"] = [
            {
                "missing": "Template applicability",
                "affected_refs": ["bp"],
                "impact": "Determines whether this template covers the measurement.",
                "source_kind": "standard",
                "material_ref": "T1",
            }
        ]
        proposal = Proposal.model_validate(raw)
        result = compile_proposal(proposal, packet(), understanding(), self.candidates)
        self.assertEqual(result.evidence_requests[0].material_ref, SYSBP)
        self.assertEqual(proposal.open_questions[0].material_ref, "T1")

    def test_source_handle_cannot_be_shadowed_by_template_alias(self):
        source = packet()
        source = source.model_copy(
            update={
                "objects": (
                    *source.objects,
                    source.objects[0].model_copy(update={"ref": "T1"}),
                )
            }
        )
        context = proposal_context(source, understanding(), self.candidates)
        self.assertEqual(context["templates"][0]["alias"], "T2")
        raw = ordinary()
        raw["records"][0]["template"] = "T2"
        raw["records"][0]["bindings"][0]["details"] = {"support": ["T1"]}
        result = compile_proposal(
            Proposal.model_validate(raw), source, understanding(), self.candidates
        )
        self.assertIn("T1", result.annotations[0].evidence_refs)
        self.assertIn(SYSBP, result.annotations[0].evidence_refs)

    def test_study_material_named_t1_is_evidence_and_question_handle(self):
        material = Material(
            ref="T1",
            kind="study",
            study_id=packet().study_id,
            text="Synthetic non-submission policy",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        context = proposal_context(
            packet(), understanding(), self.candidates, materials=(material,)
        )
        self.assertEqual(context["templates"][0]["alias"], "T2")
        raw = {
            "records": [],
            "other_outcomes": [
                {
                    "source": ["bp", "unit", "date"],
                    "outcome": "not_submitted",
                    "reason": "Study policy supplies the submission disposition.",
                    "details": {"support": ["T1"]},
                }
            ],
            "open_questions": [
                {
                    "missing": "Policy details",
                    "affected_refs": ["bp"],
                    "impact": "Confirm the policy applies to this source.",
                    "source_kind": "study",
                    "material_ref": "T1",
                }
            ],
        }
        result = compile_proposal(
            Proposal.model_validate(raw),
            packet(),
            understanding(),
            self.candidates,
            materials=(material,),
        )
        self.assertIn("T1", result.annotations[0].evidence_refs)
        self.assertNotIn(SYSBP, result.annotations[0].evidence_refs)
        self.assertEqual(result.evidence_requests[0].material_ref, "T1")
        self.assertNotIn(
            "submission_basis_unresolved",
            {
                issue.code
                for issue in check_decision(
                    result, packet(), understanding(), self.candidates, (material,)
                )
            },
        )
        # Unread material questions also cannot be converted merely because their
        # literal name resembles a template alias in an otherwise empty index.
        raw["other_outcomes"][0]["details"] = {}
        result = self.compile(raw)
        self.assertEqual(result.evidence_requests[0].material_ref, "T1")

    def test_unread_reserved_material_keeps_alias_stable_when_delivered(self):
        material = Material(
            ref="T1",
            kind="source",
            study_id=packet().study_id,
            text="Synthetic admitted completion guidance",
            source_ref=packet().objects[0].source_ref,
            origin="completion_guidance",
        )
        before = proposal_context(
            packet(), understanding(), self.candidates, reserved_refs=("T1",)
        )
        after = proposal_context(
            packet(),
            understanding(),
            self.candidates,
            reserved_refs=("T1",),
            materials=(material,),
        )
        self.assertEqual(before["templates"], after["templates"])
        self.assertEqual(before["templates"][0]["alias"], "T2")
        raw = ordinary()
        raw["records"][0]["template"] = "T2"
        proposal = Proposal.model_validate(raw)
        first = compile_proposal(
            proposal, packet(), understanding(), self.candidates, reserved_refs=("T1",)
        )
        second = compile_proposal(
            proposal,
            packet(),
            understanding(),
            self.candidates,
            reserved_refs=("T1",),
            materials=(material,),
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
