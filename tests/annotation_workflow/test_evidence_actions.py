"""Evidence actions validate references without claiming that questions are answered."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.evidence_actions import (
    DecisionTurn,
    QuestionLedger,
    QuestionUpdate,
    ReadMaterial,
    SearchStandard,
    SubmitDecision,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import wire_schema
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import Unknown


def read_action(**changes: object) -> ReadMaterial:
    return ReadMaterial.model_validate(
        {
            "op": "read_material",
            "handle": "source:page:1",
            "source": ["bp"],
            "question": "Which measurement does this unit qualify?",
            "impact": "Determines the result and unit association.",
            **changes,
        }
    )


class DecisionTurnTests(unittest.TestCase):
    def test_read_and_search_do_not_require_an_annotation_or_material_kind(self):
        read = DecisionTurn.model_validate(
            {"action": read_action().model_dump(mode="json")}
        )
        self.assertIsInstance(read.action, ReadMaterial)
        search = DecisionTurn.model_validate(
            {
                "action": {
                    "op": "search_standard",
                    "concept": "blood pressure",
                    "collected_property": "measurement result",
                    "source": ["bp"],
                    "question": "Which source-compatible definition is available?",
                    "impact": "Determines candidate applicability.",
                }
            }
        )
        self.assertIsInstance(search.action, SearchStandard)
        self.assertNotIn("records", search.action.model_dump())

    def test_discriminator_and_action_specific_fields_are_enforced(self):
        for changes in (
            {"op": "guess_standard"},
            {"source": []},
            {"source_kind": "study"},
            {"material_ref": "study:guessed-policy"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                DecisionTurn.model_validate(
                    {"action": {**read_action().model_dump(), **changes}}
                )
        raw = read_action().model_dump()
        del raw["op"]
        with self.assertRaises(ValidationError):
            DecisionTurn.model_validate({"action": raw})
        with self.assertRaises(ValidationError):
            DecisionTurn.model_validate({"action": read_action(), "action2": {}})

    def test_submit_keeps_business_fields_and_defaults_question_updates(self):
        turn = DecisionTurn.model_validate(
            {
                "action": {
                    "op": "submit",
                    "records": [],
                    "other_outcomes": [
                        {
                            "source": ["bp"],
                            "outcome": "unresolved",
                            "reason": "No applicable evidence was delivered.",
                        }
                    ],
                }
            }
        )
        self.assertIsInstance(turn.action, SubmitDecision)
        self.assertNotIn("open_questions", turn.action.model_dump())
        self.assertEqual(turn.action.question_updates, ())

    def test_submit_rejects_legacy_material_requests_in_local_model_and_wire(self):
        action = {
            "op": "submit",
            "records": [],
            "other_outcomes": [],
            "open_questions": [],
        }
        for model, value in (
            (SubmitDecision, action),
            (DecisionTurn, {"action": action}),
        ):
            with self.subTest(model=model.__name__), self.assertRaises(
                ValidationError
            ) as error:
                model.model_validate(value)
            self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")
        definition = wire_schema(DecisionTurn)["$defs"]["SubmitDecision"]
        self.assertNotIn("open_questions", definition["properties"])
        self.assertNotIn("open_questions", definition["required"])
        self.assertFalse(definition["additionalProperties"])

    def test_wire_schema_supports_all_three_op_variants(self):
        schema = wire_schema(DecisionTurn)
        alternatives = schema["properties"]["action"]["anyOf"]
        definitions = schema["$defs"]
        self.assertEqual(
            {
                definitions[item["$ref"].rsplit("/", 1)[1]]["properties"]["op"]["const"]
                for item in alternatives
            },
            {"read_material", "search_standard", "submit"},
        )


class QuestionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = QuestionLedger()

    def test_identity_uses_question_and_source_set_not_impact_or_selected_handle(self):
        first = read_action(source=["bp", "unit"])
        handle = self.ledger.observe(first, source_scope=("bp", "unit"))
        self.assertEqual(handle, "Q1")
        repeated = read_action(
            source=["unit", "bp"],
            handle="source:page:2",
            impact="Same business effect described differently.",
        )
        self.assertEqual(self.ledger.observe(repeated), handle)
        search = SearchStandard(
            op="search_standard",
            concept="blood pressure",
            collected_property="measurement unit",
            source=("unit", "bp"),
            question=first.question,
            impact="Try standard evidence for the same source question.",
        )
        self.assertEqual(self.ledger.observe(search), handle)
        self.assertEqual(len(self.ledger.records), 1)
        self.assertEqual(self.ledger.observe(read_action(source=["date"])), "Q2")
        self.assertEqual(
            self.ledger.observe(read_action(question="Is the unit explicitly stated?")),
            "Q3",
        )

    def test_legacy_unknown_can_share_the_same_backend_question(self):
        request = read_action()
        handle = self.ledger.observe(request)
        legacy = Unknown(
            missing=request.question,
            affected_refs=request.source,
            impact="Changed wording, same question.",
            source_kind="study",
            material_ref="study:other-policy",
        )
        self.assertEqual(self.ledger.observe(legacy), handle)
        self.assertEqual(len(self.ledger.records), 1)

    def test_invalid_observation_scope_or_duplicates_do_not_allocate_handles(self):
        for request, code in (
            (read_action(source=["bp", "bp"]), "evidence_source_duplicate"),
            (read_action(source=["foreign"]), "evidence_source_scope_invalid"),
        ):
            with self.subTest(code=code), self.assertRaises(WorkflowError) as error:
                self.ledger.observe(request, source_scope=("bp",))
            self.assertEqual(error.exception.code, code)
            self.assertEqual(self.ledger.records, ())
        self.assertEqual(self.ledger.observe(read_action()), "Q1")

    def test_unknown_handle_is_rejected_with_field_feedback(self):
        self.ledger.observe(read_action())
        before = self.ledger.records
        with self.assertRaises(WorkflowError) as error:
            self.ledger.apply_updates(
                (QuestionUpdate(question="Q99", status="resolved", reason="Done."),),
                visible_evidence=(),
                source_scope=("bp",),
            )
        self.assertEqual(error.exception.code, "question_handle_unknown")
        self.assertEqual(
            error.exception.feedback[0]["path"], "/action/question_updates/0/question"
        )
        self.assertEqual(error.exception.feedback[0]["allowed"], ["Q1"])
        self.assertEqual(self.ledger.records, before)

    def test_all_question_sources_must_be_within_revision_scope(self):
        handle = self.ledger.observe(read_action(source=["bp", "unit"]))
        with self.assertRaises(WorkflowError) as error:
            self.ledger.apply_updates(
                (
                    QuestionUpdate(
                        question=handle,
                        status="deferred",
                        reason="No association evidence is available.",
                    ),
                ),
                visible_evidence=("bp", "unit"),
                source_scope=("bp",),
            )
        self.assertEqual(error.exception.code, "question_source_scope_invalid")
        self.assertEqual(self.ledger.get(handle).status, "open")

    def test_known_but_undelivered_support_is_rejected_then_visible_support_passes(
        self,
    ):
        handle = self.ledger.observe(read_action())
        update = QuestionUpdate(
            question=handle,
            status="resolved",
            reason="The source page explicitly associates the unit with the result.",
            support=("source:page:1",),
        )
        with self.assertRaises(WorkflowError) as error:
            self.ledger.apply_updates(
                (update,), visible_evidence=("bp",), source_scope=("bp",)
            )
        self.assertEqual(error.exception.code, "question_support_unknown")
        self.assertEqual(self.ledger.get(handle).status, "open")
        accepted = self.ledger.apply_updates(
            (update,),
            visible_evidence=("bp", "source:page:1"),
            source_scope=("bp",),
        )
        self.assertEqual(accepted[0].support, ("source:page:1",))
        self.assertEqual(self.ledger.pending, ())

    def test_deferred_remains_pending_and_repeated_observation_does_not_erase_it(self):
        request = read_action()
        handle = self.ledger.observe(request)
        self.ledger.apply_updates(
            (
                QuestionUpdate(
                    question=handle,
                    status="deferred",
                    reason="No admitted study policy answers this question.",
                ),
            ),
            visible_evidence=(),
            source_scope=("bp",),
        )
        self.assertEqual(self.ledger.pending[0].status, "deferred")
        before = self.ledger.records
        self.assertEqual(self.ledger.observe(request), handle)
        self.assertEqual(self.ledger.records, before)

    def test_invalid_later_update_is_atomic_and_validation_is_read_only(self):
        first = self.ledger.observe(read_action())
        self.ledger.observe(read_action(source=["date"]))
        valid = QuestionUpdate(question=first, status="deferred", reason="Unavailable.")
        invalid = QuestionUpdate(question="Q99", status="deferred", reason="Missing.")
        before = self.ledger.records
        self.ledger.validate_updates(
            (valid,), visible_evidence=(), source_scope=("bp", "date")
        )
        self.assertEqual(self.ledger.records, before)
        with self.assertRaises(WorkflowError):
            self.ledger.apply_updates(
                (valid, invalid),
                visible_evidence=(),
                source_scope=("bp", "date"),
            )
        self.assertEqual(self.ledger.records, before)

    def test_conflicting_updates_and_duplicate_support_are_rejected(self):
        handle = self.ledger.observe(read_action())
        update = QuestionUpdate(
            question=handle, status="deferred", reason="Unavailable."
        )
        for updates, code in (
            ((update, update), "question_update_duplicate"),
            (
                (update.model_copy(update={"support": ("bp", "bp")}),),
                "question_support_duplicate",
            ),
        ):
            with self.subTest(code=code), self.assertRaises(WorkflowError) as error:
                self.ledger.apply_updates(
                    updates, visible_evidence=("bp",), source_scope=("bp",)
                )
            self.assertEqual(error.exception.code, code)
            self.assertEqual(self.ledger.get(handle).status, "open")

    def test_resolved_records_an_explicit_claim_without_inventing_support(self):
        handle = self.ledger.observe(read_action())
        self.ledger.apply_updates(
            (
                QuestionUpdate(
                    question=handle,
                    status="resolved",
                    reason="The model explicitly judges the existing source sufficient.",
                ),
            ),
            visible_evidence=("bp",),
            source_scope=("bp",),
        )
        self.assertEqual(self.ledger.get(handle).support, ())
        self.assertNotIn("semantic_review", self.ledger.get(handle).model_dump())
        for changes in ({"reason": " "}, {"status": "available"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                QuestionUpdate.model_validate(
                    {
                        "question": handle,
                        "status": "resolved",
                        "reason": "An explicit judgment.",
                        **changes,
                    }
                )


if __name__ == "__main__":
    unittest.main()
