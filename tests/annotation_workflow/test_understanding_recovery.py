"""Source interpretation corrections preserve evidence and respect shared limits."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from support import RULES, SNAPSHOT, ScriptedModel, packet

from pipeline.stage_b_mapper_module.annotation_workflow.common import (
    WorkflowError,
    canonical,
)
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import Reply
from pipeline.stage_b_mapper_module.annotation_workflow.session import Budget, Session
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    Material,
    MaterialStore,
)


def draft() -> dict:
    """Write the provider contract independently of the backend's compiled object."""
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
                    "reason": "The label and shared measurement context support this source interpretation.",
                },
            }
            for ref, concept, prop, role in [
                (
                    "bp",
                    "Systolic blood pressure",
                    "measurement result",
                    "measurement_result",
                ),
                ("unit", "Measurement unit", "unit", "unit"),
                ("date", "Measurement date", "date", "time"),
            ]
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
                    "reason": "The shared context links the qualifier to this result.",
                },
            }
            for ref, relation in [("unit", "unit_for"), ("date", "time_of")]
        ],
        "conditions": {"status": "none_stated_in_supplied_source", "entries": []},
        "material_unknowns": [],
        "explanation": {
            "summary": "The source collects a systolic result with a unit and measurement date.",
            "evidence_refs": ["bp", "unit", "date", "measurement"],
            "uncertainties": [],
        },
    }


class UnderstandingRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)

    def session(
        self, outputs: list[str | Reply], budget: Budget | None = None
    ) -> Session:
        self.model = ScriptedModel(outputs)
        return Session(packet(), self.knowledge, self.model, self.output, budget=budget)

    def feedback(self, request_index: int = 1) -> list[dict]:
        return json.loads(self.model.requests[request_index][1]["content"])[
            "correction"
        ]["errors"]

    def test_backend_owned_modes_and_separate_evidence_are_accepted_once(self) -> None:
        value = draft()
        session = self.session([canonical(value)])
        session._understand()
        self.assertEqual(session.calls, 1)
        self.assertEqual(
            [item.value_mode for item in session.understanding.items],
            ["entered", "printed_context", "entered"],
        )
        self.assertEqual(session.understanding.items[0].source_refs, ("bp",))
        self.assertEqual(
            session.understanding.items[0].basis.source_refs, ("bp", "measurement")
        )
        payload = json.loads(self.model.requests[0][1]["content"])
        self.assertIn("subject_choices", payload["understanding_context"])
        self.assertEqual(
            json.loads(
                (self.output / "accepted-understanding-proposal.json").read_text()
            ),
            value,
        )
        self.assertTrue((self.output / "01.accepted-understanding.json").exists())
        self.assertTrue(
            json.loads((self.output / "01.understanding-validation.json").read_text())[
                "accepted"
            ]
        )

    def test_unknown_subject_and_missing_coverage_receive_semantic_feedback(
        self,
    ) -> None:
        invalid = draft()
        invalid["items"][0]["source_ref"] = "invented-subject"
        session = self.session([canonical(invalid), canonical(draft())])
        session._understand()
        errors = self.feedback()
        self.assertEqual(session.calls, 2)
        self.assertTrue(any("bp" in item.get("affected_refs", []) for item in errors))
        self.assertTrue(any("allowed_refs" in item for item in errors))
        self.assertTrue((self.output / "01.understanding-proposal.json").exists())
        notes = json.loads((self.output / "decision-notes.json").read_text())
        self.assertEqual(
            [(item["stage"], item["author"]) for item in notes],
            [("understanding_correction", "backend"), ("understanding", "model")],
        )

    def test_schema_repair_cannot_erase_unaffected_items_bindings_conditions_or_unknowns(
        self,
    ) -> None:
        valid = draft()
        valid["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": ["bp"],
                    "kind": "collection",
                    "predicate": {"op": "eq", "source_id": "bp", "value": "collected"},
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["bp"],
                        "reason": "The provided source supplies this condition.",
                    },
                }
            ],
        }
        valid["material_unknowns"] = [
            {
                "missing": "The study-specific result convention",
                "affected_refs": ["bp"],
                "impact": "Determine the source convention before finalizing the result interpretation",
                "source_kind": "study",
                "material_ref": None,
            }
        ]
        invalid = copy.deepcopy(valid)
        invalid["items"][2]["source_role"] = "NOT_A_ROLE"
        damaged = copy.deepcopy(valid)
        damaged["items"][0]["concept"] = "Unrelated new concept"
        damaged["bindings"] = [damaged["bindings"][1]]
        damaged["conditions"] = {
            "status": "none_stated_in_supplied_source",
            "entries": [],
        }
        damaged["material_unknowns"] = []
        session = self.session(
            [canonical(invalid), canonical(damaged), canonical(valid)]
        )
        session._understand()
        initial_feedback = self.feedback()
        self.assertTrue(
            all(item["affected_refs"] == ["date"] for item in initial_feedback)
        )
        rejected = json.loads(
            (self.output / "02.understanding-validation.json").read_text()
        )
        self.assertEqual(rejected["code"], "unaffected_understanding_changed")
        paths = {item["path"] for item in rejected["feedback"]}
        self.assertTrue(
            {"/items/0", "/bindings/0", "/conditions/entries/0", "/material_unknowns/0"}
            <= paths
        )
        self.assertEqual(session.calls, 3)
        self.assertEqual(len(session.understanding.conditions.entries), 1)
        self.assertEqual(len(session.understanding.material_unknowns), 1)

    def test_explanation_correction_cannot_change_unknown_condition_status(
        self,
    ) -> None:
        valid = draft()
        valid["conditions"]["status"] = "unknown"
        invalid = copy.deepcopy(valid)
        invalid["explanation"]["evidence_refs"] = ["not-delivered"]
        damaged = copy.deepcopy(valid)
        damaged["conditions"]["status"] = "none_stated_in_supplied_source"
        session = self.session(
            [canonical(invalid), canonical(damaged), canonical(valid)]
        )
        session._understand()
        self.assertEqual(session.understanding.conditions.status, "unknown")
        self.assertEqual(session.calls, 3)
        self.assertEqual(
            json.loads((self.output / "02.understanding-validation.json").read_text())[
                "code"
            ],
            "unaffected_understanding_changed",
        )

    def test_condition_status_itself_can_be_corrected(self) -> None:
        invalid = draft()
        invalid["conditions"]["status"] = "specified"
        session = self.session([canonical(invalid), canonical(draft())])
        session._understand()
        self.assertEqual(session.calls, 2)
        self.assertEqual(
            session.understanding.conditions.status, "none_stated_in_supplied_source"
        )

    def test_failed_revision_keeps_last_accepted_understanding_and_raw_draft(
        self,
    ) -> None:
        original = canonical(draft())
        failures = []
        for index in range(3):
            invalid = draft()
            invalid["items"][0]["source_role"] = f"invalid-{index}"
            failures.append(canonical(invalid))
        session = self.session([original, *failures])
        session._understand()
        accepted = session.understanding
        snapshot = (self.output / "snapshot.json").read_bytes()
        with self.assertRaisesRegex(
            WorkflowError, "understanding_correction_exhausted"
        ):
            session._understand()
        self.assertIs(session.understanding, accepted)
        self.assertEqual((self.output / "snapshot.json").read_bytes(), snapshot)
        self.assertEqual(
            json.loads(
                (self.output / "accepted-understanding-proposal.json").read_text()
            ),
            draft(),
        )
        self.assertEqual(session.calls, 4)

    def test_material_explanation_evidence_requires_actual_delivery(self) -> None:
        guide = Material(
            ref="study-guide",
            kind="study",
            study_id=packet().study_id,
            text="The measurement date accompanies the systolic result.",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        with_guide = draft()
        with_guide["explanation"]["evidence_refs"].append(guide.ref)
        session = self.session(
            [canonical(with_guide), canonical(draft()), canonical(with_guide)]
        )
        session.materials = MaterialStore((guide,))
        session._understand()
        evidence_errors = [
            item for item in self.feedback() if "explanation" in item["path"]
        ]
        self.assertTrue(evidence_errors)
        self.assertTrue(
            all(
                guide.ref not in item.get("allowed_refs", [])
                for item in evidence_errors
            )
        )
        self.assertEqual(session.calls, 2)
        session.documents = (guide,)
        session._understand()
        self.assertEqual(session.calls, 3)
        self.assertIn(guide.ref, session.understanding.explanation.evidence_refs)

    def test_missing_root_section_cannot_authorize_unrelated_item_changes(self) -> None:
        invalid = draft()
        del invalid["bindings"]
        damaged = draft()
        damaged["items"][0]["source_role"] = "administrative"
        session = self.session(
            [canonical(invalid), canonical(damaged), canonical(draft())]
        )
        session._understand()
        self.assertTrue(self.feedback()[0]["scope_known"])
        self.assertEqual(self.feedback()[0]["path"], "/bindings")
        self.assertEqual(session.calls, 3)
        self.assertEqual(
            session.understanding.items[0].source_role, "measurement_result"
        )
        self.assertEqual(
            json.loads((self.output / "02.understanding-validation.json").read_text())[
                "code"
            ],
            "unaffected_understanding_changed",
        )

    def test_equivalent_evidence_order_is_not_a_semantic_change(self) -> None:
        invalid = draft()
        invalid["items"][0]["source_role"] = "BAD_ROLE"
        corrected = draft()
        corrected["items"][2]["basis"]["source_refs"].reverse()
        session = self.session([canonical(invalid), canonical(corrected)])
        session._understand()
        self.assertEqual(session.calls, 2)
        self.assertEqual(
            session.understanding.items[2].basis.source_refs, ("measurement", "date")
        )

    def pending_unit_question(self) -> dict:
        value = draft()
        value["material_unknowns"] = [
            {
                "missing": "The study unit convention",
                "affected_refs": ["unit"],
                "impact": "Clarify which printed unit convention applies",
                "source_kind": "source",
                "material_ref": "source-guide",
            }
        ]
        return value

    def test_evidence_followup_preserves_unrelated_accepted_interpretations(
        self,
    ) -> None:
        original = self.pending_unit_question()
        damaged = copy.deepcopy(original)
        damaged["items"][0]["source_role"] = "administrative"
        session = self.session(
            [canonical(original), canonical(damaged), canonical(original)]
        )
        session._understand()
        self.assertTrue(session._read(session.understanding.material_unknowns))
        self.assertEqual(session.lookup_results[-1]["status"], "unavailable")
        session._understand()
        self.assertEqual(session.calls, 3)
        self.assertEqual(
            session.understanding.items[0].source_role, "measurement_result"
        )
        self.assertEqual(
            self.feedback(2)[0]["code"], "unaffected_understanding_changed"
        )
        payload = json.loads(self.model.requests[2][1]["content"])
        self.assertEqual(payload["revision_scope"], ["unit"])
        self.assertEqual(
            payload["correction"]["preserved_draft"]["items"][0]["source_role"],
            "measurement_result",
        )

    def test_closing_accepted_question_requires_updated_model_explanation(self) -> None:
        original = self.pending_unit_question()
        unexplained = draft()
        explained = draft()
        explained["explanation"][
            "summary"
        ] = "No source guide was available. The printed unit is usable provisionally from its label, while study approval remains pending."
        session = self.session(
            [canonical(original), canonical(unexplained), canonical(explained)]
        )
        session._understand()
        self.assertTrue(session._read(session.understanding.material_unknowns))
        session._understand()
        self.assertEqual(session.calls, 3)
        self.assertEqual(
            self.feedback(2)[0]["code"],
            "understanding_evidence_resolution_reason_required",
        )
        self.assertEqual(session.understanding.material_unknowns, ())
        self.assertEqual(
            session.understanding.explanation.summary,
            explained["explanation"]["summary"],
        )

    def test_identical_invalid_drafts_stop_without_using_all_corrections(self) -> None:
        invalid = draft()
        invalid["items"].pop()
        session = self.session([canonical(invalid), canonical(invalid)])
        with self.assertRaisesRegex(
            WorkflowError, "understanding_correction_no_progress"
        ):
            session._understand()
        self.assertEqual(session.calls, 2)
        self.assertIsNone(session.understanding)

    def test_malformed_json_uses_the_same_bounded_correction_loop(self) -> None:
        session = self.session(["{", canonical(draft())])
        session._understand()
        self.assertEqual(session.calls, 2)
        self.assertEqual(self.feedback()[0]["code"], "invalid_json")
        self.assertFalse((self.output / "01.understanding-proposal.json").exists())
        self.assertTrue((self.output / "01.reply.json").exists())

    def test_transport_failure_is_not_reclassified_as_semantic_correction(self) -> None:
        reply = Reply(
            content=None,
            error="request_failed",
            status=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_present=False,
            response_sha256=None,
            seconds=0,
        )
        session = self.session([reply])
        outcome = session.run()
        self.assertEqual(outcome.stop_reason, "request_failed")
        self.assertEqual(outcome.calls, 1)
        self.assertFalse((self.output / "01.understanding-validation.json").exists())

    def test_shared_budget_and_operator_stop_prevent_another_correction(self) -> None:
        first = draft()
        first["items"][0]["source_role"] = "bad-one"
        second = copy.deepcopy(first)
        second["items"][0]["source_role"] = "bad-two"
        session = self.session(
            [canonical(first), canonical(second)], Budget(max_calls=2)
        )
        outcome = session.run()
        self.assertEqual(outcome.stop_reason, "budget_exhausted")
        self.assertEqual(outcome.calls, 2)
        stopped = Session(
            packet(), self.knowledge, ScriptedModel([]), self.output.parent / "stopped"
        )
        (stopped.store.path / "STOP").touch()
        outcome = stopped.run()
        self.assertEqual(outcome.stop_reason, "operator_stop")
        self.assertEqual(outcome.calls, 0)


if __name__ == "__main__":
    unittest.main()
