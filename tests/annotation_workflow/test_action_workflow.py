"""Evidence actions exercise the real Session, SQLite knowledge and artifact store."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import RULES, SNAPSHOT, ScriptedModel, packet, understanding_json

from pipeline.stage_b_mapper_module.annotation_workflow.common import canonical
from pipeline.stage_b_mapper_module.annotation_workflow.evidence_actions import (
    ReadMaterial,
)
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.session import (
    Session,
    _revision_changes,
)
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    Material,
    MaterialStore,
)


def submit(status: str = "deferred", support: list[str] | None = None) -> str:
    return canonical(
        {
            "action": {
                "op": "submit",
                "records": [],
                "other_outcomes": [
                    {
                        "source": ["bp", "unit", "date"],
                        "outcome": "unresolved",
                        "reason": "The study-specific mapping remains unconfirmed.",
                    }
                ],
                "question_updates": [
                    {
                        "question": "Q1",
                        "status": status,
                        "reason": "The material was checked; this annotation remains pending study review.",
                        "support": support or [],
                    }
                ],
            }
        }
    )


def read_action(handle: str = "source:guide") -> str:
    return canonical(
        {
            "action": {
                "op": "read_material",
                "handle": handle,
                "source": ["bp"],
                "question": "Which study mapping applies to this measurement?",
                "impact": "Determines whether the candidate is applicable.",
            }
        }
    )


class ActionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)
        self.material = Material(
            ref="source:guide",
            kind="source",
            study_id=packet().study_id,
            text="The page contains a systolic result, mmHg and measurement date. No study mapping rule is stated.",
            source_ref=packet().objects[0].source_ref,
            origin="blank_crf",
        )

    def run_session(self, outputs: list[str]) -> tuple[object, ScriptedModel]:
        model = ScriptedModel(outputs)
        result = Session(
            packet(),
            self.knowledge,
            model,
            self.output,
            materials=MaterialStore((self.material,)),
        ).run()
        return result, model

    def test_source_read_precedes_disposition_and_can_be_deferred(self) -> None:
        result, model = self.run_session(
            [understanding_json(), read_action(), understanding_json(), submit()]
        )
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 4)
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertTrue((self.output / "02.action.json").exists())
        material = json.loads((self.output / "material-001.json").read_text())
        self.assertEqual(material["document"]["kind"], "source")
        self.assertEqual(material["status"], "available")
        payload = json.loads(model.requests[-1][1]["content"])
        self.assertEqual(payload["questions"][0]["handle"], "Q1")
        self.assertEqual(payload["questions"][0]["status"], "open")
        ledger = json.loads((self.output / "question-ledger.json").read_text())
        self.assertEqual(ledger[0]["status"], "deferred")
        self.assertIn("evidence_question_deferred", [i.code for i in result.issues])
        self.assertTrue(
            all(a.disposition == "unresolved" for a in result.decision.annotations)
        )

    def test_search_action_executes_query_instead_of_resolving_a_fake_ref(self) -> None:
        search = canonical(
            {
                "action": {
                    "op": "search_standard",
                    "concept": "Diastolic blood pressure",
                    "collected_property": "measurement result",
                    "source": ["bp"],
                    "question": "Does the alternative pressure concept have a standard candidate?",
                    "impact": "Check whether a candidate represents a different measurement.",
                }
            }
        )
        result, _ = self.run_session([understanding_json(), search, submit()])
        self.assertEqual(result.stop_reason, "completed_for_review")
        material = json.loads((self.output / "material-001.json").read_text())
        self.assertIsNone(material["request"]["material_ref"])
        self.assertEqual(material["request"]["source_kind"], "standard")
        self.assertTrue(material["queries"])
        self.assertIn(
            "cosmos:sdtm:2026-05-26:DIABP", [c["ref"] for c in material["candidates"]]
        )
        self.assertFalse((self.output / "02.accepted-decision.json").exists())

    def test_unknown_handle_is_correctable_without_reading_or_accepting_it(
        self,
    ) -> None:
        result, _ = self.run_session(
            [
                understanding_json(),
                read_action("invented-policy"),
                read_action(),
                understanding_json(),
                submit(),
            ]
        )
        self.assertEqual(result.stop_reason, "completed_for_review")
        rejection = json.loads((self.output / "02.validation.json").read_text())
        self.assertEqual(rejection["code"], "action_material_unknown")
        self.assertFalse((self.output / "02.action.json").exists())
        self.assertEqual(len(list(self.output.glob("material-*.json"))), 1)
        ledger = json.loads((self.output / "question-ledger.json").read_text())
        self.assertEqual(len(ledger), 1)

    def test_unread_support_cannot_close_question_or_mutate_accepted_state(
        self,
    ) -> None:
        result, model = self.run_session(
            [
                understanding_json(),
                read_action(),
                understanding_json(),
                submit("resolved", ["unread-policy"]),
                submit(),
            ]
        )
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertFalse((self.output / "04.accepted-decision.json").exists())
        payload = json.loads(model.requests[-1][1]["content"])
        self.assertEqual(payload["questions"][0]["status"], "open")
        self.assertEqual(
            json.loads((self.output / "question-ledger.json").read_text())[0]["status"],
            "deferred",
        )

    def test_rewording_impact_does_not_close_an_existing_question(self) -> None:
        question = {
            "source_kind": "standard",
            "material_ref": "templates",
            "affected_refs": ["bp"],
            "missing": "Which mapping applies?",
            "impact": "Select a target",
        }
        old = {"records": [], "other_outcomes": [], "open_questions": [question]}
        new = {
            **old,
            "open_questions": [{**question, "impact": "Confirm the selected target"}],
        }
        self.assertEqual(_revision_changes(old, new), [])

    def test_question_outside_revision_scope_stays_open(self) -> None:
        bad = json.loads(submit("resolved", ["unit"]))
        corrected = json.loads(submit())
        corrected["action"]["question_updates"] = []
        model = ScriptedModel(
            [understanding_json(), canonical(bad), canonical(corrected)]
        )
        session = Session(packet(), self.knowledge, model, self.output)
        session._understand()
        session._retrieve()
        session.questions.observe(
            ReadMaterial(
                op="read_material",
                handle="unit-guide",
                source=("unit",),
                question="Which unit convention applies?",
                impact="Select the unit.",
            ),
            source_scope=packet().output_scope,
        )
        self.assertTrue(session._decide(affected={"bp"}))
        rejection = json.loads((self.output / "02.validation.json").read_text())
        self.assertEqual(rejection["code"], "question_source_scope_invalid")
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertEqual(session.questions.get("Q1").status, "open")
        self.assertEqual(
            json.loads(model.requests[-1][1]["content"])["revision_scope"], ["bp"]
        )

    def test_bad_read_does_not_replace_annotation_repair_draft(self) -> None:
        invalid_read = json.loads(read_action("invented-guide"))
        invalid_read["action"]["source"] = ["date"]
        valid_read = json.loads(read_action())
        valid_read["action"]["source"] = ["date"]
        bad_submit = json.loads(submit())
        bad_submit["action"]["question_updates"] = []
        bad_submit["action"]["other_outcomes"] = [
            {
                "source": [ref],
                "outcome": "unresolved" if ref != "date" else "invalid",
                "reason": "The mapping remains unconfirmed.",
            }
            for ref in ("bp", "unit", "date")
        ]
        corrected = json.loads(canonical(bad_submit))
        corrected["action"]["other_outcomes"][2]["outcome"] = "unresolved"
        corrected["action"]["question_updates"] = json.loads(submit())["action"][
            "question_updates"
        ]
        changed = json.loads(canonical(corrected))
        changed["action"]["other_outcomes"][0]["outcome"] = "not_submitted"
        result, _ = self.run_session(
            [
                understanding_json(),
                canonical(invalid_read),
                canonical(bad_submit),
                canonical(valid_read),
                understanding_json(),
                canonical(changed),
                canonical(corrected),
            ]
        )
        self.assertEqual(result.stop_reason, "completed_for_review")
        rejection = json.loads((self.output / "06.validation.json").read_text())
        self.assertEqual(rejection["code"], "unaffected_judgment_changed")
        self.assertFalse((self.output / "06.accepted-decision.json").exists())
        self.assertEqual(
            next(
                a.disposition
                for a in result.decision.annotations
                if "bp" in a.source_scope
            ),
            "unresolved",
        )


if __name__ == "__main__":
    unittest.main()
