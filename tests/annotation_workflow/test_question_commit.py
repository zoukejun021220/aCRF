"""A decision and its question dispositions have one authoritative publication."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import (
    RULES,
    SNAPSHOT,
    ScriptedModel,
    packet,
    understanding,
    understanding_json,
)

from pipeline.stage_b_mapper_module.annotation_workflow.common import (
    WorkflowError,
    canonical,
)
from pipeline.stage_b_mapper_module.annotation_workflow.evidence_actions import (
    ReadMaterial,
)
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.session import Session


def submit(status: str) -> str:
    return canonical(
        {
            "action": {
                "op": "submit",
                "records": [],
                "other_outcomes": [
                    {
                        "source": ["bp", "unit", "date"],
                        "outcome": "unresolved",
                        "reason": "The annotation remains pending study review.",
                    }
                ],
                "question_updates": [
                    {
                        "question": "Q1",
                        "status": status,
                        "reason": "The visible source was checked for this disposition.",
                        "support": ["bp"],
                    }
                ],
            }
        }
    )


class QuestionCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "run"
        knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(knowledge.close)
        self.session = Session(
            packet(),
            knowledge,
            ScriptedModel([submit("deferred"), submit("resolved")]),
            self.output,
        )
        self.session.understanding = understanding()
        self.session.questions.observe(
            ReadMaterial(
                op="read_material",
                handle="source:page",
                source=("bp",),
                question="Does the visible source identify the measurement?",
                impact="Determines which measurement is interpreted.",
            ),
            source_scope=self.session.packet.output_scope,
        )
        self.assertTrue(self.session._decide())

    def snapshot(self) -> dict[str, object]:
        return json.loads((self.output / "snapshot.json").read_text())

    def fail_publication(self, name: str):
        replace = os.replace

        def checked_replace(source, target):
            if Path(target).name == name:
                raise OSError("Injected publication failure")
            return replace(source, target)

        return patch(
            "pipeline.stage_b_mapper_module.annotation_workflow.store.os.replace",
            side_effect=checked_replace,
        )

    def test_snapshot_contains_the_decision_and_complete_question_state(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(snapshot["phase"], "decision")
        self.assertEqual(
            snapshot["value"], self.session.decision.model_dump(mode="json")
        )
        self.assertEqual(
            snapshot["questions"],
            [
                record.model_dump(mode="json")
                for record in self.session.questions.records
            ],
        )
        self.assertEqual(snapshot["questions"][0]["status"], "deferred")
        self.assertEqual(snapshot["question_updates"][0]["status"], "deferred")
        self.assertEqual(
            snapshot["questions"],
            json.loads((self.output / "question-ledger.json").read_text()),
        )

    def test_snapshot_failure_keeps_previous_authority_and_memory(self) -> None:
        session = self.session
        before = {
            name: (self.output / name).read_bytes()
            for name in (
                "snapshot.json",
                "question-ledger.json",
                "accepted-proposal.json",
            )
        }
        old_decision, old_proposal, old_questions = (
            session.decision,
            session.proposal,
            session.questions,
        )
        revision = session.store.revision
        with self.fail_publication("snapshot.json"):
            with self.assertRaises(WorkflowError) as error:
                session._decide()
        self.assertEqual(error.exception.code, "artifact_write_failed")
        self.assertIs(session.decision, old_decision)
        self.assertIs(session.proposal, old_proposal)
        self.assertIs(session.questions, old_questions)
        self.assertEqual(session.questions.get("Q1").status, "deferred")
        self.assertEqual(session.store.revision, revision)
        for name, content in before.items():
            self.assertEqual((self.output / name).read_bytes(), content)
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertFalse((self.output / "02.question-updates.json").exists())

    def test_derived_ledger_failure_keeps_new_authority_and_memory_coherent(
        self,
    ) -> None:
        old_questions = self.session.questions
        revision = self.session.store.revision
        with self.fail_publication("question-ledger.json"):
            with self.assertRaises(WorkflowError) as error:
                self.session._decide()
        self.assertEqual(error.exception.code, "artifact_write_failed")
        snapshot = self.snapshot()
        self.assertEqual(snapshot["revision"], revision + 1)
        self.assertEqual(snapshot["questions"][0]["status"], "resolved")
        self.assertEqual(snapshot["question_updates"][0]["status"], "resolved")
        self.assertEqual(
            snapshot["value"], self.session.decision.model_dump(mode="json")
        )
        self.assertEqual(self.session.questions.get("Q1").status, "resolved")
        self.assertEqual(old_questions.get("Q1").status, "deferred")
        self.assertEqual(
            json.loads((self.output / "question-ledger.json").read_text())[0]["status"],
            "deferred",
        )

    def test_first_derived_artifact_failure_does_not_leave_memory_at_old_revision(
        self,
    ) -> None:
        with self.fail_publication("02.accepted-decision.json"):
            with self.assertRaises(WorkflowError) as error:
                self.session._decide()
        self.assertEqual(error.exception.code, "artifact_write_failed")
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertEqual(self.snapshot()["questions"][0]["status"], "resolved")
        self.assertEqual(self.session.questions.get("Q1").status, "resolved")
        self.assertEqual(
            self.snapshot()["value"], self.session.decision.model_dump(mode="json")
        )

    def test_public_run_reports_derived_failure_with_committed_decision(self) -> None:
        session = Session(
            packet(),
            self.session.knowledge,
            ScriptedModel([understanding_json(), submit("resolved")]),
            self.output / "public-run",
        )
        session.questions = self.session.questions
        with self.fail_publication("question-ledger.json"):
            result = session.run()
        self.assertEqual(result.stop_reason, "artifact_write_failed")
        self.assertEqual(result.status, "partial")
        snapshot = json.loads((session.store.path / "snapshot.json").read_text())
        self.assertEqual(snapshot["phase"], "decision")
        self.assertEqual(snapshot["questions"][0]["status"], "resolved")
        self.assertEqual(snapshot["value"], result.decision.model_dump(mode="json"))
        self.assertEqual(session.questions.get("Q1").status, "resolved")
        self.assertEqual(self.session.questions.get("Q1").status, "deferred")


if __name__ == "__main__":
    unittest.main()
