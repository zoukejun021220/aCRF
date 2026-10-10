"""Decision explanations must be required, grounded and visible to the reviewer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from support import (
    RULES,
    SNAPSHOT,
    SYSBP,
    ScriptedModel,
    decision,
    packet,
    proposal,
    understanding,
    understanding_json,
    understanding_proposal,
)

from pipeline.stage_b_mapper_module.annotation_workflow.checks import check_decision
from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.decision import MappingDecision
from pipeline.stage_b_mapper_module.annotation_workflow.evaluation import compare
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.review import render
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import Reply, schema_for
from pipeline.stage_b_mapper_module.annotation_workflow.session import Session
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
    queries,
)
from pipeline.stage_b_mapper_module.annotation_workflow.understanding_proposal import (
    UnderstandingProposal,
)


class ExplanationTests(unittest.TestCase):
    """Missing reasons cannot be hidden by valid mapping structure."""

    def setUp(self) -> None:
        """Bind real source evidence and private output storage for each test."""
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"

    def test_mapped_annotation_requires_nonblank_reason(self) -> None:
        """A plausible target without an explanation must fail local validation."""
        value = decision().model_dump(mode="json")
        value["annotations"][0]["reason"] = "   "
        with self.assertRaises(ValidationError):
            MappingDecision.model_validate(value)

    def test_candidate_selection_reason_is_visible_in_review(self) -> None:
        """A correct selection reason in JSON must not disappear from human review."""
        knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(knowledge.close)
        result = render(decision(), (knowledge.resolve(SYSBP),))
        self.assertIn(decision().decisions[0].reason, result)

    def test_each_stage_requires_a_bounded_explanation(self) -> None:
        """Omission, whitespace and oversized prose cannot pass either stage schema."""
        for phase, value in [
            (
                "understanding",
                UnderstandingProposal.model_validate(understanding_proposal()),
            ),
            ("decision", decision()),
        ]:
            if phase == "understanding":
                self.assertIn("explanation", schema_for(phase)["required"])
            for mutation in ("missing", "blank", "long"):
                raw = value.model_dump(mode="json")
                if mutation == "missing":
                    raw.pop("explanation")
                else:
                    raw["explanation"]["summary"] = (
                        "  " if mutation == "blank" else "x" * 601
                    )
                with self.subTest(phase=phase, mutation=mutation), self.assertRaises(
                    ValidationError
                ):
                    type(value).model_validate(raw)

    def test_undeclared_explanation_evidence_cannot_be_published(self) -> None:
        """Plausible prose cannot cite a source or standard that was never delivered."""
        raw = decision().model_dump(mode="json")
        raw["explanation"]["evidence_refs"] = ["invented-rule"]
        with self.assertRaises(WorkflowError):
            check_decision(
                MappingDecision.model_validate(raw),
                packet(),
                understanding(),
                (self.knowledge.resolve(SYSBP),),
            )
        source = understanding_proposal()
        source["explanation"]["evidence_refs"] = ["other-study"]
        result = Session(
            packet(),
            self.knowledge,
            ScriptedModel([json.dumps(source)] * 3),
            self.output,
        ).run()
        self.assertEqual(result.stop_reason, "understanding_correction_no_progress")
        self.assertIsNone(result.understanding)
        self.assertFalse((self.output / "snapshot.json").exists())

        for collection in ("items", "bindings"):
            value = understanding().model_dump(mode="json")
            value[collection][0]["basis"]["reason"] = " "
            with self.subTest(collection=collection), self.assertRaises(
                ValidationError
            ):
                CollectionUnderstanding.model_validate(value)

    def test_understanding_decision_and_consistency_reasons_survive_in_order(
        self,
    ) -> None:
        """Every saved model turn is visible separately, with backend retrieval labelled."""
        initial = proposal()
        reviewed = json.loads(json.dumps(initial))
        reviewed["records"][0][
            "reason"
        ] = "Retained the grouping: the printed unit and date refer to the same measurement."
        with patch(
            "pipeline.stage_b_mapper_module.annotation_workflow.session.needs_consistency",
            return_value=True,
        ):
            result = Session(
                packet(),
                self.knowledge,
                ScriptedModel(
                    [
                        understanding_json(),
                        json.dumps(initial),
                        json.dumps(reviewed),
                    ]
                ),
                self.output,
            ).run()
        self.assertTrue(result.consistency_reviewed)
        notes = json.loads((self.output / "decision-notes.json").read_text())
        self.assertEqual(
            [(note["stage"], note["author"]) for note in notes],
            [
                ("understanding", "model"),
                ("retrieval", "backend"),
                ("decision", "model"),
                ("consistency", "model"),
            ],
        )
        self.assertEqual(
            [note["revision"] for note in notes if note["author"] == "model"], [1, 2, 3]
        )
        rendered = (self.output / "decision-notes.md").read_text()
        for explanation in (
            understanding().explanation.summary,
            initial["records"][0]["reason"],
            reviewed["records"][0]["reason"],
            understanding().bindings[0].basis.reason,
        ):
            self.assertIn(explanation, rendered)

    def test_failed_turn_preserves_prior_reason_without_inventing_new_model_reason(
        self,
    ) -> None:
        """An empty formal response yields only a backend stop entry, not fabricated prose."""
        failed = Reply(
            content="",
            error="response_content_empty",
            status=200,
            finish_reason="stop",
            prompt_tokens=62,
            completion_tokens=6,
            reasoning_present=True,
            response_sha256=None,
            seconds=0,
        )
        result = Session(
            packet(),
            self.knowledge,
            ScriptedModel([understanding_json(), failed]),
            self.output,
        ).run()
        self.assertEqual(result.stop_reason, "response_content_empty")
        notes = json.loads((self.output / "decision-notes.json").read_text())
        self.assertEqual(
            [note["stage"] for note in notes if note["author"] == "model"],
            ["understanding"],
        )
        self.assertEqual(notes[-1]["stage"], "stopped")
        self.assertIn(
            understanding().explanation.summary,
            (self.output / "decision-notes.md").read_text(),
        )

    def test_prose_changes_do_not_change_queries_or_semantic_scores(self) -> None:
        """Explanations are audit data and cannot create a score improvement by wording."""
        raw = understanding().model_dump(mode="json")
        raw["explanation"]["summary"] = "Same interpretation in different words."
        raw["items"][0]["basis"][
            "reason"
        ] = "The source explicitly labels systolic pressure."
        self.assertEqual(
            queries(CollectionUnderstanding.model_validate(raw), packet()),
            queries(understanding(), packet()),
        )
        mapped = decision().model_dump(mode="json")
        mapped["explanation"]["summary"] = "Same supported targets in different words."
        mapped["annotations"][0][
            "reason"
        ] = "Collected systolic result, with SYSBP as its topic."
        self.assertTrue(
            compare(MappingDecision.model_validate(mapped), decision()).exact
        )
