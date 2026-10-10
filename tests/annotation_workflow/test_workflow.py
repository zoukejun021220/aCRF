"""Real SQLite/source/decision/storage seams with the model replaced only at its edge."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.review import (
    needs_consistency,
    render,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import Reply
from pipeline.stage_b_mapper_module.annotation_workflow.session import Budget, Session
from pipeline.stage_b_mapper_module.annotation_workflow.store import RunStore
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
    MappingQuery,
    queries,
)


class RetrievalTests(unittest.TestCase):
    """Exercise real verified exports and SQLite ranking, including no-template fallback."""

    def setUp(self) -> None:
        """Open a fresh isolated index backed by pinned public source bytes."""
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)

    def test_full_bundle_and_version_filter(self) -> None:
        """Correct result-only guesses fail the full-template and exact-value assertions."""
        results = self.knowledge.search(queries(understanding(), packet())[0])
        self.assertEqual(results[0].ref, SYSBP)
        self.assertEqual(len(results[0].rows), 11)
        self.assertEqual(results[0].row("VSTESTCD").values()["assigned_value"], "SYSBP")
        self.assertEqual(results[0].row("VSSTRESN").values()["origin_type"], "Derived")
        self.assertTrue(
            results[0].source_url.endswith(
                "cdisc_sdtm_dataset_specializations_latest.csv"
            )
        )
        with self.assertRaises(WorkflowError):
            self.knowledge.resolve("cosmos:sdtm:2024-10-01:TPLAB")
        results[0].row("VSTESTCD").values()["assigned_value"] = "CHANGED"
        self.assertEqual(
            self.knowledge.resolve(SYSBP).row("VSTESTCD").values()["assigned_value"],
            "SYSBP",
        )

    def test_concept_change_changes_real_candidates(self) -> None:
        """Systolic and diastolic source interpretations select different actual bundles."""
        query = queries(understanding(), packet())[0]
        changed = MappingQuery.model_validate(
            {**query.model_dump(mode="json"), "concept": "Diastolic blood pressure"}
        )
        self.assertEqual(
            self.knowledge.search(changed)[0].title, "Diastolic Blood Pressure"
        )
        self.assertNotEqual(
            self.knowledge.search(query)[0].ref, self.knowledge.search(changed)[0].ref
        )

    def test_positive_rule_fallback_and_truly_missing_rule(self) -> None:
        """A no-template path must retrieve an actual rule, not always return a gap."""
        rule_only = Knowledge(
            SNAPSHOT, packet().standard, RULES, include_templates=False
        )
        self.addCleanup(rule_only.close)
        query = MappingQuery(
            item_ref="item:1",
            source_refs=("bp",),
            concept="concomitant medication indication",
            collected_property="reason for use",
            context=(),
            conditions=(),
        )
        result = rule_only.search(query)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "rule_evidence_bundle")
        self.assertEqual(result[0].row("CMINDC").variable, "CMINDC")
        self.assertIn("CMINDC", result[0].row("CMINDC").source_locator)
        missing = Knowledge(SNAPSHOT, packet().standard, include_templates=False)
        self.addCleanup(missing.close)
        self.assertEqual(missing.search(query), ())
        wrong_property = MappingQuery.model_validate(
            {
                **query.model_dump(mode="json"),
                "collected_property": "performance status",
            }
        )
        self.assertEqual(rule_only.search(wrong_property), ())

    def test_unknown_concept_and_injection_do_not_match_arbitrary_rows(self) -> None:
        """SQL-bound values cannot become SQL or an invented mapping answer."""
        query = MappingQuery(
            item_ref="item:1",
            source_refs=("bp",),
            concept="zznonexistent '); DROP TABLE candidates; --",
            collected_property="zzunknown",
            context=(),
            conditions=(),
        )
        self.assertEqual(self.knowledge.search(query), ())
        self.assertEqual(
            self.knowledge.search(queries(understanding(), packet())[0])[0].ref, SYSBP
        )

    def test_positive_rule_maps_real_target_without_any_template(self) -> None:
        """D11A must produce a supported binding, not merely return a search hit."""
        raw_source = packet().model_dump(mode="json")
        raw_source["objects"] = [
            {
                **raw_source["objects"][1],
                "text": "Reason medication was taken",
                "context_ref": None,
                "datatype": "text",
            }
        ]
        raw_source["output_scope"] = ["bp"]
        source = type(packet()).model_validate(raw_source)
        raw_understanding = understanding().model_dump(mode="json")
        raw_understanding["items"] = [
            {
                **raw_understanding["items"][0],
                "concept": "concomitant medication indication",
                "collected_property": "reason for use",
                "source_role": "event_or_intervention_description",
            }
        ]
        raw_understanding["bindings"] = []
        meaning = CollectionUnderstanding.model_validate(raw_understanding)
        knowledge = Knowledge(SNAPSHOT, source.standard, RULES, include_templates=False)
        self.addCleanup(knowledge.close)
        candidates = knowledge.search(queries(meaning, source)[0])
        ref = candidates[0].ref
        raw = decision().model_dump(mode="json")
        raw["explanation"] = {
            "summary": "The supplied indication rule directly supports CMINDC for this collected medication reason.",
            "evidence_refs": ["bp", ref],
            "uncertainties": ["Study profile approval is not supplied."],
        }
        raw["decisions"][0]["candidate_refs"] = [ref]
        raw["target_observations"][0].update(
            candidate_ref=ref, source_context_refs=["bp"], evidence_refs=["bp", ref]
        )
        raw["annotations"] = [
            {
                **raw["annotations"][0],
                "variable": "CMINDC",
                "evidence_refs": ["bp", ref],
                "record_selector": {"status": "not_applicable", "value": None},
            }
        ]
        value = MappingDecision.model_validate(raw)
        issues = check_decision(value, source, meaning, candidates)
        self.assertEqual(value.annotations[0].variable, "CMINDC")
        self.assertEqual(
            {issue.code for issue in issues}, {"study_standard_unconfirmed"}
        )
        self.assertIn("CMINDC", render(value, candidates))


class CheckAndSessionTests(unittest.TestCase):
    """Typed synthetic model output traverses real retrieval, checks and disk publication."""

    def setUp(self) -> None:
        """Create isolated index and output root per test; no shared mutable fixtures."""
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.candidates = (self.knowledge.resolve(SYSBP),)

    def test_two_decisions_produce_complete_persisted_review(self) -> None:
        """Assert source-to-evidence-to-outcome behavior, not merely model call counts."""
        model = ScriptedModel([understanding_json(), json.dumps(proposal())])
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(
            [(a.source_scope, a.variable) for a in result.decision.annotations],
            [(("bp",), "VSORRES"), (("unit",), "VSORRESU"), (("date",), "VSDTC")],
        )
        self.assertEqual(
            len({a.observation_key for a in result.decision.annotations}), 1
        )
        self.assertEqual(result.calls, 2)
        saved = MappingDecision.model_validate(
            json.loads((self.output / "snapshot.json").read_text())["value"]
        )
        self.assertEqual(saved.annotations[1].fixed_assignments[0].value, "mmHg")
        payload = json.loads(model.requests[1][1]["content"])
        self.assertEqual(payload["decision_context"]["templates"][0]["alias"], "T1")
        self.assertEqual(len(payload["decision_context"]["templates"][0]["rows"]), 11)
        self.assertNotIn("previous_decision", payload)
        self.assertEqual(
            payload["understanding"]["bindings"][0]["target_source_refs"], ["bp"]
        )
        self.assertEqual(
            (self.output / "review.md").read_text(), render(saved, self.candidates)
        )
        self.assertEqual(result.study_approval, "pending")

    def test_plausible_wrong_targets_values_and_scopes_are_rejected(self) -> None:
        """Derived values, wrong test codes, forged evidence and neighbor output fail."""
        for kind in ("derived", "selector", "evidence", "scope", "constant"):
            value = decision().model_dump(mode="json")
            first = value["annotations"][0]
            if kind == "derived":
                first["variable"] = "VSSTRESN"
            elif kind == "selector":
                first["record_selector"]["value"]["value"] = "DIABP"
            elif kind == "evidence":
                first["evidence_refs"] = ["not-delivered"]
            elif kind == "scope":
                first["source_scope"] = ["measurement"]
            else:
                value["annotations"][1]["fixed_assignments"][0]["evidence_refs"] = [
                    SYSBP
                ]
            with self.subTest(kind=kind), self.assertRaises(WorkflowError):
                check_decision(
                    MappingDecision.model_validate(value),
                    packet(),
                    understanding(),
                    self.candidates,
                )

    def test_shared_qualifier_wrong_observation_rejected(self) -> None:
        """All IDs can exist while a unit is assigned to the wrong observation."""
        value = decision().model_dump(mode="json")
        value["target_observations"].append(
            {**value["target_observations"][0], "key": "measurement:2"}
        )
        value["annotations"][1]["observation_key"] = "measurement:2"
        wrong = MappingDecision.model_validate(value)
        self.assertTrue(needs_consistency(understanding(), packet(), wrong))
        with self.assertRaises(WorkflowError):
            check_decision(wrong, packet(), understanding(), self.candidates)

    def test_consistency_uses_source_context_boundaries_not_provenance_count(
        self,
    ) -> None:
        value = decision().model_dump(mode="json")
        value["target_observations"][0]["source_context_refs"] = ["bp", "unit", "date"]
        mapped = MappingDecision.model_validate(value)
        self.assertFalse(needs_consistency(understanding(), packet(), mapped))
        raw = packet().model_dump(mode="json")
        raw["objects"].append({**raw["objects"][0], "ref": "second-context"})
        raw["objects"][2]["context_ref"] = "second-context"
        separate = type(packet()).model_validate(raw)
        self.assertTrue(needs_consistency(understanding(), separate, mapped))

    def test_branch_not_submitted_is_visible_and_unknown_is_not_approved(self) -> None:
        """Regression: the old renderer preserved this fact in JSON but displayed nothing."""
        value = decision().model_dump(mode="json")
        value["annotations"].append(
            {
                "key": "branch-yes",
                "source_scope": ["bp"],
                "source_condition": {
                    "status": "specified",
                    "value": {"op": "eq", "source_id": "bp", "value": "Yes"},
                },
                "disposition": "not_submitted",
                "observation_key": None,
                "variable": None,
                "record_selector": {"status": "not_applicable", "value": None},
                "fixed_assignments": [],
                "evidence_refs": ["bp"],
                "reason": "Structural branch fixture, approval not established",
            }
        )
        mixed = MappingDecision.model_validate(value)
        text = render(mixed, self.candidates)
        self.assertIn("branch-yes [bp]: [NOT SUBMITTED] source_if=", text)
        issues = check_decision(mixed, packet(), understanding(), self.candidates)
        self.assertIn("submission_basis_unresolved", {item.code for item in issues})

    def test_failed_first_call_does_not_invent_a_candidate_or_zero_usage(self) -> None:
        """Empty formal output retains measured failure tokens but creates no decision."""
        reply = Reply(
            content="",
            error="response_content_empty",
            status=200,
            finish_reason="stop",
            prompt_tokens=62,
            completion_tokens=6,
            reasoning_present=True,
            response_sha256=None,
            seconds=1,
        )
        result = Session(
            packet(), self.knowledge, ScriptedModel([reply]), self.output
        ).run()
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.decision)
        self.assertEqual(result.measured_tokens, 68)
        self.assertFalse((self.output / "snapshot.json").exists())

    def test_atomic_failure_does_not_publish_half_snapshot(self) -> None:
        """The last committed bytes and revision survive an actual publication failure."""
        store = RunStore(self.output)
        store.commit("understanding", {"value": "original"})
        before = (self.output / "snapshot.json").read_bytes()
        with patch(
            "pipeline.stage_b_mapper_module.annotation_workflow.store.os.replace",
            side_effect=OSError("disk unavailable"),
        ):
            with self.assertRaises(WorkflowError):
                store.commit("decision", {"value": "replacement"})
        self.assertEqual(store.revision, 1)
        self.assertEqual((self.output / "snapshot.json").read_bytes(), before)
        with self.assertRaises(WorkflowError):
            RunStore(self.output)

    def test_evidence_capacity_is_explicit_not_truncated(self) -> None:
        """An oversized complete bundle stops before a misleading partial evidence turn."""
        result = Session(
            packet(),
            self.knowledge,
            ScriptedModel([understanding_json()]),
            self.output,
            budget=Budget(max_evidence_bytes=1000),
        ).run()
        self.assertEqual(result.stop_reason, "evidence_capacity_exceeded")
        self.assertIsNone(result.decision)
        self.assertEqual(result.calls, 1)
