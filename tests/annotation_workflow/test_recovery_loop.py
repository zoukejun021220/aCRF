"""Compact judgment recovery at real SQLite and persisted provider boundaries."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from support import (
    RULES,
    SNAPSHOT,
    SYSBP,
    ScriptedModel,
    packet,
    understanding,
    understanding_json,
)

from pipeline.stage_b_mapper_module.annotation_workflow.common import (
    WorkflowError,
    canonical,
)
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.proposal import proposal_context
from pipeline.stage_b_mapper_module.annotation_workflow.session import (
    Session,
    _preserve_unaffected,
    _revision_changes,
)
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    Material,
    MaterialStore,
)
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
    queries,
)
from pipeline.stage_b_mapper_module.graphrag.conditions import leaves


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)
        candidates = {}
        for query in queries(understanding(), packet()):
            for candidate in self.knowledge.search(query):
                candidates[candidate.ref] = candidate
        self.candidates = tuple(candidates.values())
        self.alias = f"T{list(candidates).index(SYSBP) + 1}"

    def proposal(self) -> dict:
        return {
            "records": [
                {
                    "template": self.alias,
                    "bindings": [
                        {"source": ["bp"], "target": "VSORRES"},
                        {"source": ["unit"], "target": "VSORRESU"},
                        {"source": ["date"], "target": "VSDTC"},
                    ],
                    "reason": "The source labels and shared measurement context support one systolic observation.",
                }
            ],
            "other_outcomes": [],
            "open_questions": [],
        }

    def question(self, kind: str = "study", ref: str = "guide") -> dict:
        return {
            "missing": "The study-specific measurement convention",
            "affected_refs": ["bp"],
            "impact": "Confirm applicability of the selected systolic template",
            "source_kind": kind,
            "material_ref": ref,
        }

    def submit(self, value: dict | None = None, updates: tuple[dict, ...] = ()) -> str:
        """Author the new wire action explicitly; never translate legacy questions."""
        value = copy.deepcopy(value if value is not None else self.proposal())
        self.assertFalse(value.pop("open_questions", []))
        return canonical(
            {"action": {"op": "submit", **value, "question_updates": list(updates)}}
        )

    def read_action(self, handle: str) -> str:
        return canonical(
            {
                "action": {
                    "op": "read_material",
                    "handle": handle,
                    "source": ["bp"],
                    "question": "Does this guide confirm the measurement convention?",
                    "impact": "Confirm applicability of the selected systolic template.",
                }
            }
        )

    def unavailable_search(self) -> str:
        """An actual zero-hit SQLite query, not a fabricated material handle."""
        return canonical(
            {
                "action": {
                    "op": "search_standard",
                    "concept": "SyntheticAbsentZqxjConvention",
                    "collected_property": "SyntheticAbsentZqxjProperty",
                    "source": ["bp"],
                    "question": "Is there an additional template for this convention?",
                    "impact": "The proposed alternative needs standard evidence.",
                }
            }
        )

    def question_update(
        self, *, status: str = "resolved", support: tuple[str, ...] = ()
    ) -> dict:
        return {
            "question": "Q1",
            "status": status,
            "reason": (
                "The supplied evidence confirms the already proposed convention."
                if status == "resolved"
                else "The additional evidence is unavailable; retain uncertainty and stop this search."
            ),
            "support": list(support),
        }

    def consistency_source(self):
        """A real source dependency keeps accepted state visible during later actions."""
        source = packet()
        return source.model_copy(
            update={
                "objects": tuple(
                    (
                        obj.model_copy(update={"parent_ref": "measurement"})
                        if obj.ref == "bp"
                        else obj
                    )
                    for obj in source.objects
                )
            }
        )

    def test_invalid_target_has_choices_and_repairs_without_internal_bookkeeping(
        self,
    ) -> None:
        wrong = self.proposal()
        wrong["records"][0]["bindings"][0]["target"] = "BAD_TARGET"
        model = ScriptedModel(
            [
                understanding_json(),
                canonical(wrong),
                canonical(self.proposal()),
            ]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 3)
        self.assertEqual(len(result.decision.annotations), 3)
        payload = json.loads(model.requests[2][1]["content"])
        self.assertNotIn("previous_decision", payload)
        self.assertIn("decision_context", payload)
        errors = payload["correction"]["errors"]
        self.assertTrue(any("VSORRES" in item.get("allowed", []) for item in errors))
        self.assertEqual(errors[0]["affected_refs"], ["bp"])
        self.assertTrue((self.output / "02.proposal.json").exists())
        self.assertTrue((self.output / "03.accepted-decision.json").exists())

    def test_missing_source_gets_explicit_feedback_and_empty_repeats_stop(self) -> None:
        missing = self.proposal()
        missing["records"][0]["bindings"].pop()
        model = ScriptedModel(
            [
                understanding_json(),
                canonical(missing),
                canonical(self.proposal()),
            ]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        feedback = json.loads(model.requests[2][1]["content"])["correction"]["errors"]
        self.assertEqual(feedback[0]["code"], "source_not_covered")
        self.assertEqual(feedback[0]["affected_refs"], ["date"])
        empty = canonical({"records": [], "other_outcomes": [], "open_questions": []})
        repeated = Session(
            packet(),
            self.knowledge,
            ScriptedModel([understanding_json(), empty, empty]),
            self.output.parent / "empty",
        ).run()
        self.assertEqual(repeated.stop_reason, "correction_no_progress")
        self.assertEqual(repeated.calls, 3)
        self.assertIsNone(repeated.decision)

    def test_binding_schema_error_cannot_delete_unaffected_constant(self) -> None:
        original = self.proposal()
        original["records"][0]["bindings"][1]["details"] = {
            "constants": [
                {"target": "VSORRESU", "value": "mmHg", "support": ["unit", self.alias]}
            ]
        }
        invalid = copy.deepcopy(original)
        invalid["records"][0]["bindings"][0]["target"] = None
        lost = self.proposal()
        model = ScriptedModel(
            [
                understanding_json(),
                canonical(invalid),
                canonical(lost),
                canonical(original),
            ]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 4)
        feedback = json.loads(model.requests[2][1]["content"])["correction"]["errors"]
        binding_errors = [item for item in feedback if "/bindings/0/" in item["path"]]
        self.assertTrue(binding_errors)
        self.assertTrue(all(item["affected_refs"] == ["bp"] for item in binding_errors))
        rejection = json.loads((self.output / "03.validation.json").read_text())
        self.assertEqual(rejection["code"], "unaffected_judgment_changed")
        self.assertTrue(
            any(
                a.fixed_assignments
                for a in result.decision.annotations
                if "unit" in a.source_scope
            )
        )

    def test_unavailable_lookup_returns_to_model_and_requires_explicit_resolution(
        self,
    ) -> None:
        unresolved = {
            "records": [],
            "other_outcomes": [
                {
                    "source": ["bp", "unit", "date"],
                    "outcome": "unresolved",
                    "reason": "The alternative convention lacks evidence; its mapping remains unresolved.",
                }
            ],
        }
        model = ScriptedModel(
            [
                understanding_json(),
                self.unavailable_search(),
                self.submit(unresolved),
                self.submit(unresolved, (self.question_update(status="deferred"),)),
            ]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 4)
        payload = json.loads(model.requests[2][1]["content"])
        self.assertEqual(payload["lookup_results"][-1]["status"], "unavailable")
        self.assertIsNone(payload["previous_proposal"])
        self.assertEqual(payload["questions"][0]["status"], "open")
        self.assertEqual(
            json.loads((self.output / "03.validation.json").read_text())["code"],
            "question_update_required",
        )
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertFalse((self.output / "03.accepted-decision.json").exists())
        self.assertTrue(
            all(a.disposition == "unresolved" for a in result.decision.annotations)
        )
        ledger = json.loads((self.output / "question-ledger.json").read_text())
        self.assertEqual(ledger[0]["status"], "deferred")
        self.assertIn("evidence_question_deferred", {i.code for i in result.issues})

    def test_repeated_evidence_action_keeps_previously_accepted_decision(self) -> None:
        output = self.output.parent / "repeated"
        model = ScriptedModel(
            [
                understanding_json(),
                self.submit(),
                self.unavailable_search(),
                self.unavailable_search(),
                self.unavailable_search(),
            ]
        )
        repeated = Session(
            self.consistency_source(),
            self.knowledge,
            model,
            output,
        ).run()
        self.assertEqual(repeated.stop_reason, "evidence_no_progress")
        self.assertEqual(repeated.calls, 5)
        self.assertIsNotNone(repeated.decision)
        accepted = json.loads((output / "02.accepted-decision.json").read_text())
        self.assertEqual(repeated.decision.model_dump(mode="json"), accepted)
        self.assertEqual(len(list(output.glob("*.accepted-decision.json"))), 1)
        self.assertTrue((output / "review.md").exists())
        self.assertEqual(
            json.loads((output / "material-002.json").read_text())["cached"], True
        )
        self.assertFalse((output / "05.action.json").exists())
        self.assertEqual(
            json.loads((output / "question-ledger.json").read_text())[0]["status"],
            "open",
        )

    def test_no_new_standard_lookup_is_returned_once(self) -> None:
        model = ScriptedModel(
            [
                understanding_json(),
                self.read_action(self.alias),
                self.submit(updates=(self.question_update(support=(self.alias,)),)),
            ]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        payload = json.loads(model.requests[2][1]["content"])
        self.assertEqual(payload["lookup_results"][-1]["status"], "no_new_evidence")
        self.assertEqual(payload["questions"][0]["status"], "open")
        material = json.loads((self.output / "material-001.json").read_text())
        self.assertEqual(material["request"]["source_kind"], "standard")
        self.assertEqual(material["request"]["material_ref"], SYSBP)
        self.assertEqual(len(list(self.output.glob("material-*.json"))), 1)
        self.assertEqual(
            result.decision.annotations[0].reason,
            self.proposal()["records"][0]["reason"],
        )

    def test_revision_can_restore_an_unrelated_binding_after_feedback(self) -> None:
        resolved = self.proposal()
        resolved["records"][0][
            "reason"
        ] = "The lookup returned the existing standard rows; the source and template still support this provisional mapping while study approval remains pending."
        damaged = copy.deepcopy(resolved)
        damaged["records"][0]["bindings"][1]["target"] = "BAD_UNIT_TARGET"
        model = ScriptedModel(
            [
                understanding_json(),
                self.submit(),
                self.read_action(self.alias),
                self.submit(damaged, (self.question_update(support=(self.alias,)),)),
                self.submit(resolved, (self.question_update(support=(self.alias,)),)),
            ]
        )
        result = Session(
            self.consistency_source(), self.knowledge, model, self.output
        ).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 5)
        self.assertEqual(
            json.loads((self.output / "04.validation.json").read_text())["code"],
            "unaffected_judgment_changed",
        )
        self.assertEqual(
            json.loads(model.requests[4][1]["content"])["questions"][0]["status"],
            "open",
        )
        self.assertTrue((self.output / "02.accepted-decision.json").exists())
        self.assertFalse((self.output / "04.accepted-decision.json").exists())
        self.assertEqual(
            next(
                item.variable
                for item in result.decision.annotations
                if item.source_scope == ("unit",)
            ),
            "VSORRESU",
        )

    def test_indexed_unread_support_cannot_close_question_after_acceptance(
        self,
    ) -> None:
        unread = Material(
            ref="unread-guide",
            kind="study",
            study_id=packet().study_id,
            text="A policy body that must not be used before an explicit read.",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
        )
        model = ScriptedModel(
            [
                understanding_json(),
                self.submit(),
                self.read_action(self.alias),
                self.submit(updates=(self.question_update(support=(unread.ref,)),)),
                self.submit(updates=(self.question_update(support=(self.alias,)),)),
            ]
        )
        result = Session(
            self.consistency_source(),
            self.knowledge,
            model,
            self.output,
            MaterialStore((unread,)),
        ).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 5)
        rejection = json.loads((self.output / "04.validation.json").read_text())
        self.assertEqual(rejection["code"], "question_support_unknown")
        self.assertFalse((self.output / "04.accepted-decision.json").exists())
        preserved = json.loads((self.output / "02.accepted-decision.json").read_text())
        self.assertEqual(result.decision.model_dump(mode="json"), preserved)
        payload = json.loads(model.requests[4][1]["content"])
        self.assertEqual(payload["questions"][0]["status"], "open")
        self.assertIn({"ref": unread.ref, "kind": "study"}, payload["material_index"])
        self.assertNotIn(unread.ref, payload["decision_context"]["support_materials"])
        self.assertEqual(len(list(self.output.glob("material-*.json"))), 1)
        ledger = json.loads((self.output / "question-ledger.json").read_text())
        self.assertEqual(ledger[0]["support"], [self.alias])

    def test_consistency_requests_are_serviced_then_reviewed_once_per_new_evidence(
        self,
    ) -> None:
        source = self.consistency_source()
        resolved = self.proposal()
        resolved["records"][0][
            "reason"
        ] = "The admitted study guide confirms the existing systolic measurement convention and closes the applicability question."
        guide = Material(
            ref="guide",
            kind="study",
            study_id=source.study_id,
            text="Use the systolic measurement convention shown on the CRF.",
            source_ref=source.objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        model = ScriptedModel(
            [
                understanding_json(),
                self.submit(),
                self.read_action("guide"),
                self.submit(resolved, (self.question_update(support=("guide",)),)),
                self.submit(resolved),
            ]
        )
        result = Session(
            source, self.knowledge, model, self.output, MaterialStore((guide,))
        ).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 5)
        self.assertTrue(result.consistency_reviewed)
        phases = [
            json.loads((self.output / f"{index:02d}.request.json").read_text())["phase"]
            for index in range(1, 6)
        ]
        self.assertEqual(
            phases,
            ["understanding", "decision", "consistency", "consistency", "consistency"],
        )
        payload = json.loads(model.requests[3][1]["content"])
        self.assertEqual(payload["lookup_results"][-1]["status"], "available")
        self.assertEqual(payload["decision_context"]["support_materials"], ["guide"])
        self.assertEqual(payload["previous_proposal"], self.proposal())
        self.assertFalse((self.output / "03.accepted-decision.json").exists())
        self.assertEqual(len(list(self.output.glob("material-*.json"))), 1)

    def test_retrieval_keeps_previously_delivered_template_aliases(self) -> None:
        session = Session(packet(), self.knowledge, ScriptedModel([]), self.output)
        session.understanding = understanding()
        session.delivered = tuple(reversed(self.candidates))
        before = tuple(candidate.ref for candidate in session.delivered)
        session._retrieve()
        self.assertEqual(
            tuple(candidate.ref for candidate in session.delivered)[: len(before)],
            before,
        )
        context = proposal_context(packet(), understanding(), session.delivered)
        self.assertEqual(context["templates"][0]["alias"], "T1")
        self.assertEqual(context["templates"][0]["title"], self.candidates[-1].title)

    def test_parse_feedback_has_field_path_without_echoing_input(self) -> None:
        session = Session(packet(), self.knowledge, ScriptedModel([]), self.output)
        draft = self.proposal()
        error = WorkflowError(
            "invalid_structured_output",
            feedback=(
                {
                    "path": "/records/0/ProposedRecord/bindings/1/Binding/target",
                    "code": "string_type",
                    "message": "Input should be a valid string",
                },
            ),
        )
        result = session._correction_feedback(error, draft)
        self.assertEqual(result[0]["affected_refs"], ["unit"])

    def test_unavailable_source_context_keeps_partial_mapping_possible(self) -> None:
        source_value = understanding().model_dump(mode="json")
        source_value["items"][-1].update(
            {"concept": None, "collected_property": None, "source_role": "unknown"}
        )
        source_value["bindings"].pop()
        gap = self.question("source", "missing-date-context")
        gap["affected_refs"] = ["date"]
        source_value["material_unknowns"] = [gap]
        partial = CollectionUnderstanding.model_validate(source_value)
        value = self.proposal()
        value["records"][0]["bindings"].pop()
        value["other_outcomes"] = [
            {
                "source": ["date"],
                "outcome": "unresolved",
                "reason": "The missing date context is unavailable; the date target remains unconfirmed.",
            }
        ]
        model = ScriptedModel(
            [understanding_json(partial), understanding_json(partial), canonical(value)]
        )
        result = Session(packet(), self.knowledge, model, self.output).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.calls, 3)
        self.assertEqual(
            sum(item.disposition == "mapped" for item in result.decision.annotations), 2
        )
        self.assertTrue(
            any(
                item.disposition == "unresolved" and item.source_scope == ("date",)
                for item in result.decision.annotations
            )
        )
        self.assertEqual(
            json.loads(model.requests[1][1]["content"])["lookup_results"][-1]["status"],
            "unavailable",
        )

    def test_explicit_condition_handle_keeps_its_meaning_after_source_reinterpretation(
        self,
    ) -> None:
        source_value = understanding().model_dump(mode="json")
        source_value["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": [ref],
                    "kind": "collection",
                    "predicate": {"op": "eq", "source_id": ref, "value": number},
                    "basis": {
                        "kind": "explicit",
                        "source_refs": [ref],
                        "reason": "Synthetic source branch for condition identity regression.",
                    },
                }
                for ref, number in (("bp", 1), ("date", 2))
            ],
        }
        before = CollectionUnderstanding.model_validate(source_value)
        source_value["conditions"]["entries"].reverse()
        after = CollectionUnderstanding.model_validate(source_value)
        first = self.proposal()
        first["records"][0]["bindings"][0]["when"] = "C1"
        revised = self.proposal()
        revised["records"][0]["bindings"][0]["when"] = "C2"
        revised["records"][0][
            "reason"
        ] = "The source guide confirms the same branch predicate despite the changed condition ordering."
        guide = Material(
            ref="guide",
            kind="source",
            study_id=packet().study_id,
            text="The source condition order may differ; preserve the explicit predicate.",
            source_ref=packet().objects[0].source_ref,
            origin="completion_guidance",
        )
        model = ScriptedModel(
            [
                understanding_json(before),
                self.submit(first),
                self.read_action("guide"),
                understanding_json(after),
                self.submit(revised, (self.question_update(support=("guide",)),)),
                self.submit(revised),
            ]
        )
        result = Session(
            self.consistency_source(),
            self.knowledge,
            model,
            self.output,
            MaterialStore((guide,)),
        ).run()
        self.assertEqual(result.stop_reason, "completed_for_review")
        self.assertEqual(result.calls, 6)
        payload = json.loads(model.requests[4][1]["content"])
        retained = payload["previous_proposal"]["records"][0]["bindings"][0]["when"]
        self.assertEqual(retained, {"op": "eq", "source_id": "bp", "value": 1})
        mapped = next(
            item for item in result.decision.annotations if item.source_scope == ("bp",)
        )
        self.assertTrue(
            all(item.value == 1 for item in leaves(mapped.source_condition.value))
        )

    def test_record_order_cannot_silently_redirect_relationship_or_supp_parent(
        self,
    ) -> None:
        before = self.proposal()
        base = before["records"].pop()
        before["records"] = [
            {
                "template": self.alias,
                "bindings": [binding],
                "reason": f"Separate synthetic group {index}.",
            }
            for index, binding in enumerate(base["bindings"], 1)
        ]
        before["records"][0]["links"] = [
            {
                "record": 2,
                "relation": "related",
                "support": ["bp", "unit"],
                "reason": "The first group relates to the second.",
            }
        ]
        shifted = copy.deepcopy(before)
        shifted["records"][1:] = reversed(shifted["records"][1:])
        with self.assertRaises(WorkflowError) as context:
            _preserve_unaffected(before, shifted, set())
        self.assertEqual(context.exception.code, "unaffected_judgment_changed")
        with self.assertRaises(WorkflowError):
            _revision_changes(before, shifted)
        # An actual reviewed relation change is still expressible and audited.
        shifted["records"][0][
            "reason"
        ] = "The review redirects the first group to the date group."
        _preserve_unaffected(before, shifted, {"bp"})
        self.assertEqual(len(_revision_changes(before, shifted)), 1)
        repaired = copy.deepcopy(before)
        repaired["records"][1:] = reversed(repaired["records"][1:])
        repaired["records"][0]["links"][0]["record"] = 3
        _preserve_unaffected(before, repaired, set())
        self.assertEqual(_revision_changes(before, repaired), [])

        before["records"][0].pop("links")
        before["other_outcomes"] = [
            {
                "source": ["supp"],
                "outcome": "supplemental",
                "parent_record": 2,
                "qnam": "DETAIL",
                "qlabel": "Detail",
                "support": ["bp"],
                "reason": "The detail belongs to the unit group.",
            }
        ]
        shifted = copy.deepcopy(before)
        shifted["records"][1:] = reversed(shifted["records"][1:])
        with self.assertRaises(WorkflowError):
            _preserve_unaffected(before, shifted, set())
        with self.assertRaises(WorkflowError):
            _revision_changes(before, shifted)
        shifted["other_outcomes"][0][
            "reason"
        ] = "The reviewed detail instead belongs to the date group."
        _preserve_unaffected(before, shifted, {"supp"})
        self.assertEqual(len(_revision_changes(before, shifted)), 1)


if __name__ == "__main__":
    unittest.main()
