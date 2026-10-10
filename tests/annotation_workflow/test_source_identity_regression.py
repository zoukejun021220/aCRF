"""Source identity regressions shaped by local cohort 007/018/025/046.

Fixtures are synthetic contract probes, not clinical gold. Passing coverage never
establishes applicability of the selected template or a source-value conversion.
"""

from __future__ import annotations

import copy
import json
import unittest

from support import packet, understanding

from pipeline.stage_b_mapper_module.annotation_workflow import proposal as proposals
from pipeline.stage_b_mapper_module.annotation_workflow.checks import check_decision
from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import (
    Candidate,
    TemplateRow,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import wire_schema
from pipeline.stage_b_mapper_module.annotation_workflow.source import SourcePacket
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
)


def choice_packet(datatype: str = "radio") -> SourcePacket:
    raw = packet().model_dump(mode="json")
    field = {**raw["objects"][1], "ref": "question", "datatype": datatype}
    raw["objects"] = [
        raw["objects"][0],
        field,
        *[
            {
                **field,
                "ref": ref,
                "kind": "option",
                "parent_ref": "question",
                "text": label,
                "datatype": "unknown",
                "stored_code": code,
            }
            for ref, label, code in [
                ("option-a", "None or minimal", "0"),
                ("option-b", "Mild", "1"),
            ]
        ],
    ]
    raw["output_scope"] = ["question", "option-a", "option-b"]
    return SourcePacket.model_validate(raw)


def choice_understanding() -> CollectionUnderstanding:
    raw = understanding().model_dump(mode="json")
    raw["items"] = [
        {
            **raw["items"][0],
            "source_refs": ["question"],
            "basis": {
                "kind": "explicit",
                "source_refs": ["question"],
                "reason": "The whole question collects one response from its options.",
            },
        }
    ]
    raw["bindings"] = []
    raw["explanation"]["evidence_refs"] = ["question"]
    return CollectionUnderstanding.model_validate(raw)


def candidate(*, values: str = "", codelist: str = "") -> Candidate:
    return Candidate(
        ref="synthetic:template",
        kind="released_template",
        title="Synthetic graded response",
        domain="QS",
        sdtmig="3.4",
        source_url="https://example.invalid/synthetic",
        source_sha256="b" * 64,
        source_version="synthetic",
        rows=tuple(
            TemplateRow(
                variable=variable,
                source_locator=f"row:{i}",
                values_json=json.dumps(values),
            )
            for i, (variable, values) in enumerate(
                [
                    (
                        "QSTESTCD",
                        {"role": "Topic", "assigned_value": "SYNTHETIC"},
                    ),
                    (
                        "QSORRES",
                        {
                            "origin_type": "Collected",
                            "value_list": values,
                            "codelist": codelist,
                        },
                    ),
                ]
            )
        ),
    )


def draft(*sources: str) -> dict:
    return {
        "records": [
            {
                "template": "T1",
                "bindings": [{"source": list(sources), "target": "QSORRES"}],
                "reason": "The selected template is proposed for review.",
            }
        ],
        "other_outcomes": [],
        "open_questions": [],
    }


class SourceIdentityRegressionTests(unittest.TestCase):
    def compile(self, raw, source=None, understood=None, target=None):
        return proposals.compile_proposal(
            proposals.Proposal.model_validate(raw),
            source or choice_packet(),
            understood or choice_understanding(),
            (target or candidate(),),
        )

    def issues(self, decision, source=None, understood=None, target=None):
        return check_decision(
            decision,
            source or choice_packet(),
            understood or choice_understanding(),
            (target or candidate(),),
        )

    def test_parent_interpretation_resolves_option_binding_without_rewriting(self):
        """025/046: parent-ref is enough to identify the interpreted source item."""
        raw = draft("option-a", "option-b")
        original = copy.deepcopy(raw)
        result = self.compile(raw)
        self.assertEqual(result.decisions[0].item_refs, ("item:1",))
        self.assertEqual(result.annotations[0].source_scope, ("option-a", "option-b"))
        self.assertEqual(raw, original)
        self.assertFalse(result.annotations[0].fixed_assignments)

    def test_all_single_choice_options_cover_parent_and_parent_covers_options(self):
        """007/018: both representations describe one complete single-choice answer."""
        for sources in [("option-a", "option-b"), ("question",)]:
            with self.subTest(sources=sources):
                result = self.compile(draft(*sources))
                codes = {issue.code for issue in self.issues(result)}
                self.assertFalse(
                    codes & {"source_not_covered", "option_branch_not_covered"}
                )
                self.assertEqual(len(result.annotations), 1)
                self.assertEqual(result.annotations[0].disposition, "mapped")

    def test_partial_options_do_not_cover_parent_or_missing_sibling(self):
        result = self.compile(draft("option-a"))
        missing = {
            ref
            for issue in self.issues(result)
            if issue.code == "source_not_covered"
            for ref in issue.refs
        }
        self.assertEqual(missing, {"question", "option-b"})

    def test_checkbox_has_no_single_choice_coverage_inference(self):
        source = choice_packet("checkbox")
        for sources, expected in [
            (("question",), {"option-a", "option-b"}),
            (("option-a", "option-b"), {"question"}),
        ]:
            result = self.compile(draft(*sources), source=source)
            missing = {
                ref
                for issue in self.issues(result, source=source)
                if issue.code == "source_not_covered"
                for ref in issue.refs
            }
            self.assertEqual(missing, expected)

    def test_known_parent_condition_is_inherited_by_option_binding(self):
        understood = choice_understanding().model_dump(mode="json")
        understood["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": ["question"],
                    "kind": "visibility",
                    "predicate": {
                        "op": "eq",
                        "source_id": "measurement",
                        "value": "active",
                    },
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["measurement"],
                        "reason": "Synthetic explicit collection condition.",
                    },
                }
            ],
        }
        understood = CollectionUnderstanding.model_validate(understood)
        result = self.compile(draft("option-a", "option-b"), understood=understood)
        self.assertEqual(result.annotations[0].source_condition.status, "specified")
        raw = draft("option-a", "option-b")
        raw["records"][0]["bindings"][0]["when"] = "unknown"
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw, understood=understood)
        self.assertEqual(raised.exception.code, "proposal_condition_weakened")

    def test_explicit_branches_remain_distinct_and_missing_branch_is_visible(self):
        raw = draft("question")
        raw["records"][0]["bindings"][0]["when"] = {
            "op": "eq",
            "source_id": "question",
            "value": "0",
        }
        result = self.compile(raw)
        self.assertIn(
            "option_branch_not_covered", {i.code for i in self.issues(result)}
        )
        raw["other_outcomes"] = [
            {
                "source": ["question"],
                "outcome": "unresolved",
                "reason": "The other response needs evidence.",
                "details": {
                    "when": {"op": "eq", "source_id": "question", "value": "1"}
                },
            }
        ]
        result = self.compile(raw)
        self.assertFalse(
            {i.code for i in self.issues(result)}
            & {
                "source_not_covered",
                "option_branch_not_covered",
                "option_branch_conflict",
            }
        )
        self.assertEqual(len(result.annotations), 2)

    def test_distinct_option_records_are_not_merged(self):
        raw = draft("option-a")
        raw["records"].append(draft("option-b")["records"][0])
        result = self.compile(raw)
        self.assertEqual(len(result.target_observations), 2)
        self.assertEqual(len({a.observation_key for a in result.annotations}), 2)
        self.assertEqual(
            [a.source_scope for a in result.annotations],
            [("option-a",), ("option-b",)],
        )
        self.assertNotIn("source_not_covered", {i.code for i in self.issues(result)})

    def test_unknown_or_foreign_condition_does_not_prove_branch_coverage(self):
        for when in (
            "unknown",
            {"op": "eq", "source_id": "measurement", "value": "active"},
        ):
            with self.subTest(when=when):
                raw = draft("question")
                raw["records"][0]["bindings"][0]["when"] = when
                result = self.compile(raw)
                issues = self.issues(result)
                self.assertEqual(
                    {
                        ref
                        for issue in issues
                        if issue.code == "source_not_covered"
                        for ref in issue.refs
                    },
                    {"option-a", "option-b"},
                )
                self.assertIn(
                    "option_condition_requires_review", {i.code for i in issues}
                )

    def test_visible_unbound_sibling_prevents_parent_inference(self):
        source = choice_packet()
        sibling = source.object("option-b").model_copy(
            update={"ref": "readonly-option", "text": "Severe", "stored_code": "2"}
        )
        source = source.model_copy(update={"objects": (*source.objects, sibling)})
        result = self.compile(draft("option-a", "option-b"), source=source)
        self.assertIn(
            ("source_not_covered", ("question",)),
            [(i.code, i.refs) for i in self.issues(result, source=source)],
        )
        self.assertFalse(
            any("readonly-option" in a.source_scope for a in result.annotations)
        )

    def test_parent_mapping_does_not_hide_explicit_conflicting_option_outcome(self):
        raw = draft("question")
        raw["other_outcomes"] = [
            {
                "source": ["option-a", "option-b"],
                "outcome": "context_only",
                "reason": "Deliberately conflicting disposition from actual case018 shape.",
            }
        ]
        result = self.compile(raw)
        self.assertEqual(
            {
                ref
                for i in self.issues(result)
                if i.code == "option_branch_conflict"
                for ref in i.refs
            },
            {"option-a", "option-b"},
        )
        self.assertEqual(result.annotations[-1].disposition, "context_only")

    def test_parent_coverage_does_not_hide_a_known_child_collection_condition(self):
        raw_understanding = choice_understanding().model_dump(mode="json")
        raw_understanding["conditions"] = {
            "status": "specified",
            "entries": [
                {
                    "affected_refs": ["option-a"],
                    "kind": "visibility",
                    "predicate": {
                        "op": "eq",
                        "source_id": "measurement",
                        "value": "active",
                    },
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["measurement"],
                        "reason": "This explicit condition affects the option only.",
                    },
                }
            ],
        }
        understood = CollectionUnderstanding.model_validate(raw_understanding)
        result = self.compile(draft("question"), understood=understood)
        self.assertIn(
            ("source_condition_not_preserved", ("option-a",)),
            [(i.code, i.refs) for i in self.issues(result, understood=understood)],
        )

    def test_readonly_sources_are_support_not_output_and_never_silently_removed(self):
        raw = draft("question")
        raw["records"][0]["bindings"][0]["details"] = {"support": ["measurement"]}
        self.assertIn("measurement", self.compile(raw).annotations[0].evidence_refs)
        raw["other_outcomes"] = [
            {
                "source": ["measurement"],
                "outcome": "context_only",
                "reason": "This is readonly evidence.",
            }
        ]
        with self.assertRaises(WorkflowError) as raised:
            self.compile(raw)
        self.assertEqual(raised.exception.code, "proposal_source_scope_invalid")
        self.assertEqual(raw["other_outcomes"][0]["source"], ["measurement"])

    def test_laterality_or_sibling_field_is_not_parent_identity(self):
        source = choice_packet()
        sibling = source.object("question").model_copy(update={"ref": "other-field"})
        source = source.model_copy(
            update={
                "objects": (*source.objects, sibling),
                "output_scope": (*source.output_scope, sibling.ref),
            }
        )
        with self.assertRaises(WorkflowError) as raised:
            self.compile(draft("other-field"), source=source)
        self.assertEqual(raised.exception.code, "proposal_source_not_understood")

    def test_018_named_codelist_without_terms_retains_value_mapping_review(self):
        target = candidate(codelist="SYNTHETIC_TERMS")
        result = self.compile(draft("question"), target=target)
        self.assertIn(
            "source_value_mapping_unresolved",
            {i.code for i in self.issues(result, target=target)},
        )
        self.assertFalse(result.annotations[0].fixed_assignments)
        matched = candidate(values="0;1")
        result = self.compile(draft("question"), target=matched)
        self.assertNotIn(
            "source_value_mapping_unresolved",
            {i.code for i in self.issues(result, target=matched)},
        )
        mismatch = candidate(values="N;Y")
        result = self.compile(draft("question"), target=mismatch)
        self.assertIn(
            "source_value_mapping_unresolved",
            {i.code for i in self.issues(result, target=mismatch)},
        )
        self.assertFalse(result.annotations[0].fixed_assignments)

    def test_bad_candidate_is_not_repaired_by_source_identity(self):
        """025 seam repair leaves the selected wrong topic and unresolved concern visible."""
        target = candidate(codelist="SYNTHETIC_TERMS")
        topic = target.rows[0].model_copy(
            update={
                "values_json": json.dumps(
                    {"role": "Topic", "assigned_value": "OTHER_CONDITION"}
                )
            }
        )
        target = target.model_copy(update={"rows": (topic, target.rows[1])})
        raw = draft("option-a", "option-b")
        raw["open_questions"] = [
            {
                "affected_refs": ["question"],
                "source_kind": "standard",
                "material_ref": "T1",
                "missing": "Applicable source concept template",
                "impact": "This candidate topic is not established by the source.",
            }
        ]
        result = self.compile(raw, target=target)
        self.assertEqual(
            result.annotations[0].record_selector.value.value, "OTHER_CONDITION"
        )
        self.assertTrue(result.evidence_requests)
        self.assertIn(
            "evidence_request_open",
            {i.code for i in self.issues(result, target=target)},
        )

    def test_scoped_schema_rejects_readonly_outputs_but_keeps_support(self):
        schema = wire_schema(proposals.Proposal)
        original = copy.deepcopy(schema)
        restricted = proposals.restrict_proposal_sources(
            schema, choice_packet().output_scope
        )
        for name in (
            "Binding",
            "ConditionalBinding",
            "QualifiedBinding",
            "OtherOutcome",
            "QualifiedOutcome",
            "SupplementalOutcome",
        ):
            allowed = restricted["$defs"][name]["properties"]["source"]["items"]["enum"]
            self.assertEqual(allowed, list(choice_packet().output_scope))
            self.assertNotIn("measurement", allowed)
        self.assertNotIn(
            "enum",
            restricted["$defs"]["BindingDetails"]["properties"]["support"]["items"],
        )
        self.assertEqual(schema, original)
        context = proposals.proposal_context(
            choice_packet(), choice_understanding(), (candidate(),)
        )
        self.assertEqual(context["readonly_support_sources"], ["measurement"])
        self.assertEqual(context["choice_groups"][0]["parent"], "question")


if __name__ == "__main__":
    unittest.main()
