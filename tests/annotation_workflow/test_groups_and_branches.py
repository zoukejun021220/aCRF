"""Group isolation and finite branch coverage use source objects, not gold retrieval."""

import unittest

from support import decision, packet

from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.coverage import (
    applies,
    option_coverage,
)
from pipeline.stage_b_mapper_module.annotation_workflow.decision import MappingDecision
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    SourcePacket,
    group_packets,
)
from pipeline.stage_b_mapper_module.graphrag.conditions import parse_source_condition


class GroupTests(unittest.TestCase):
    """Shared objects cannot silently change or cross the caller's study boundary."""

    def test_three_valued_conditions_preserve_unknown_dependencies(self) -> None:
        """Boolean operations never turn unknown neighboring answers into certainty."""
        yes = {"op": "eq", "source_id": "q", "value": "Yes"}
        no = {"op": "in", "source_id": "q", "values": ["No"]}
        foreign = {"op": "eq", "source_id": "neighbor", "value": "Yes"}
        for expression, expected in [
            (yes, True),
            (no, False),
            (foreign, None),
            ({"op": "not", "arg": yes}, False),
            ({"op": "not", "arg": foreign}, None),
            ({"op": "all", "args": [yes, no]}, False),
            ({"op": "all", "args": [yes, foreign]}, None),
            ({"op": "all", "args": [yes]}, True),
            ({"op": "any", "args": [no, foreign]}, None),
            ({"op": "any", "args": [yes, foreign]}, True),
            ({"op": "any", "args": [no]}, False),
        ]:
            with self.subTest(expression=expression):
                self.assertIs(
                    applies(parse_source_condition(expression), "q", ("Yes",)), expected
                )

    def test_group_retains_shared_objects_and_rejects_foreign_study(self) -> None:
        """Output scope unions preserve actual source identity and immutable originals."""
        source = packet()
        grouped = group_packets("group", (source, source))
        self.assertEqual(grouped.objects, source.objects)
        self.assertEqual(grouped.output_scope, source.output_scope)
        self.assertEqual(grouped.task_ref, "group")
        foreign = source.model_copy(update={"study_id": "other"})
        with self.assertRaises(WorkflowError):
            group_packets("group", (source, foreign))
        changed = source.model_copy(
            update={
                "objects": (
                    source.objects[0].model_copy(update={"text": "conflicting"}),
                    *source.objects[1:],
                )
            }
        )
        with self.assertRaises(WorkflowError):
            group_packets("group", (source, changed))
        with self.assertRaises(WorkflowError):
            group_packets("group", ())

    def test_missing_option_and_overlapping_dispositions_are_detected(self) -> None:
        """One field-level output cannot hide a missing or contradictory No branch."""
        raw = packet().model_dump(mode="json")
        field = {**raw["objects"][1], "datatype": "single_choice", "context_ref": None}
        raw["objects"] = [
            field,
            *[
                {
                    **field,
                    "ref": name,
                    "kind": "option",
                    "parent_ref": "bp",
                    "text": label,
                }
                for name, label in [("yes", "Yes"), ("no", "No")]
            ],
        ]
        raw["output_scope"] = ["bp", "yes", "no"]
        source = SourcePacket.model_validate(raw)
        value = decision().model_dump(mode="json")
        value["annotations"] = [
            {
                **value["annotations"][0],
                "source_condition": {
                    "status": "specified",
                    "value": {"op": "eq", "source_id": "bp", "value": "Yes"},
                },
            }
        ]
        self.assertEqual(
            [
                (issue.code, issue.refs)
                for issue in option_coverage(
                    MappingDecision.model_validate(value), source
                )
            ],
            [("option_branch_not_covered", ("no",))],
        )
        value["annotations"].append(
            {
                "key": "no",
                "source_scope": ["bp"],
                "source_condition": {
                    "status": "specified",
                    "value": {"op": "eq", "source_id": "bp", "value": "No"},
                },
                "disposition": "not_submitted",
                "observation_key": None,
                "variable": None,
                "record_selector": {"status": "not_applicable", "value": None},
                "fixed_assignments": [],
                "evidence_refs": ["bp"],
                "reason": "Structural fixture, no study approval",
            }
        )
        self.assertEqual(
            option_coverage(MappingDecision.model_validate(value), source), ()
        )
        value["annotations"][0]["source_condition"] = {
            "status": "not_applicable",
            "value": None,
        }
        self.assertEqual(
            [
                (issue.code, issue.refs)
                for issue in option_coverage(
                    MappingDecision.model_validate(value), source
                )
            ],
            [("option_branch_conflict", ("no",))],
        )
