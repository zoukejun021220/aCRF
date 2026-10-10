"""Decision inputs separate annotation subjects from read-only source evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import RULES, SNAPSHOT, ScriptedModel, packet, proposal, understanding_json

from pipeline.stage_b_mapper_module.annotation_workflow.common import canonical
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.session import Session


class DecisionScopeViewTests(unittest.TestCase):
    def test_readonly_neighbors_remain_evidence_without_appearing_as_subjects(self):
        original = packet()
        neighbor = original.object("bp").model_copy(
            update={"ref": "neighbor", "text": "Another measurement, outside this task"}
        )
        source = original.model_copy(update={"objects": (*original.objects, neighbor)})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            knowledge = Knowledge(SNAPSHOT, source.standard, RULES)
            self.addCleanup(knowledge.close)
            model = ScriptedModel([understanding_json(), canonical(proposal())])
            result = Session(source, knowledge, model, output).run()
            self.assertEqual(result.stop_reason, "completed_for_review")
            self.assertEqual(
                [item.variable for item in result.decision.annotations],
                ["VSORRES", "VSORRESU", "VSDTC"],
            )
            decision_input = json.loads(model.requests[1][1]["content"])
            self.assertEqual(
                {item["ref"] for item in decision_input["source"]["objects"]},
                set(source.output_scope),
            )
            context = decision_input["readonly_context"]
            self.assertEqual(
                {item["ref"] for item in context}, {"measurement", "neighbor"}
            )
            self.assertEqual(
                next(item for item in context if item["ref"] == "neighbor"),
                neighbor.model_dump(mode="json"),
            )
            self.assertEqual(
                next(item for item in context if item["ref"] == "measurement"),
                source.object("measurement").model_dump(mode="json"),
            )
            # Understanding and persisted source retain the complete source graph.
            self.assertEqual(
                json.loads(model.requests[0][1]["content"])["source"],
                source.model_dump(mode="json"),
            )
            self.assertEqual(
                json.loads((output / "source.json").read_text()),
                source.model_dump(mode="json"),
            )


if __name__ == "__main__":
    unittest.main()
