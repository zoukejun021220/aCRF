"""Reproduce recorded retrieval failures against the actual pinned public KB.

The queries use source concepts/labels only. Expected template identifiers prove
the regression independently of benchmark reference files or inference outputs.
"""

import unittest
from pathlib import Path

from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import \
    Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import \
    MappingQuery
from pipeline.stage_b_mapper_module.graphrag.field_contracts import \
    StandardProfile

ROOT = Path(__file__).resolve().parents[2]


class StandardRetrievalTests(unittest.TestCase):
    """Keep meaningful richer wording and source timepoints in candidate supply."""

    def setUp(self):
        profile = StandardProfile(
            sdtm="2.0",
            sdtmig="3.4",
            ct_package="sdtmct-2025-03-28",
            cosmos_commit="031429b1d14823721991cd23ee88a11616686ce3",
        )
        self.k = Knowledge(ROOT / "kb/cosmos" / profile.cosmos_commit, profile)
        self.addCleanup(self.k.close)

    def query(self, concept, context=()):
        return MappingQuery.model_validate(
            {
                "item_ref": "item:1",
                "source_refs": ["F005"],
                "concept": concept,
                "collected_property": "distance",
                "context": list(context),
                "conditions": [],
            }
        )

    def test_richer_first_minute_concept_is_not_filtered_out(self):
        result = self.k.search(
            self.query("distance walked at 1 minute during six minute walk test")
        )
        self.assertTrue(any(c.ref.endswith(":SIXMW101") for c in result))

    def test_fourth_minute_source_modifier_survives_top_three(self):
        result = self.k.search(
            self.query(
                "distance",
                [
                    {
                        "name": "source_label",
                        "value": "Distance at 4 Minutes",
                        "source_refs": ["F005"],
                    }
                ],
            )
        )
        self.assertTrue(result[0].ref.endswith(":SIXMW104"))
        self.assertFalse(
            any(c.ref.endswith((":SIXMW101", ":SIXMW102", ":SIXMW103")) for c in result)
        )


if __name__ == "__main__":
    unittest.main()
