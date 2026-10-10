"""Ranking regressions over real SQLite FTS and the pinned COSMoS export."""

import unittest

from support import SNAPSHOT, packet

from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import (
    Candidate,
    Knowledge,
)
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    MappingQuery,
)


class RankingRegressionTests(unittest.TestCase):
    """Keep semantic scores primary without discarding FTS relevance on ties."""

    def knowledge(self, *, include_templates: bool = False) -> Knowledge:
        knowledge = Knowledge(
            SNAPSHOT, packet().standard, include_templates=include_templates
        )
        self.addCleanup(knowledge.close)
        return knowledge

    def add(self, knowledge: Knowledge, ref: str, title: str, text: str) -> None:
        knowledge._add(
            Candidate(
                ref=ref,
                kind="released_template",
                title=title,
                domain="VS",
                sdtmig="3.4",
                source_url="https://example.invalid/synthetic-ranking-fixture",
                source_sha256="a" * 64,
                source_version="synthetic",
                rows=(),
            ),
            text,
        )

    def query(self, concept: str) -> MappingQuery:
        return MappingQuery(
            item_ref="item:1",
            source_refs=("measurement",),
            concept=concept,
            collected_property="reading",
            context=(),
            conditions=(),
        )

    def test_equal_scores_keep_fts_relevance_then_stable_ref(self) -> None:
        knowledge = self.knowledge()
        # Reverse insertion order distinguishes stable ref fallback from rowid.
        for suffix in ("c", "b", "a"):
            self.add(
                knowledge,
                f"template:{suffix}",
                "Broad measurement",
                "marker " + "unrelated " * 40,
            )
        self.add(knowledge, "template:z", "Focused measurement", "marker")
        ranks = dict(
            knowledge._db.execute(
                "SELECT ref, rank FROM candidates WHERE text MATCH ?",
                ('"marker" OR "reading"',),
            ).fetchall()
        )
        self.assertLess(ranks["template:z"], ranks["template:a"])
        self.assertEqual(ranks["template:a"], ranks["template:b"])
        self.assertEqual(ranks["template:b"], ranks["template:c"])
        self.assertEqual(
            [candidate.ref for candidate in knowledge.search(self.query("marker"))],
            ["template:z", "template:a", "template:b"],
        )

    def test_higher_semantic_score_still_precedes_better_fts_rank(self) -> None:
        knowledge = self.knowledge()
        self.add(knowledge, "template:a", "Broad measurement", "marker")
        self.add(knowledge, "template:z", "marker", "marker " + "unrelated " * 40)
        fts = knowledge._db.execute(
            "SELECT ref FROM candidates WHERE text MATCH ? ORDER BY rank",
            ('"marker" OR "reading"',),
        ).fetchall()
        self.assertEqual(fts[0][0], "template:a")
        self.assertEqual(knowledge.search(self.query("marker"))[0].ref, "template:z")

    def test_frozen_hemoglobin_query_retrieves_blood_template_in_top_three(
        self,
    ) -> None:
        knowledge = self.knowledge(include_templates=True)
        # Exact query recorded in the local-random50 understanding-fix case 040.
        # Keep the regression independent of local experiment-output availability.
        query = MappingQuery.model_validate(
            {
                "item_ref": "item:1",
                "source_refs": ["p036:CL_HGB"],
                "concept": "Hemoglobin",
                "collected_property": "measured_value",
                "context": [
                    {
                        "name": "unit_for",
                        "value": "g/dL",
                        "source_refs": ["p036:CL_HGB:unit:1"],
                    }
                ],
                "conditions": [],
            }
        )
        results = knowledge.search(query)
        self.assertEqual(len(results), 3)
        self.assertIn(
            "cosmos:sdtm:2026-05-26:HGBBLD", [candidate.ref for candidate in results]
        )
        candidate = knowledge.resolve("cosmos:sdtm:2026-05-26:HGBBLD")
        self.assertEqual(candidate.row("LBTESTCD").values()["assigned_value"], "HGB")
        self.assertEqual(candidate.row("LBORRES").values()["origin_type"], "Collected")


if __name__ == "__main__":
    unittest.main()
