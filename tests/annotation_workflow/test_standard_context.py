"""Public-corpus integration checks without reference/evaluator access.

Exercise source wording, neighbor association and permission boundaries against
the pinned corpus. A plausible wrong unit/timepoint or invented constant must
fail even if its output shape is otherwise valid.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.stage_b_mapper_module.annotation_workflow.standard_annotation import (
    StandardAnnotation,
    answer_dict,
    validate_plan,
)
from pipeline.stage_b_mapper_module.annotation_workflow.standard_context import (
    StandardsCorpus,
    wording,
)

ROOT = Path(__file__).resolve().parents[2]


def source_for(group, focused_name, *, occurrence=0, offset=0):
    """Create a blank page directly from public fields, with arbitrary local IDs."""
    rows = []
    for i, field in enumerate(group["fields"], 1):
        rows.append(
            {
                "field_id": f"F{i+offset:03d}",
                "question": wording(field),
                "response": {
                    "unit": field.get("display_unit") or None,
                    "ui_text": field.get("prompt", ""),
                },
            }
        )
    indices = [
        i
        for i, f in enumerate(group["fields"])
        if f.get("name", f.get("variable_name")) == focused_name
    ]
    index = indices[occurrence]
    return {
        **rows[index],
        "form": group["form"],
        "section": group["section"] or "",
        "neighbors": [r for i, r in enumerate(rows) if i != index],
    }


def generic(corpus, name):
    group = next(
        g for g in corpus.groups if any(f.get("name") == name for f in g["fields"])
    )
    return corpus.retrieve(source_for(group, name))


def proposal(fact=None, status="mapped"):
    return StandardAnnotation.model_validate(
        {
            "status": status,
            "facts": [fact] if fact else [],
            "missing": [],
            "not_submitted_when": [],
            "explanation": "Supported by delivered public mapping instructions.",
            "evidence_refs": ["E1"],
        }
    )


def fact(domain, variable, value, record=None):
    return {
        "domain": domain,
        "variable": variable,
        "value": value,
        "when": None,
        "record": record or {"kind": "collection_record"},
    }


class StandardContextTests(unittest.TestCase):
    """Check real public evidence and enforce semantic source permissions."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = StandardsCorpus(
            ROOT / "kb/standard_mapping_cdash23_sdtmig34_20261004"
        )

    def test_duplicate_units_bind_by_wording_and_order_not_case_identifier(self):
        corpus = self.corpus
        group = next(g for g in corpus.groups if g["form"] == "Six Minute Walk")
        source = source_for(group, "FTORRESU", occurrence=3)
        context = corpus.retrieve(source)
        rows = context["evidence"][0]["specializations"]
        assert (
            next(r["assigned_value"] for r in rows if r["sdtm_variable"] == "FTTESTCD")
            == "SIXMW104"
        )
        renamed = corpus.retrieve(
            source_for(group, "FTORRESU", occurrence=3, offset=80)
        )
        assert renamed["evidence"][0]["specializations"] == rows
        # Without the full ordered page, duplicate unit labels cannot establish
        # which test they qualify; retrieval must not fabricate a unique match.
        partial = copy.deepcopy(source)
        partial["neighbors"] = []
        assert corpus.retrieve(partial)["unmatched_source"]

    def test_generic_duration_has_both_collected_components(self):
        corpus = self.corpus
        context = generic(corpus, "HOCDUR")
        e = context["evidence"][0]
        assert ("HO", "HODUR") in {
            (d["domain"], d["variable"]) for d in e["declarations"]
        }
        assert "ISO 8601 Period" in e["collection_metadata"]["mappingInstructions"]
        assert any(c["collection_field"] == "HOCDURU" for c in e["companions"])

    def test_export_preserves_transform_and_record_without_filling_answers(self):
        p = proposal(
            fact(
                "HO",
                "HODUR",
                {"kind": "iso8601_duration", "field_ids": ["F003", "F004"]},
            )
        )
        exported = answer_dict(p)
        assert exported["facts"][0]["value"]["field_ids"] == ["F003", "F004"]
        assert exported["facts"][0]["record"] == {"kind": "collection_record"}
        assert "evidence_refs" not in exported

    def test_ns_requires_positive_evidence(self):
        corpus = self.corpus
        positive = generic(corpus, "AEACNOYN")
        assert validate_plan(proposal(status="not_submitted"), positive) == []
        direct = generic(corpus, "HODECOD")
        assert "not_submitted_lacks_positive_evidence" in validate_plan(
            proposal(status="not_submitted"), direct
        )

    def test_allowed_vocabulary_does_not_authorize_constant(self):
        corpus = self.corpus
        context = generic(corpus, "MLOCCUR")
        fid = context["source"]["field_id"]
        p = proposal(fact("ML", "MLOCCUR", {"kind": "constant", "value": "N"}))
        assert "constant_lacks_delivered_support" in validate_plan(p, context)
        p = proposal(fact("ML", "MLOCCUR", {"kind": "source", "field_id": fid}))
        # Source permission remains valid; full occurrence completion separately
        # requires its published answered/unanswered branch conditions.
        assert "bind_focused_source_only" not in validate_plan(p, context)

    def test_transform_does_not_drop_partner_or_include_unrelated_date(self):
        corpus = self.corpus
        context = generic(corpus, "HOSTDAT")
        fid = context["source"]["field_id"]
        e = context["evidence"][0]
        partner = next(
            c["source_field_id"]
            for c in e["companions"]
            if c["collection_field"] == "HOSTTIM"
        )
        unrelated = next(
            c["source_field_id"]
            for c in e["companions"]
            if c["collection_field"] == "HOENDAT"
        )
        for bad in ([fid], [fid, unrelated]):
            p = proposal(
                fact("HO", "HOSTDTC", {"kind": "iso8601_datetime", "field_ids": bad})
            )
            assert "transform_components_do_not_match_public_target" in validate_plan(
                p, context
            )
        p = proposal(
            fact(
                "HO",
                "HOSTDTC",
                {"kind": "iso8601_datetime", "field_ids": sorted([fid, partner])},
            )
        )
        assert validate_plan(p, context) == []

    def test_source_scope_rejects_neighbor_as_mapped_subject(self):
        corpus = self.corpus
        context = generic(corpus, "HODECOD")
        neighbor = context["source"]["neighbors"][0]["field_id"]
        p = proposal(fact("HO", "HODECOD", {"kind": "source", "field_id": neighbor}))
        assert "bind_focused_source_only" in validate_plan(p, context)

    def test_pin_verification_rejects_modified_public_file(self):
        corpus = self.corpus
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        tmp_path = Path(folder.name)
        manifest = copy.deepcopy(corpus.manifest)
        manifest["files"] = {"cdashig-2-3.json": manifest["files"]["cdashig-2-3.json"]}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "cdashig-2-3.json").write_text("{}")
        with self.assertRaisesRegex(Exception, "standard_context_source_changed"):
            StandardsCorpus(tmp_path)


if __name__ == "__main__":
    unittest.main()
