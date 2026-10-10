"""Check public-evidence obligations against plausible incomplete annotations.

Synthetic bindings come from the actual public corpus, never benchmark answers.
The checks distinguish correct target names from complete record/value/condition
semantics and preserve missing supplemental conventions instead of guessing.
"""

import unittest

from pipeline.stage_b_mapper_module.annotation_workflow.standard_annotation import (
    StandardAnnotation,
    validate_plan,
)
from tests.annotation_workflow.test_standard_context import (
    ROOT,
    StandardsCorpus,
    fact,
    generic,
    proposal,
    source_for,
)


class RequirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = StandardsCorpus(
            ROOT / "kb/standard_mapping_cdash23_sdtmig34_20261004"
        )

    def test_direct_qualifier_requires_collected_test_identity(self):
        ctx = generic(self.corpus, "RSEVALID")
        fid = ctx["source"]["field_id"]
        p = proposal(fact("RS", "RSEVALID", {"kind": "source", "field_id": fid}))
        partner = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "RSTEST"
        )
        assert "required_collected_test_binding:" + partner in validate_plan(p, ctx)
        p = proposal(
            fact(
                "RS",
                "RSEVALID",
                {"kind": "source", "field_id": fid},
                {"kind": "collection_record", "test_source_field": partner},
            )
        )
        assert validate_plan(p, ctx) == []

    def test_unit_obligation_uses_printed_value_and_original_field_scope(self):
        group = next(g for g in self.corpus.groups if g["form"] == "Six Minute Walk")
        src = source_for(group, "FTORRESU")
        src["response"]["unit"] = "m"
        ctx = self.corpus.retrieve(src)
        req = next(
            r for r in ctx["requirements"] if r["kind"] == "printed_original_unit"
        )
        assert req["literal"] == "m" and req["variable"] == "FTORRESU"
        assert (
            next(r for r in ctx["requirements"] if r["kind"] == "original_input_scope")[
                "primary_variable"
            ]
            == "FTORRESU"
        )
        assert all(r["basis"]["evidence_id"] == "E1" for r in ctx["requirements"])

    def test_other_result_retains_parent_condition_and_fixed_branch(self):
        ctx = generic(self.corpus, "RERESOTH")
        fid = ctx["source"]["field_id"]
        r = next(r for r in ctx["requirements"] if r["kind"] == "other_result_branch")
        p = proposal(fact("RE", "REORRES", {"kind": "source", "field_id": fid}))
        errors = validate_plan(p, ctx)
        assert (
            "other_result_parent_condition_required" in errors
            and "other_result_fixed_branch_required" in errors
        )
        binder = next(
            r["source_field"]
            for r in ctx["requirements"]
            if r["kind"] == "collected_test_binding"
        )
        record = {"kind": "collection_record", "test_source_field": binder}
        when = {
            "field_id": r["parent_source_field"],
            "operator": "eq",
            "value": "OTHER",
        }
        a = fact("RE", "REORRES", {"kind": "source", "field_id": fid}, record)
        a["when"] = when
        b = fact("RE", "RESTRESC", {"kind": "constant", "value": "OTHER"}, record)
        b["when"] = when
        p = StandardAnnotation.model_validate(
            {
                "status": "mapped",
                "facts": [a, b],
                "missing": [],
                "not_submitted_when": [],
                "explanation": "Original public OTHER mapping instructions support both branches.",
                "evidence_refs": ["E1"],
            }
        )
        assert validate_plan(p, ctx) == []

    def test_supplemental_suggestion_does_not_supply_a_study_design(self):
        ctx = generic(self.corpus, "EXCINTDU")
        assert (
            "could be submitted"
            in ctx["evidence"][0]["collection_metadata"]["mappingInstructions"]
        )
        p = proposal(
            fact(
                "SUPPQUAL",
                "QVAL",
                {"kind": "source", "field_id": ctx["source"]["field_id"]},
            )
        )
        assert "supplemental_design_requires_extension_convention" in validate_plan(
            p, ctx
        )
        p = StandardAnnotation.model_validate(
            {
                "status": "needs_convention",
                "facts": [],
                "missing": ["extension_convention"],
                "not_submitted_when": [],
                "explanation": "Supplemental record design remains open.",
                "evidence_refs": ["E1"],
            }
        )
        assert validate_plan(p, ctx) == []

    def test_occurrence_requires_answered_condition_on_collected_value(self):
        ctx = generic(self.corpus, "MLOCCUR")
        fid = ctx["source"]["field_id"]
        p = proposal(fact("ML", "MLOCCUR", {"kind": "source", "field_id": fid}))
        assert "collected_occurrence_requires_answered_condition" in validate_plan(
            p, ctx
        )

    def test_panel_performance_preserves_known_branch_and_missing_scope(self):
        ctx = generic(self.corpus, "MBPERF")
        fid = ctx["source"]["field_id"]
        p = proposal(fact("MB", "MBSTAT", {"kind": "source", "field_id": fid}))
        assert "required_missing_convention:test_identity" in validate_plan(p, ctx)
        branch = fact("MB", "MBSTAT", {"kind": "constant", "value": "NOT DONE"})
        branch["when"] = {"field_id": fid, "operator": "eq", "value": "N"}
        p = StandardAnnotation.model_validate(
            {
                **proposal(branch).model_dump(),
                "status": "partial",
                "missing": ["test_identity"],
            }
        )
        assert validate_plan(p, ctx) == []
        assert not any(
            r["kind"] == "collected_test_binding" for r in ctx["requirements"]
        )

    def test_public_timepoint_anchor_and_scale_require_partial(self):
        for name, domain, missing in [
            ("CPTPT", "CP", "timing_anchor"),
            ("CETOX", "CE", "coding_dictionary"),
        ]:
            ctx = generic(self.corpus, name)
            p = proposal(
                fact(
                    domain,
                    name,
                    {"kind": "source", "field_id": ctx["source"]["field_id"]},
                )
            )
            assert "required_missing_convention:" + missing in validate_plan(p, ctx)
            p = StandardAnnotation.model_validate(
                {**p.model_dump(), "status": "partial", "missing": [missing]}
            )
            assert validate_plan(p, ctx) == []

    def test_collected_test_requires_lookup_without_self_record_binding(self):
        ctx = generic(self.corpus, "OETEST")
        fid = ctx["source"]["field_id"]
        p = proposal(fact("OE", "OETEST", {"kind": "source", "field_id": fid}))
        assert "required_public_value:OETESTCD:ct_lookup" in validate_plan(p, ctx)
        p = StandardAnnotation.model_validate(
            {
                **p.model_dump(),
                "facts": [
                    *p.model_dump()["facts"],
                    fact("OE", "OETESTCD", {"kind": "ct_lookup", "field_id": fid}),
                ],
            }
        )
        assert validate_plan(p, ctx) == []

    def test_explicit_date_conversion_is_required_even_for_one_component(self):
        ctx = generic(self.corpus, "DTHDAT")
        p = proposal(
            fact(
                "DM",
                "DTHDTC",
                {"kind": "source", "field_id": ctx["source"]["field_id"]},
            )
        )
        assert "required_public_value:DTHDTC:iso8601_datetime" in validate_plan(p, ctx)

    def test_placeholder_test_and_optional_linkage_do_not_become_complete(self):
        for name, missing in [
            ("[SCTESTCD]_SCORRES", "test_identity"),
            ("EGAENO", "relationship_convention"),
            ("PRDOSADJ", "extension_convention"),
        ]:
            ctx = generic(self.corpus, name)
            p = proposal(status="abstain")
            assert "required_missing_convention:" + missing in validate_plan(p, ctx)
            p = StandardAnnotation.model_validate(
                {**p.model_dump(), "status": "needs_convention", "missing": [missing]}
            )
            assert validate_plan(p, ctx) == []

    def test_public_interpretation_assignment_authorizes_fixed_test_record(self):
        ctx = generic(self.corpus, "INTP_EGORRES")
        fid = ctx["source"]["field_id"]
        record = {"kind": "test", "test_code": "INTP"}
        p = StandardAnnotation.model_validate(
            {
                **proposal(
                    fact("EG", "EGORRES", {"kind": "source", "field_id": fid})
                ).model_dump(),
                "facts": [
                    fact("EG", "EGORRES", {"kind": "source", "field_id": fid}, record),
                    fact(
                        "EG", "EGTESTCD", {"kind": "constant", "value": "INTP"}, record
                    ),
                    fact(
                        "EG",
                        "EGTEST",
                        {"kind": "constant", "value": "Interpretation"},
                        record,
                    ),
                ],
            }
        )
        assert validate_plan(p, ctx) == []
        bad = p.model_dump()
        bad["facts"][0]["record"]["test_code"] = "UNKNOWN"
        assert "test_record_not_delivered" in validate_plan(
            StandardAnnotation.model_validate(bad), ctx
        )

    def test_description_requires_activating_branch_and_preserves_uncoded_option(self):
        ctx = generic(self.corpus, "PEDESC")
        fid = ctx["source"]["field_id"]
        partner = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "PETEST"
        )
        parent = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "PERES"
        )
        record = {"kind": "collection_record", "test_source_field": partner}
        a = fact("PE", "PEORRES", {"kind": "source", "field_id": fid}, record)
        assert "required_public_value:PEORRES:source" in validate_plan(proposal(a), ctx)
        a["when"] = {"field_id": parent, "operator": "eq", "value": "ABNORMAL"}
        b = {**a, "variable": "PESTRESC"}
        p = StandardAnnotation.model_validate(
            {**proposal(a).model_dump(), "facts": [a, b]}
        )
        assert validate_plan(p, ctx) == []

    def test_one_target_branch_cannot_have_two_different_value_bindings(self):
        # A field annotation represents one mapping per target/record/condition.
        # Two different values on the same branch are conflicting proposals,
        # even when each value is separately supported by public prose.
        ctx = generic(self.corpus, "PERES")
        fid = ctx["source"]["field_id"]
        a = fact("PE", "PEORRES", {"kind": "source", "field_id": fid})
        a["when"] = {"field_id": fid, "operator": "eq", "value": "NORMAL"}
        b = {**a, "value": {"kind": "constant", "value": "NORMAL"}}
        p = StandardAnnotation.model_validate(
            {**proposal(a).model_dump(), "facts": [a, b]}
        )
        assert "conflicting_target_value_bindings" in validate_plan(p, ctx)

    def test_uncoded_result_and_unresolved_dictionary_are_alternative_decisions(self):
        ctx = generic(self.corpus, "PEDESC")
        fid = ctx["source"]["field_id"]
        partner = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "PETEST"
        )
        parent = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "PERES"
        )
        a = fact(
            "PE",
            "PEORRES",
            {"kind": "source", "field_id": fid},
            {"kind": "collection_record", "test_source_field": partner},
        )
        a["when"] = {"field_id": parent, "operator": "eq", "value": "ABNORMAL"}
        p = StandardAnnotation.model_validate(
            {
                **proposal(a).model_dump(),
                "status": "partial",
                "missing": ["coding_dictionary"],
                "facts": [a, {**a, "variable": "PESTRESC"}],
            }
        )
        assert (
            "uncoded_result_and_missing_dictionary_are_alternatives"
            in validate_plan(p, ctx)
        )

    def test_nearby_test_name_does_not_resolve_a_panel_checkbox(self):
        # An explicit TEST elsewhere is a collection binding for individual
        # results, but cannot resolve the performance checkbox's panel scope.
        group = next(
            g
            for g in self.corpus.groups
            if {"REPERF", "RETEST"} <= {f.get("name") for f in g["fields"]}
        )
        ctx = self.corpus.retrieve(source_for(group, "REPERF"))
        fid = ctx["source"]["field_id"]
        partner = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "RETEST"
        )
        a = fact(
            "RE",
            "RESTAT",
            {"kind": "constant", "value": "NOT DONE"},
            {"kind": "collection_record", "test_source_field": partner},
        )
        a["when"] = {"field_id": fid, "operator": "eq", "value": "N"}
        p = StandardAnnotation.model_validate(
            {
                **proposal(a).model_dump(),
                "status": "partial",
                "missing": ["test_identity"],
            }
        )
        assert "unresolved_panel_cannot_bind_a_specific_test" in validate_plan(p, ctx)

    def test_panel_cannot_add_an_unconditional_raw_status_copy(self):
        ctx = generic(self.corpus, "MBPERF")
        fid = ctx["source"]["field_id"]
        a = fact("MB", "MBSTAT", {"kind": "constant", "value": "NOT DONE"})
        a["when"] = {"field_id": fid, "operator": "eq", "value": "N"}
        p = StandardAnnotation.model_validate(
            {
                **proposal(a).model_dump(),
                "status": "partial",
                "missing": ["test_identity"],
                "facts": [a, fact("MB", "MBSTAT", {"kind": "source", "field_id": fid})],
            }
        )
        assert "unsupported_additional_mapping_on_unresolved_panel" in validate_plan(
            p, ctx
        )

    def test_unresolved_submission_design_cannot_invent_a_response_condition(self):
        ctx = generic(self.corpus, "PRDOSADJ")
        p = StandardAnnotation.model_validate(
            {
                **proposal(status="abstain").model_dump(),
                "status": "needs_convention",
                "missing": ["extension_convention"],
                "not_submitted_when": [
                    {
                        "field_id": ctx["source"]["field_id"],
                        "operator": "eq",
                        "value": "Y",
                    }
                ],
            }
        )
        assert (
            "unresolved_design_cannot_invent_not_submitted_conditions"
            in validate_plan(p, ctx)
        )

    def test_collected_test_record_feedback_explains_inactive_slots(self):
        ctx = generic(self.corpus, "OETEST")
        fid = ctx["source"]["field_id"]
        a = fact(
            "OE",
            "OETEST",
            {"kind": "source", "field_id": fid},
            {"kind": "collection_record", "test_source_field": fid},
        )
        p = StandardAnnotation.model_validate(
            {
                **proposal(a).model_dump(),
                "facts": [
                    a,
                    {
                        **a,
                        "variable": "OETESTCD",
                        "value": {"kind": "ct_lookup", "field_id": fid},
                    },
                ],
            }
        )
        assert (
            "collected_test_name_requires_collection_record_with_null_test_source_field_and_test_code"
            in validate_plan(p, ctx)
        )

    def test_specific_dataset_mapping_precedes_same_named_model_variable(self):
        # The public OE field links to OESTRESC even though SDTM also defines a
        # same-named OERESCAT. A model vocabulary entry is not a field mapping.
        ctx = generic(self.corpus, "OERESCAT")
        fid = ctx["source"]["field_id"]
        partner = next(
            c["source_field_id"]
            for c in ctx["evidence"][0]["companions"]
            if c["collection_field"] == "OETEST"
        )
        a = fact(
            "OE",
            "OERESCAT",
            {"kind": "source", "field_id": fid},
            {"kind": "collection_record", "test_source_field": partner},
        )
        assert "required_public_value:OESTRESC:source" in validate_plan(
            proposal(a), ctx
        )
        a["variable"] = "OESTRESC"
        assert validate_plan(proposal(a), ctx) == []

    def test_result_branch_cannot_submit_status_for_a_performed_normal_result(self):
        # The public notes reserve NOT DONE for the unexamined branch; source
        # NORMAL/ABNORMAL must not become STAT and a null ORRES has no STRESC.
        ctx = generic(self.corpus, "PERES")
        fid = ctx["source"]["field_id"]
        a = fact("PE", "PESTAT", {"kind": "source", "field_id": fid})
        a["when"] = {"field_id": fid, "operator": "eq", "value": "NORMAL"}
        assert "unsupported_published_result_branch:PESTAT" in validate_plan(
            proposal(a), ctx
        )
        a["variable"] = "PESTRESC"
        a["when"]["value"] = "NOT DONE"
        assert "unsupported_published_result_branch:PESTRESC" in validate_plan(
            proposal(a), ctx
        )


if __name__ == "__main__":
    unittest.main()
