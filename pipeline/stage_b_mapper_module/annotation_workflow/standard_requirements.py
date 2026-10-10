"""Expose actionable field obligations from delivered public standards evidence.

Rules inspect source bindings and original mapping instructions, never evaluator
references/case IDs. Keep their evidence basis visible and reject unsupported
completion; do not silently fill output facts or study supplemental conventions.
"""

from __future__ import annotations

import json
import re
from typing import Any


def derive(source: dict, evidence: list[dict]) -> list[dict[str, Any]]:
    """Summarize public record/value/condition requirements without private gold."""
    result = []
    for e in evidence:
        meta = e["collection_metadata"]
        domain = e["domain"]
        basis = {
            "evidence_id": e["evidence_id"],
            "collection_field": e["collection_field"],
        }
        text = meta.get("mappingInstructions") or ""
        notes = meta.get("implementationNotes") or ""
        notes += " " + " ".join(
            c.get("related_mapping_notes", "") for c in e["companions"]
        )
        combined = text + " " + notes
        field = e["collection_field"]
        fid = source["field_id"]
        targets = [t for t in e["declarations"] if t["domain"] != "SUPPQUAL"]
        # A public panel example leaves record scope unresolved even if a TEST
        # happens to occur elsewhere on the page. Preserve the known N branch;
        # neither nearby fields nor example ALL codes instantiate study scope.
        performance = bool(
            field.endswith("PERF")
            and "STAT" in text
            and re.search(r"multiple tests|entire panel|specific test", combined, re.I)
        )
        missing = None
        empty_scope = False
        if performance:
            missing = "test_identity"
        elif (
            re.match(r"\[\w+\]_\w+ORRES$", field)
            and not e["specializations"]
            and any(t["variable"].endswith("TESTCD") for t in targets)
        ):
            missing, empty_scope = "test_identity", True
        elif "RELREC" in text and not targets:
            missing, empty_scope = "relationship_convention", True
        elif re.search(r"SUPP(?:Q|QUAL)\b", text, re.I):
            missing, empty_scope = "extension_convention", True
        elif re.search(r"time.point anchors", text, re.I):
            missing = "timing_anchor"
        elif re.search(
            r"scale name and version|dictionary name and version", text, re.I
        ):
            missing = "coding_dictionary"
        elif re.search(r"multiple actions taken is up to the sponsor", notes, re.I):
            missing = "extension_convention"
        elif re.search(r"multiple races.*sponsor.defined", notes, re.I) and re.search(
            r"more than one|multiple", source["question"], re.I
        ):
            missing = "extension_convention"
        if missing:
            result.append(
                {
                    "kind": "decision_scope",
                    "domain": domain,
                    "status": "needs_convention" if empty_scope else "partial",
                    "missing": [missing],
                    "panel_scope_unresolved": performance,
                    "basis": basis,
                    # Name the native decision action explicitly: a missing
                    # convention does not erase the public facts of a partial
                    # annotation. Symbolic error codes alone caused oscillation.
                    "reason": (
                        "Set decision=needs_convention, facts=[], and missing="
                        if empty_scope
                        else "Set decision=partial, retain every supported public_value fact, and missing="
                    )
                    + json.dumps([missing])
                    + "; not_submitted_when=[]. This convention remains unresolved in the delivered public instructions.",
                }
            )

        def require_value(variable, value, when=None, target_domain=None):
            # These are visible obligations, not output generation: the model
            # must still represent each value/condition in its own proposal.
            result.append(
                {
                    "kind": "public_value",
                    "domain": target_domain or domain,
                    "variable": variable,
                    "value": value,
                    "when": when,
                    "basis": {**basis, "metadata_key": "mappingInstructions"},
                }
            )

        if performance:
            status = next(
                (t["variable"] for t in targets if t["variable"].endswith("STAT")), None
            )
            if status and '"NOT DONE"' in text and '"N"' in text:
                require_value(
                    status,
                    {"kind": "constant", "value": "NOT DONE"},
                    {"field_id": fid, "operator": "eq", "value": "N"},
                )
        elif not empty_scope and text.startswith("Maps directly") and len(targets) == 1:
            # Preprinting/coding advice does not remove a delivered blank source
            # field. Only an unconditional positive direct mapping is enforced.
            if not re.search(r"If |OTHER|not asked|not answered", text):
                t = targets[0]
                require_value(
                    t["variable"],
                    {"kind": "source", "field_id": fid},
                    target_domain=t["domain"],
                )
        if field == domain + "TEST" and re.search(r"TESTCD may be determined", text):
            if not any(r.get("variable") == field for r in result):
                require_value(field, {"kind": "source", "field_id": fid})
            require_value(domain + "TESTCD", {"kind": "ct_lookup", "field_id": fid})
            # The focused TEST name defines the lookup input. It cannot bind
            # itself as a neighboring result's record or supply a fixed code.
            result.append(
                {
                    "kind": "collected_test_name_record",
                    "domain": domain,
                    "record": {"kind": "collection_record"},
                    "basis": basis,
                    "wire_record": {
                        "kind": "collection_record",
                        "name": None,
                        "test_code": None,
                        "test_source_field": None,
                    },
                }
            )
        if (
            "concatenate all collected CDASH DATE and TIME" in text
            and "ISO 8601" in text
        ):
            for t in targets:
                components = [fid] + [
                    c["source_field_id"]
                    for c in e["companions"]
                    if any(
                        (v["domain"], v["variable"]) == (t["domain"], t["variable"])
                        for v in c["targets"]
                    )
                ]
                require_value(
                    t["variable"],
                    {"kind": "iso8601_datetime", "field_ids": components},
                    target_domain=t["domain"],
                )
        numeric = re.search(
            r"Numeric values map to (\w+).*Non-numeric values.*map to (\w+)", text
        )
        if numeric:
            for variable, operator in zip(numeric.groups(), ("numeric", "non_numeric")):
                require_value(
                    variable,
                    {"kind": "source", "field_id": fid},
                    {"field_id": fid, "operator": operator, "value": None},
                )
        # Positive TEST/TESTCD assignments in public prose are as authoritative
        # as specialization rows. Restrict this parser to quoted '=' assignments
        # for the delivered domain; example vocabularies alone are insufficient.
        fixed = dict(
            re.findall(
                r"\b(" + re.escape(domain) + r'TEST(?:CD)?)\s*=\s*"([^"]+)"', text
            )
        )
        if domain + "TESTCD" in fixed and domain + "TEST" in fixed:
            for variable, literal in fixed.items():
                result.append(
                    {
                        "kind": "fixed_descriptor",
                        "domain": domain,
                        "variable": variable,
                        "literal": literal,
                        "basis": basis,
                    }
                )
            for t in targets:
                require_value(t["variable"], {"kind": "source", "field_id": fid})
            result.append(
                {
                    "kind": "fixed_test_record",
                    "domain": domain,
                    "test_code": fixed[domain + "TESTCD"],
                    "basis": basis,
                }
            )

        # Follow explicit normal/abnormal source branches. A neighbor value is
        # permitted only for this published target and activating condition, so
        # the focused-source boundary remains closed to unrelated borrowing.
        branches = re.findall(
            r'If (\w+)="(Normal|Abnormal)", populate (\w+) with the value of (\w+)',
            text,
        )
        ids_by_name = {
            c["collection_field"]: c["source_field_id"] for c in e["companions"]
        }
        ids_by_name[field] = fid
        branch_start = len(result)
        for parent, literal, variable, value_field in branches:
            if parent not in ids_by_name or value_field not in ids_by_name:
                continue
            if field not in {parent, value_field}:
                continue
            when = {
                "field_id": ids_by_name[parent],
                "operator": "eq",
                "value": literal.upper(),
            }
            value = {"kind": "source", "field_id": ids_by_name[value_field]}
            if literal == "Normal" and re.search(
                r"value in " + variable + r' should be "NORMAL"', notes
            ):
                value = {"kind": "constant", "value": "NORMAL"}
            require_value(variable, value, when)
            if "if not coded" in notes:
                if literal == "Normal":
                    require_value(domain + "STRESC", value, when)
                else:
                    result.append(
                        {
                            "kind": "result_coding_choice",
                            "domain": domain,
                            "variable": domain + "STRESC",
                            "value": value,
                            "when": when,
                            "basis": basis,
                            "reason": "Represent the published uncoded original-text branch, or leave coding_dictionary unresolved as partial.",
                        }
                    )
            if (
                field == parent
                and literal == "Normal"
                and '"NOT DONE"' in notes
                and domain + "STAT" in notes
            ):
                require_value(
                    domain + "STAT",
                    {"kind": "constant", "value": "NOT DONE"},
                    {"field_id": fid, "operator": "eq", "value": "NOT DONE"},
                )
        # These explicit result branches form a closed field mapping: a null
        # original result cannot manufacture STRESC, and NORMAL/ABNORMAL are not
        # NOT DONE status values. Permit the published uncoded alternative too,
        # while preserving the model's choice to defer dictionary coding.
        permitted_branches = [
            {k: r[k] for k in ("variable", "value", "when")}
            for r in result[branch_start:]
            if r["kind"] in {"public_value", "result_coding_choice"}
        ]
        if branches and permitted_branches:
            result.append(
                {
                    "kind": "published_result_branch_scope",
                    "domain": domain,
                    "permitted": permitted_branches,
                    "basis": basis,
                    "reason": "Only these published result value/condition branches belong to this focused field. Do not copy NORMAL/ABNORMAL into STAT or map STRESC when the original result is null.",
                }
            )
        partners = [
            c for c in e["companions"] if c["collection_field"] == domain + "TEST"
        ]
        if len(partners) == 1 and not performance and not empty_scope:
            result.append(
                {
                    "kind": "collected_test_binding",
                    "domain": domain,
                    "source_field": partners[0]["source_field_id"],
                    "basis": basis,
                }
            )
        assigned = [
            r
            for r in e["specializations"]
            if r["assigned_value"]
            and r["sdtm_variable"]
            in {domain + "TESTCD", domain + "TEST", domain + "CAT"}
        ]
        for r in assigned:
            result.append(
                {
                    "kind": "fixed_descriptor",
                    "domain": domain,
                    "variable": r["sdtm_variable"],
                    "literal": r["assigned_value"],
                    "basis": {
                        **basis,
                        "specialization_variable": r["sdtm_variable"],
                        "assigned_term": r["assigned_term"],
                    },
                }
            )
        if e["kind"] == "COSMoS_draft_collection_template":
            # Scope is the original collection field plus its fixed descriptors;
            # normalized dataset values belong to a later derivation task.
            if e["collection_field"] in {domain + "ORRES", domain + "ORRESU"}:
                result.append(
                    {
                        "kind": "original_input_scope",
                        "domain": domain,
                        "primary_variable": e["collection_field"],
                        "excluded_derived_variables": [
                            domain + "STRESC",
                            domain + "STRESN",
                            domain + "STRESU",
                        ],
                        "basis": basis,
                    }
                )
            unit = source["response"].get("unit")
            fixed_units = {
                r["assigned_value"]
                for r in e["specializations"]
                if r["sdtm_variable"] == domain + "ORRESU" and r["assigned_value"]
            }
            if e["collection_field"] == domain + "ORRESU" and unit in fixed_units:
                result.append(
                    {
                        "kind": "printed_original_unit",
                        "domain": domain,
                        "variable": domain + "ORRESU",
                        "literal": unit,
                        "basis": basis,
                    }
                )
            if (
                meta.get("data_type") in {"date", "datetime"}
                and meta.get("sdtm_target_variable") == domain + "DTC"
                and not e["specializations"]
            ):
                result.append(
                    {
                        "kind": "assessment_datetime",
                        "domain": domain,
                        "variable": domain + "DTC",
                        "instrument": source["form"],
                        "published_group_alias": meta["vlm_group_id"],
                        "source_fields": [source["field_id"]],
                        "basis": basis,
                    }
                )
        if any(t["domain"] == "SUPPQUAL" for t in e["declarations"]) or re.search(
            r"\bQVAL\s+when\s+QNAM\s*=", meta.get("sdtm_annotation", ""), re.I
        ):
            result.append(
                {
                    "kind": "unresolved_supplemental_design",
                    "missing": "extension_convention",
                    "basis": basis,
                    "reason": "A suggested QNAM or generic SUPPQUAL target does not supply the study's supplemental record design and linkage.",
                }
            )
        other = re.search(
            r'"(OTHER)" value collected in the CDASH field (\w+) is mapped to the SDTM variable (\w+)',
            text,
        )
        if other:
            literal, parent, var = other.groups()
            match = [c for c in e["companions"] if c["collection_field"] == parent]
            if len(match) == 1:
                result.append(
                    {
                        "kind": "other_result_branch",
                        "domain": domain,
                        "parent_source_field": match[0]["source_field_id"],
                        "literal": literal,
                        "fixed_variable": var,
                        "source_variables": [
                            t["variable"]
                            for t in e["declarations"]
                            if t["domain"] == domain
                        ],
                        "basis": {**basis, "metadata_key": "mappingInstructions"},
                    }
                )
        status = re.search(
            r'If the (?:question|response) was not asked or answered, populate the SDTMIG variable (\w+) with "(NOT DONE)"',
            text,
        )
        if status:
            variable, literal = status.groups()
            result.append(
                {
                    "kind": "answered_unanswered_branches",
                    "domain": domain,
                    "source_field": source["field_id"],
                    "source_variables": [
                        t["variable"]
                        for t in e["declarations"]
                        if t["domain"] == domain
                    ],
                    "status_variable": variable,
                    "literal": literal,
                    "basis": {**basis, "metadata_key": "mappingInstructions"},
                }
            )
    return result


def violations(plan: dict, requirements: list[dict]) -> list[str]:
    """Check represented semantics against public obligations; never repair facts."""
    errors = []
    facts = plan["facts"]
    for r in requirements:
        kind = r["kind"]
        domain = r.get("domain")
        selected = [f for f in facts if f["domain"] == domain]
        if kind == "decision_scope":
            if plan["missing"] != r["missing"] or plan["status"] != r["status"]:
                errors.extend("required_missing_convention:" + m for m in r["missing"])
                errors.append(r["reason"])
            if r["status"] == "needs_convention" and facts:
                errors.append("uninstantiated_design_has_no_supported_facts")
            if plan["not_submitted_when"]:
                # A study design choice is not a patient-answer predicate.
                # No response condition is supplied by these unresolved rules.
                errors.append(
                    "unresolved_design_cannot_invent_not_submitted_conditions"
                )
            if r.get("panel_scope_unresolved"):
                allowed = [
                    v
                    for v in requirements
                    if v["kind"] == "public_value" and v["domain"] == domain
                ]
                for f in selected:
                    if f["record"] != {"kind": "collection_record"}:
                        errors.append("unresolved_panel_cannot_bind_a_specific_test")
                    if not any(
                        all(f[k] == v[k] for k in ("variable", "value", "when"))
                        for v in allowed
                    ):
                        errors.append(
                            "unsupported_additional_mapping_on_unresolved_panel"
                        )
        elif kind == "public_value":
            # Component order is presentation only; identity and completeness
            # matter. All other value/condition tags must match exactly.
            def matches(f):
                value = f["value"]
                expected = r["value"]
                equal = value == expected
                if (
                    expected["kind"] == "iso8601_datetime"
                    and value["kind"] == expected["kind"]
                ):
                    equal = set(value["field_ids"]) == set(expected["field_ids"])
                return (
                    f["variable"] == r["variable"] and equal and f["when"] == r["when"]
                )

            if not any(matches(f) for f in selected):
                errors.append(
                    "required_public_value:" + r["variable"] + ":" + r["value"]["kind"]
                )
        elif kind == "fixed_test_record":
            if any(
                f["record"] != {"kind": "test", "test_code": r["test_code"]}
                for f in selected
            ):
                errors.append("required_public_fixed_test_record:" + r["test_code"])
        elif kind == "collected_test_name_record":
            if any(f["record"] != r["record"] for f in selected):
                errors.append(
                    "collected_test_name_requires_collection_record_with_null_test_source_field_and_test_code"
                )
        elif kind == "published_result_branch_scope":
            for f in selected:
                if not any(
                    all(f[k] == v[k] for k in ("variable", "value", "when"))
                    for v in r["permitted"]
                ):
                    errors.append(
                        "unsupported_published_result_branch:" + f["variable"]
                    )
                    errors.append(
                        "Remove unsupported extra branches; use only the value/condition mappings listed in published_result_branch_scope.permitted."
                    )
        elif kind == "result_coding_choice":
            represented = any(
                f["variable"] == r["variable"]
                and f["value"] == r["value"]
                and f["when"] == r["when"]
                for f in selected
            )
            deferred = plan["status"] == "partial" and plan["missing"] == [
                "coding_dictionary"
            ]
            if not represented and not deferred:
                errors.append("represent_uncoded_result_or_partial_coding_dictionary")
            if represented and deferred:
                errors.append("uncoded_result_and_missing_dictionary_are_alternatives")
                # Correct the decision and retained branch as a single action;
                # do not insert or remove any facts on the model's behalf.
                errors.append(
                    "Choose exactly one public alternative: keep "
                    + domain
                    + "."
                    + r["variable"]
                    + " for the uncoded abnormal branch and set decision=mapped, missing=[]; "
                    + "or remove only that abnormal standardized-result fact and set decision=partial, "
                    + 'missing=["coding_dictionary"]. Preserve all other required public facts.'
                )
        elif kind == "collected_test_binding":
            for f in selected:
                if f["record"] != {
                    "kind": "collection_record",
                    "test_source_field": r["source_field"],
                }:
                    errors.append(
                        "required_collected_test_binding:" + r["source_field"]
                    )
        elif kind == "original_input_scope":
            if any(f["variable"] in r["excluded_derived_variables"] for f in selected):
                errors.append("derived_dataset_variable_outside_original_input_scope")
        elif kind in {"fixed_descriptor", "printed_original_unit"}:
            if selected and not any(
                f["variable"] == r["variable"]
                and f["value"] == {"kind": "constant", "value": r["literal"]}
                for f in selected
            ):
                errors.append("required_public_fixed_assignment:" + r["variable"])
        elif kind == "assessment_datetime":
            if selected and not any(
                f["variable"] == r["variable"]
                and f["record"] == {"kind": "instrument", "instrument": r["instrument"]}
                and f["value"]
                == {"kind": "iso8601_datetime", "field_ids": r["source_fields"]}
                for f in selected
            ):
                errors.append("required_assessment_datetime_binding")
        elif kind == "unresolved_supplemental_design":
            if (
                plan["status"] != "needs_convention"
                or facts
                or plan["missing"] != [r["missing"]]
            ):
                errors.append("supplemental_design_requires_extension_convention")
        elif kind == "other_result_branch":
            when = {
                "field_id": r["parent_source_field"],
                "operator": "eq",
                "value": r["literal"],
            }
            if any(
                f["variable"] in r["source_variables"] and f["when"] != when
                for f in selected
            ):
                errors.append("other_result_parent_condition_required")
            if selected and not any(
                f["variable"] == r["fixed_variable"]
                and f["value"] == {"kind": "constant", "value": r["literal"]}
                and f["when"] == when
                for f in selected
            ):
                errors.append("other_result_fixed_branch_required")
        elif kind == "answered_unanswered_branches":
            answered = {
                "field_id": r["source_field"],
                "operator": "answered",
                "value": None,
            }
            unanswered = {**answered, "operator": "unanswered"}
            if any(
                f["variable"] in r["source_variables"] and f["when"] != answered
                for f in selected
            ):
                errors.append("collected_occurrence_requires_answered_condition")
            if selected and not any(
                f["variable"] == r["status_variable"]
                and f["value"] == {"kind": "constant", "value": r["literal"]}
                and f["when"] == unanswered
                for f in selected
            ):
                errors.append("unanswered_status_branch_required")
    return sorted(set(errors))
