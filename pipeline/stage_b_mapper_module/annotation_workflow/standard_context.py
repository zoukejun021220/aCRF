"""Retrieve pinned public collection/target evidence without evaluator access.

Match blank source wording and its ordered neighbors to published collection
templates. Local IDs are bindings, never lookup keys for private answers. Keep
original mapping instructions, model inheritance and specialization/CT evidence;
source-specific ambiguities remain visible instead of becoming invented targets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import WorkflowError, canonical, digest
from .standard_requirements import derive

PROFILE = {
    "cdashig": "2.3",
    "cdash_model": "1.3",
    "sdtmig": "3.4",
    "sdtm": "2.0",
    "terminology": "sdtmct-2025-03-28",
}


def normalized(text: str | None) -> str:
    """Normalize typography only; leave all clinical words and numbers distinct."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def wording(field: dict[str, Any]) -> str:
    """Read published collection labels, not target-variable or annotation strings."""
    return (
        field.get("questionText")
        or field.get("question_text")
        or field.get("prompt")
        or field.get("label", "")
    )


def inherited(classes: dict[str, Any], owner: str, target: str) -> bool:
    """Require forward/reciprocal versioned model inheritance before instantiation."""
    seen = set()
    while owner != target:
        if owner in seen or owner not in classes:
            return False
        seen.add(owner)
        parent = classes[owner]["_links"].get("parentClass", {}).get("href")
        if parent not in classes or not any(
            x["href"] == owner for x in classes[parent]["_links"].get("subclasses", [])
        ):
            return False
        owner = parent
    return target in classes


class StandardsCorpus:
    """A hash-pinned public corpus independent of benchmark/task/reference files."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.manifest = json.loads((path / "manifest.json").read_text())
        if (
            self.manifest.get("schema_version") != "public-standard-context-corpus-v1"
            or self.manifest.get("profile") != PROFILE
            or not self.manifest.get("source_only")
            or self.manifest.get("reference_files_included")
        ):
            raise WorkflowError("standard_context_profile_invalid")
        self.data = {}
        for name, info in self.manifest["files"].items():
            source = (path / name).resolve()
            if (
                not source.is_relative_to(self.path)
                or digest(source.read_bytes()) != info["sha256"]
            ):
                raise WorkflowError("standard_context_source_changed")
            self.data[name] = json.loads(source.read_text())
        self.model_classes = {
            c["_links"]["self"]["href"]: c
            for c in self.data["sdtm-model-2-0.json"]["classes"]
        }
        self.model_variables = {
            v["_links"]["self"]["href"]: v
            for c in self.model_classes.values()
            for v in c.get("classVariables", [])
        }
        self.cdash_model = {
            f["_links"]["self"]["href"]: f
            for c in self.data["cdash-model-1-3.json"]["classes"]
            for f in c.get("cdashModelFields", [])
        }
        self.domains = {
            d["name"]: (c, d)
            for c in self.data["sdtmig-3-4.json"]["classes"]
            for d in c.get("datasets", [])
        }
        self.ig_variables = {
            v["_links"]["self"]["href"]: (d["name"], v)
            for _, d in self.domains.values()
            for v in d.get("datasetVariables", [])
        }
        self.groups = []
        cdash = self.data["cdashig-2-3.json"]
        domain_headers = {
            d["_links"]["self"]["href"]: d
            for c in cdash["classes"]
            for d in c.get("domains", [])
        }
        for cls in cdash["classes"]:
            for group in [*cls.get("domains", []), *cls.get("scenarios", [])]:
                if not group.get("fields"):
                    continue
                domain = (
                    domain_headers[group["_links"]["parentDomain"]["href"]]
                    if "parentDomain" in group["_links"]
                    else group
                )
                self.groups.append(
                    {
                        "domain": domain["name"],
                        "form": domain["label"],
                        "section": group["_links"]["self"]["title"],
                        "fields": group["fields"],
                        "kind": "canonical_CDASH_collection_template",
                    }
                )
        qrs = self.data["qrs-cosmos-rows.json"]
        self.specializations = qrs["rows"][
            "cdisc_sdtm_dataset_specializations_latest.csv"
        ]
        collection = qrs["rows"]["cdisc_crf_specializations_draft.csv"]
        # Instrument labels are published group metadata; this inventory covers
        # every cached collection row, not a selected benchmark case roster.
        instrument_names = {"SIXMW1": "Six Minute Walk", "EQ5D02": "EQ-5D-5L"}
        for prefix, name in instrument_names.items():
            fields = [
                r
                for r in collection
                if r["vlm_group_id"].startswith(prefix) and r["display_hidden"] != "Y"
            ]
            if fields:
                self.groups.append(
                    {
                        "domain": fields[0]["domain"],
                        "form": name,
                        "section": None,
                        "fields": fields,
                        "kind": "COSMoS_draft_collection_template",
                    }
                )
        self.ct = {
            r["codelist"]["ncitCode"]: r
            for r in self.data["cdisc_ct.json"]["codelists"]
            if r["packageId"] == PROFILE["terminology"] and r["standard"] == "sdtmct"
        }

    def declarations(self, domain: str, field: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve public IG targets and verified implemented model targets only."""
        declared = {}
        for link in field.get("_links", {}).get("sdtmigDatasetMappingTargets", []):
            found = self.ig_variables.get(link["href"])
            if found:
                d, v = found
                declared[(d, v["name"])] = v
        implementation = field.get("_links", {}).get("implements", {}).get("href")
        model_field = self.cdash_model.get(implementation)
        # Dataset-specific links are the positive mapping for this collection
        # field. Use inherited model mappings as a fallback only; a same-named
        # model qualifier must not compete with an explicit IG target.
        if not declared and model_field and domain in self.domains:
            cls, _ = self.domains[domain]
            owner = cls["_links"].get("modelClass", {}).get("href")
            for link in model_field["_links"].get("sdtmClassMappingTargets", []):
                target = self.model_variables.get(link["href"])
                if target and inherited(
                    self.model_classes, owner, target["_links"]["parentClass"]["href"]
                ):
                    name = (
                        target["name"].replace("--", domain)
                        if target["name"].startswith("--")
                        else target["name"]
                    )
                    declared[(domain, name)] = target
        return [
            {
                "domain": d,
                "variable": v,
                "definition": m.get("definition", ""),
                "label": m.get("label", ""),
                "role": m.get("role", ""),
                "type": m.get("simpleDatatype", m.get("dataType", "")),
                "source_uri": m["_links"]["self"]["href"],
            }
            for (d, v), m in sorted(declared.items())
        ]

    def vocabulary(self, domain: str) -> list[dict[str, Any]]:
        """Return positive domain/class declarations; permission remains field-scoped."""
        if domain not in self.domains:
            return []
        cls, ds = self.domains[domain]
        owner = cls["_links"].get("modelClass", {}).get("href")
        result = {
            v["name"]: {
                "domain": domain,
                "variable": v["name"],
                "definition": v.get("definition", ""),
                "role": v.get("role", ""),
                "source_uri": v["_links"]["self"]["href"],
            }
            for v in ds.get("datasetVariables", [])
        }
        for v in self.model_variables.values():
            if not inherited(
                self.model_classes, owner, v["_links"]["parentClass"]["href"]
            ):
                continue
            name = (
                v["name"].replace("--", domain)
                if v["name"].startswith("--")
                else v["name"]
            )
            result.setdefault(
                name,
                {
                    "domain": domain,
                    "variable": name,
                    "definition": v.get("definition", ""),
                    "role": v.get("role", ""),
                    "source_uri": v["_links"]["self"]["href"],
                },
            )
        return list(result.values())

    def _alignment(
        self, fields: list[dict[str, Any]], source: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Align exact visible wording in source order, independent of ID values.

        F-number identifiers in this source contract encode object order only.
        Renumbering while retaining that order does not change the alignment.
        If a page is incomplete or duplicated, keep only uniquely matched fields.
        """
        local = [source, *source["neighbors"]]
        local = sorted(local, key=lambda r: int(re.search(r"\d+$", r["field_id"])[0]))
        if len(local) == len(fields) and all(
            normalized(r["question"]) == normalized(wording(f))
            for r, f in zip(local, fields)
        ):
            return {r["field_id"]: f for r, f in zip(local, fields)}
        result = {}
        for r in local:
            matches = [
                f
                for f in fields
                if normalized(r["question"]) == normalized(wording(f))
                and (
                    not r["response"].get("ui_text")
                    or normalized(r["response"]["ui_text"])
                    == normalized(f.get("prompt"))
                )
            ]
            if len(matches) == 1:
                result[r["field_id"]] = matches[0]
        return result

    def retrieve(self, source: dict[str, Any]) -> dict[str, Any]:
        """Retrieve original applicable evidence for every kind of collected field."""
        matches = [
            g
            for g in self.groups
            if normalized(g["form"]) == normalized(source["form"])
            and (
                g["section"] is None
                or normalized(g["section"]) == normalized(source["section"])
            )
        ]
        evidence = []
        for group in matches:
            alignment = self._alignment(group["fields"], source)
            focus = alignment.get(source["field_id"])
            if focus is None:
                continue
            domain = group["domain"]
            companions = []
            for fid, f in alignment.items():
                if fid == source["field_id"]:
                    continue
                companions.append(
                    {
                        "source_field_id": fid,
                        "collection_field": f.get("name", f.get("variable_name")),
                        "question": wording(f),
                        "declared_datatype": f.get(
                            "simpleDatatype", f.get("data_type")
                        ),
                        "targets": self.declarations(domain, f),
                        "mapping_instructions": f.get("mappingInstructions", ""),
                        # Shared branch instructions link result/description
                        # fields. Deliver that companion's implementation notes
                        # only for an exact shared instruction, avoiding a full
                        # page vocabulary or unrelated neighbor permissions.
                        "related_mapping_notes": (
                            f.get("implementationNotes") or ""
                            if f.get("mappingInstructions")
                            == focus.get("mappingInstructions")
                            and "If " in (focus.get("mappingInstructions") or "")
                            else ""
                        ),
                        "published_group": f.get("vlm_group_id"),
                    }
                )
            item = {
                "evidence_id": f"E{len(evidence)+1}",
                "kind": group["kind"],
                "domain": domain,
                "collection_field": focus.get("name", focus.get("variable_name")),
                "collection_metadata": focus,
                "declarations": self.declarations(domain, focus),
                "companions": companions,
                "specializations": [],
            }
            if group["kind"] == "COSMoS_draft_collection_template":
                for row in self.specializations:
                    if row["vlm_group_id"] != focus["vlm_group_id"]:
                        continue

                    def version(text):
                        return tuple(map(int, text.replace(".", "-").split("-")))

                    if (
                        version(row["sdtmig_start_version"]) > (3, 4)
                        or row["sdtmig_end_version"]
                        and version(row["sdtmig_end_version"]) < (3, 4)
                    ):
                        continue
                    if row["assigned_value"] and row["codelist"]:
                        cl = self.ct.get(row["codelist"])
                        if not cl or not any(
                            t["submissionValue"] == row["assigned_value"]
                            and t["ncitCode"] == row["assigned_term"]
                            for t in cl["terms"]
                        ):
                            raise WorkflowError(
                                "specialization_CT_corroboration_failed"
                            )
                    item["specializations"].append(row)
            evidence.append(item)
        domains = {e["domain"] for e in evidence}
        # Definitions for mentioned/linked targets and record identity are enough;
        # the entire domain registry is not permission to choose unrelated fields.
        relevant = set()
        for e in evidence:
            relevant.update(t["variable"] for t in e["declarations"])
            for r in e["specializations"]:
                relevant.add(r["sdtm_variable"])
            texts = canonical(e["collection_metadata"])
            texts += " ".join(
                c.get("related_mapping_notes", "") for c in e["companions"]
            )
            relevant.update(re.findall(r"\b[A-Z]{2,8}[A-Z0-9]{2,12}\b", texts))
        declarations = [
            v for d in domains for v in self.vocabulary(d) if v["variable"] in relevant
        ]
        context = {
            "schema_version": "public-standard-field-context-v1",
            "profile": PROFILE,
            "source": source,
            "evidence": evidence,
            # Keep source-derived obligations explicit beside original text so
            # record bindings/conditions cannot disappear in a long metadata blob.
            "requirements": derive(source, evidence),
            "target_declarations": declarations,
            "unmatched_source": not evidence,
            "source_provenance": {
                "corpus_manifest_sha256": digest(
                    (self.path / "manifest.json").read_bytes()
                ),
                "files": self.manifest["files"],
            },
        }
        if len(canonical(context).encode()) > 64000:
            raise WorkflowError("standard_context_capacity_exceeded")
        return context
