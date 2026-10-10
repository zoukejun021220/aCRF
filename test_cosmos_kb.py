"""Exercise COSMoS import boundaries and retrieval evidence without model calls."""

import csv
import importlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def export_bytes(rows: list[dict[str, str]]) -> bytes:
    """Serialize literal test rows as CSV, including quoted and multiline values."""
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fixture_exports() -> dict[str, bytes]:
    """Return small independent concepts with reused variable names and a draft."""
    bc = [
        {
            "package_date": "2026-05-26",
            "bc_id": "C81328",
            "short_name": "Weight",
            "parent_bc_id": "",
            "definition": "Body mass, as measured.\nOriginal value.",
            "synonyms": "Body weight;Body mass",
            "dec_id": "C70856",
        },
        {
            "package_date": "2026-05-26",
            "bc_id": "C25298",
            "short_name": "Systolic",
            "parent_bc_id": "C81328",
            "definition": "Synthetic second concept",
            "synonyms": "",
            "dec_id": "C70856",
        },
    ]
    base = {
        "package_date": "2026-05-26",
        "bc_id": "C81328",
        "domain": "VS",
        "vlm_group_id": "WEIGHT",
        "short_name": "Weight",
        "sdtmig_start_version": "3-2",
        "sdtmig_end_version": "3-4",
        "sdtm_variable": "",
        "subject": "",
        "object": "",
        "predicate_term": "",
        "linking_phrase": "",
        "assigned_value": "",
        "value_list": "",
        "origin_type": "",
        "mandatory_variable": "Y",
    }
    sdtm = [
        {
            **base,
            "sdtm_variable": "VSTESTCD",
            "assigned_value": "WEIGHT",
            "origin_type": "Assigned",
        },
        {
            **base,
            "sdtm_variable": "VSORRES",
            "subject": "VSORRES",
            "object": "VSTESTCD",
            "predicate_term": "IS_RESULT_OF",
            "linking_phrase": "is the result of the test in",
            "origin_type": "Collected",
        },
        {
            **base,
            "sdtm_variable": "VSORRESU",
            "value_list": "LB;g;kg",
            "subject": "VSORRESU",
            "object": "VSORRES",
            "predicate_term": "IS_UNIT_FOR",
        },
        {
            **base,
            "vlm_group_id": "SYSBP",
            "bc_id": "C25298",
            "short_name": "Systolic",
            "sdtmig_start_version": "3-5",
            "sdtmig_end_version": "",
            "sdtm_variable": "VSORRES",
            "assigned_value": "DO_NOT_MERGE_WITH_WEIGHT",
        },
    ]
    crf = [
        {
            "package_date": "2026-06-30",
            "bc_id": "C81328",
            "domain": "VS",
            "crf_group_id": "WEIGHT_DENORMALIZED",
            "vlm_group_id": "WEIGHT",
            "short_name": "Weight (Denormalized)",
            "standard": "CDASHIG",
            "standard_start_version": "2-1",
            "standard_end_version": "",
            "crf_item": "WEIGHT_RESULT",
            "sdtm_target_variable": "VSORRES;VSTESTCD",
            "sdtm_annotation": "VSORRES when VSTESTCD = WEIGHT",
            "prompt": "Weight",
        }
    ]
    return {
        "bc": export_bytes(bc),
        "sdtm": export_bytes(sdtm),
        "crf": export_bytes(crf),
    }


class CosmosCatalogTests(unittest.TestCase):
    """Check source preservation, rejection paths, and consumer-facing evidence."""

    def setUp(self) -> None:
        """Load the actual implementation; absence is an explicit feature failure."""
        try:
            self.cosmos = importlib.import_module("sdtm_knowledge_base.cosmos")
        except ModuleNotFoundError:
            self.fail("COSMoS knowledge source has not been implemented")
        self.exports = fixture_exports()

    def catalog(self):
        """Parse fresh literal exports without network access or shared state."""
        return self.cosmos.CosmosCatalog.from_exports(self.exports)

    def test_complete_specialization_preserves_scope_and_provenance(self) -> None:
        """A plausible result-only mapping must fail the relationship assertions."""
        catalog = self.catalog()
        record = catalog.lookup("sdtm", "WEIGHT").to_dict()
        variables = {row["values"]["sdtm_variable"]: row for row in record["rows"]}
        self.assertEqual(set(variables), {"VSTESTCD", "VSORRES", "VSORRESU"})
        self.assertEqual(variables["VSTESTCD"]["values"]["assigned_value"], "WEIGHT")
        self.assertEqual(variables["VSORRESU"]["values"]["value_list"], "LB;g;kg")
        self.assertEqual(variables["VSORRES"]["values"]["object"], "VSTESTCD")
        self.assertEqual(variables["VSORRES"]["record_number"], 2)
        self.assertEqual(record["status"], "published")
        self.assertFalse(record["source"]["verified_snapshot"])
        self.assertEqual(
            catalog.lookup("bc", "C81328").to_dict()["rows"][0]["values"]["definition"],
            "Body mass, as measured.\nOriginal value.",
        )
        # A returned dictionary must not let callers mutate subsequent answers.
        variables["VSTESTCD"]["values"]["assigned_value"] = "CORRUPTED"
        self.assertEqual(
            catalog.lookup("sdtm", "WEIGHT").to_dict()["rows"][0]["values"][
                "assigned_value"
            ],
            "WEIGHT",
        )

    def test_version_ranges_are_inclusive_and_do_not_merge_variables(self) -> None:
        """The same variable in different templates retains its own version/scope."""
        catalog = self.catalog()
        for version in ("3-2", "3.4"):
            self.assertEqual(
                catalog.lookup("sdtm", "WEIGHT", sdtmig_version=version).source_id,
                "WEIGHT",
            )
        for version in ("3.1.2", "3.5"):
            with self.subTest(version=version), self.assertRaises(LookupError):
                catalog.lookup("sdtm", "WEIGHT", sdtmig_version=version)
        self.assertEqual(
            catalog.lookup("sdtm", "SYSBP", sdtmig_version="3.5").to_dict()["rows"][0][
                "values"
            ]["assigned_value"],
            "DO_NOT_MERGE_WITH_WEIGHT",
        )

    def test_drafts_are_excluded_by_default_and_stay_labelled(self) -> None:
        """Draft CRF answers cannot quietly enter published retrieval evidence."""
        catalog = self.catalog()
        with self.assertRaises(LookupError):
            catalog.lookup("crf", "WEIGHT_DENORMALIZED")
        draft = catalog.lookup("crf", "WEIGHT_DENORMALIZED", include_drafts=True)
        self.assertEqual(draft.to_dict()["status"], "draft")
        self.assertEqual(
            draft.to_dict()["rows"][0]["values"]["sdtm_target_variable"],
            "VSORRES;VSTESTCD",
        )
        self.assertFalse(
            any(doc["kind"] == "cosmos_crf" for doc in catalog.evidence_documents())
        )
        self.assertTrue(
            any(
                doc["kind"] == "cosmos_crf"
                for doc in catalog.evidence_documents(include_drafts=True)
            )
        )

    def test_unknown_ids_invalid_filters_and_empty_sources_fail(self) -> None:
        """Invalid input must not be converted into a plausible empty success."""
        catalog = self.catalog()
        with self.assertRaises(LookupError):
            catalog.lookup("sdtm", "MISSING")
        for kwargs in ({"kind": "bad"}, {"sdtmig_version": "latest"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                catalog.evidence_documents(**kwargs)
        for exports in (
            {},
            {**self.exports, "bc": b""},
            {**self.exports, "bc": b"wrong,header\nx,y\n"},
        ):
            with self.subTest(exports=list(exports)), self.assertRaises(ValueError):
                self.cosmos.CosmosCatalog.from_exports(exports)

    def test_conflicting_group_metadata_and_duplicate_children_fail(self) -> None:
        """A partial overwrite or repeated variable cannot silently redefine a template."""
        original = list(csv.DictReader(io.StringIO(self.exports["sdtm"].decode())))
        conflict = [dict(row) for row in original]
        conflict[1]["domain"] = "LB"
        for rows in (conflict, original + [original[0]]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.cosmos.CosmosCatalog.from_exports(
                    {**self.exports, "sdtm": export_bytes(rows)}
                )

    def test_malformed_csv_and_invalid_record_boundaries_fail(self) -> None:
        """Reject malformed encodings, record shapes, identities, dates and ranges."""
        rows = list(csv.DictReader(io.StringIO(self.exports["sdtm"].decode())))
        invalid = [b"\xff", self.exports["sdtm"].splitlines()[0] + b"\n"]
        for key, value in (
            ("vlm_group_id", ""),
            ("short_name", ""),
            ("sdtm_variable", ""),
            ("package_date", "not-a-date"),
            ("sdtmig_start_version", "latest"),
            ("sdtmig_end_version", "3-1"),
        ):
            changed = [{**row, key: value} for row in rows[:3]]
            invalid.append(export_bytes(changed))
        header, remainder = self.exports["sdtm"].split(b"\r\n", 1)
        invalid.append(header + b",package_date\r\n" + remainder)
        invalid.append(header + b"\r\nonly_one_cell\r\n")
        for index, content in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.cosmos.CosmosCatalog.from_exports(
                    {**self.exports, "sdtm": content}
                )

    def test_explicit_links_respect_visibility_and_empty_bc_elements_are_valid(
        self,
    ) -> None:
        """Keep source entity links while preventing draft or future-version leakage."""
        catalog = self.catalog()
        allowed = {record.id for record in catalog.records()}
        self.assertTrue(catalog.links())
        self.assertTrue(
            all(
                edge["source"] in allowed and edge["target"] in allowed
                for edge in catalog.links()
            )
        )
        draft = catalog.lookup("crf", "WEIGHT_DENORMALIZED", include_drafts=True)
        self.assertFalse(any(edge["source"] == draft.id for edge in catalog.links()))
        self.assertTrue(
            any(
                edge["source"] == draft.id and edge["relation"] == "collects_as"
                for edge in catalog.links(include_drafts=True)
            )
        )
        self.assertEqual(
            catalog.lookup("sdtm", "SYSBP", sdtmig_version=None).source_id, "SYSBP"
        )
        self.assertEqual(catalog.evidence_documents(domain="NO_SUCH_DOMAIN"), [])
        rows = list(csv.DictReader(io.StringIO(self.exports["bc"].decode())))
        rows[0]["dec_id"] = ""
        bare = self.cosmos.CosmosCatalog.from_exports(
            {**self.exports, "bc": export_bytes(rows)}
        )
        self.assertEqual(
            bare.lookup("bc", "C81328").to_dict()["rows"][0]["values"]["dec_id"], ""
        )

    def test_unresolved_source_relationships_are_preserved_and_reported(self) -> None:
        """Source typos or external references remain evidence, never inferred edges."""
        rows = list(csv.DictReader(io.StringIO(self.exports["sdtm"].decode())))
        rows[1]["object"] = "TYPO_TARGET"
        catalog = self.cosmos.CosmosCatalog.from_exports(
            {**self.exports, "sdtm": export_bytes(rows)}
        )
        self.assertEqual(
            catalog.lookup("sdtm", "WEIGHT").to_dict()["rows"][1]["values"]["object"],
            "TYPO_TARGET",
        )
        issues = catalog.issues()
        self.assertTrue(
            any(
                issue["target"] == "TYPO_TARGET"
                and issue["reason"] == "unresolved_in_specialization_scope"
                for issue in issues
            )
        )
        self.assertFalse(
            any(edge["target"].endswith(":TYPO_TARGET") for edge in catalog.links())
        )

    def test_evidence_documents_preserve_complete_records(self) -> None:
        """Check the public evidence contract without an experimental retriever.

        Select a literal fixture ID and compare the complete record, so omitted
        qualifiers or provenance fail independently of retrieval ranking.
        """
        catalog = self.catalog()
        documents = catalog.evidence_documents(kind="sdtm", domain="VS")
        evidence = next(
            doc for doc in documents if doc["record"]["source_id"] == "WEIGHT"
        )
        self.assertEqual(evidence["record"], catalog.lookup("sdtm", "WEIGHT").to_dict())
        self.assertEqual(evidence["record"]["source_id"], "WEIGHT")
        self.assertEqual(len(evidence["record"]["rows"]), 3)
        self.assertEqual(
            evidence["record"]["rows"][2]["values"]["predicate_term"], "IS_UNIT_FOR"
        )
        self.assertEqual(
            documents, self.catalog().evidence_documents(kind="sdtm", domain="VS")
        )


class CosmosSnapshotTests(unittest.TestCase):
    """Verify the shipped public snapshot, its rebuild, and a real offline CLI."""

    def setUp(self) -> None:
        """Require the snapshot API before accessing bundled, network-free fixtures."""
        self.cosmos = importlib.import_module("sdtm_knowledge_base.cosmos")
        self.assertTrue(
            callable(getattr(self.cosmos, "load_snapshot", None)),
            "Verified COSMoS snapshot loader has not been implemented",
        )
        self.directory = self.cosmos.DEFAULT_SNAPSHOT

    def test_pinned_snapshot_has_all_source_rows_and_correct_weight_mapping(
        self,
    ) -> None:
        """Counts and independently selected cells catch omissions and template merging."""
        catalog = self.cosmos.load_snapshot(self.directory)
        self.assertEqual(
            catalog.summary()["records"], {"bc": 1475, "sdtm": 1475, "crf": 303}
        )
        self.assertEqual(
            catalog.summary()["csv_records"], {"bc": 6306, "sdtm": 13922, "crf": 2073}
        )
        weight = catalog.lookup("sdtm", "WEIGHT").to_dict()
        variables = {
            row["values"]["sdtm_variable"]: row["values"] for row in weight["rows"]
        }
        self.assertEqual(len(variables), 8)
        self.assertEqual(variables["VSTESTCD"]["assigned_value"], "WEIGHT")
        self.assertEqual(variables["VSORRESU"]["value_list"], "LB;g;kg")
        self.assertEqual(variables["VSSTRESN"]["origin_type"], "Derived")
        self.assertTrue(weight["source"]["verified_snapshot"])
        self.assertEqual(
            weight["source"]["commit"], "031429b1d14823721991cd23ee88a11616686ce3"
        )
        self.assertEqual(weight["source"]["license"], "CC-BY-4.0")
        missing = {
            issue["target"]
            for issue in catalog.issues()
            if issue["reason"] == "unresolved_entity_reference"
        }
        self.assertEqual(missing, {"EQ5D02", "SIXMW10"})

    def test_every_shipped_csv_cell_survives_grouping(self) -> None:
        """Independently reconstruct all 22,301 CSV records from catalog evidence."""
        from sdtm_knowledge_base.cosmos.source import FILES

        catalog = self.cosmos.load_snapshot(self.directory)
        for kind, filename in FILES.items():
            with (self.directory / "raw" / filename).open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                expected = list(csv.DictReader(stream))
            rows = [
                row
                for record in catalog.records(
                    kind=kind, include_drafts=True, sdtmig_version=None
                )
                for row in record.to_dict()["rows"]
            ]
            self.assertEqual(len({row["record_number"] for row in rows}), len(expected))
            actual = [
                row["values"]
                for row in sorted(rows, key=lambda row: row["record_number"])
            ]
            self.assertEqual(actual, expected, kind)

    def test_snapshot_rebuild_is_identical_and_refuses_to_overwrite(self) -> None:
        """A second offline import must reproduce every file and preserve existing data."""
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "snapshot"
            self.cosmos.build_snapshot(self.directory / "raw", target)
            originals = {
                path.relative_to(self.directory): path.read_bytes()
                for path in self.directory.rglob("*")
                if path.is_file()
            }
            rebuilt = {
                path.relative_to(target): path.read_bytes()
                for path in target.rglob("*")
                if path.is_file()
            }
            self.assertEqual(rebuilt, originals)
            with self.assertRaises(FileExistsError):
                self.cosmos.build_snapshot(self.directory / "raw", target)
            self.assertEqual(
                (target / "manifest.json").read_bytes(),
                originals[Path("manifest.json")],
            )

    def test_modified_source_or_manifest_is_rejected(self) -> None:
        """Changing plausible metadata cannot retain verified-source status."""
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "snapshot"
            shutil.copytree(self.directory, target)
            source = target / "raw/export/cdisc_sdtm_dataset_specializations_latest.csv"
            original = source.read_bytes()
            source.write_bytes(original.replace(b"VSORRESU", b"VSSTRESU", 1))
            with self.assertRaises(ValueError):
                self.cosmos.load_snapshot(target)
            source.write_bytes(original)
            manifest = json.loads((target / "manifest.json").read_text())
            manifest["commit"] = "0" * 40
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.cosmos.load_snapshot(target)

    def test_missing_source_does_not_leave_a_partial_snapshot(self) -> None:
        """Import validates all required bytes before creating the destination."""
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "new"
            with self.assertRaises(ValueError):
                self.cosmos.build_snapshot(Path(name) / "missing", target)
            self.assertFalse(target.exists())

    def test_real_cli_returns_json_and_fails_for_unavailable_draft(self) -> None:
        """Exercise argument parsing and output, with no model initialization."""
        base = [sys.executable, "-m", "sdtm_knowledge_base.cosmos", "show"]
        root = Path(__file__).resolve().parent
        result = subprocess.run(
            base + ["sdtm", "WEIGHT"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["source_id"], "WEIGHT")
        rejected = subprocess.run(
            base + ["crf", "WEIGHT_DENORMALIZED"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(rejected.stdout, "")
        accepted = subprocess.run(
            base + ["crf", "WEIGHT_DENORMALIZED", "--include-drafts"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(json.loads(accepted.stdout)["status"], "draft")

    def test_existing_kb_loader_can_opt_in_without_changing_default_resources(
        self,
    ) -> None:
        """Exercise the real file-based loader without initializing unrelated ML models."""
        path = (
            Path(__file__).resolve().parent
            / "pipeline/stage_b_mapper_module/kb_handlers/kb_loader.py"
        )
        specification = importlib.util.spec_from_file_location(
            "cosmos_test_kb_loader", path
        )
        if specification is None or specification.loader is None:
            self.fail("Cannot load the existing KB loader module")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as name:
            loader = module.KBLoader(name)
            self.assertTrue(
                hasattr(loader, "load_cosmos_resources"),
                "Existing KB loader cannot load COSMoS yet",
            )
            loader.load_kb_resources()
            self.assertIsNone(loader.cosmos)
            loader.load_kb_resources(include_cosmos=True)
            self.assertEqual(loader.cosmos.lookup("sdtm", "WEIGHT").source_id, "WEIGHT")
            valid = loader.cosmos
            with self.assertRaises(ValueError):
                loader.load_cosmos_resources(str(Path(name) / "missing"))
            self.assertIs(loader.cosmos, valid)


if __name__ == "__main__":
    unittest.main()
