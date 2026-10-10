"""Parse COSMoS CSV exports without discarding source fields or record scope."""

import csv
import hashlib
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

COMMIT = "031429b1d14823721991cd23ee88a11616686ce3"
REPOSITORY = "https://github.com/cdisc-org/COSMoS"
FILES = {
    "bc": "export/cdisc_biomedical_concepts_latest.csv",
    "sdtm": "export/cdisc_sdtm_dataset_specializations_latest.csv",
    "crf": "export/cdisc_crf_specializations_draft.csv",
}
HASHES = {
    FILES["bc"]: "4273fdd56880e5d051a1c65cb90062f994a3836002f267ae4385097d51645ed6",
    FILES["sdtm"]: "c5530d0d425e36f0bb36ab1cbc18be493147f968b0dedbd39feb864cb86b4dd2",
    FILES["crf"]: "6f2af712a9fd3d66f2cc20cbf68732526fd179d8fe9c69adbdd2a92ac3f4b28b",
    "README.md": "9f1819c1b7971567ef7a19bac30243f11804b8911bc1ee30b2671b632afc9649",
    "LICENSE": "7de7787967d7f4d687a73d553b12940f21f2d539c31fc7c75462167b12746bac",
}
GROUP_KEYS = {"bc": "bc_id", "sdtm": "vlm_group_id", "crf": "crf_group_id"}
CHILD_KEYS = {"bc": "dec_id", "sdtm": "sdtm_variable", "crf": "crf_item"}
COMMON = {
    "bc": ("package_date", "bc_id", "short_name", "parent_bc_id", "definition"),
    "sdtm": (
        "package_date",
        "vlm_group_id",
        "short_name",
        "bc_id",
        "domain",
        "sdtmig_start_version",
        "sdtmig_end_version",
    ),
    "crf": (
        "package_date",
        "crf_group_id",
        "short_name",
        "bc_id",
        "domain",
        "vlm_group_id",
        "standard",
        "standard_start_version",
        "standard_end_version",
    ),
}


def version_tuple(value: str) -> tuple[int, int, int]:
    """Parse an explicit two/three-part standards version; reject floating labels."""
    if not isinstance(value, str) or not re.fullmatch(r"\d+[.-]\d+(?:[.-]\d+)?", value):
        raise ValueError(f"Expected an explicit standards version, got {value!r}")
    parts = [int(part) for part in re.split(r"[.-]", value)]
    return parts[0], parts[1], parts[2] if len(parts) == 3 else 0


@dataclass(frozen=True)
class SourceRow:
    """Immutable parsed CSV record; number excludes the header, not physical lines."""

    number: int
    fields: tuple[tuple[str, str], ...]

    def value(self, key: str) -> str:
        """Return a source field verbatim, or empty text for an absent optional field."""
        return dict(self.fields).get(key, "")


@dataclass(frozen=True)
class CosmosRecord:
    """Keep one entire concept/template, including every original child record."""

    kind: str
    source_id: str
    package_date: str
    rows: tuple[SourceRow, ...]
    source_sha256: str
    verified_snapshot: bool = False

    @property
    def id(self) -> str:
        """Return a kind/version-scoped identity; repeated variables never merge."""
        return f"cosmos:{self.kind}:{self.package_date}:{self.source_id}"

    @property
    def status(self) -> str:
        """Distinguish the upstream draft CRF export from published BC/SDTM exports."""
        return "draft" if self.kind == "crf" else "published"

    def value(self, key: str) -> str:
        """Read the first source row; only COMMON metadata is guaranteed group-invariant."""
        return self.rows[0].value(key)

    def to_dict(self) -> dict:
        """Return a detached evidence record, so consumer edits cannot mutate the KB."""
        source_path = FILES[self.kind]
        return {
            "id": self.id,
            "kind": self.kind,
            "source_id": self.source_id,
            "package_date": self.package_date,
            "status": self.status,
            "title": self.value("short_name"),
            "domain": self.value("domain"),
            "metadata": {key: self.value(key) for key in COMMON[self.kind]},
            "rows": [
                {"record_number": row.number, "values": dict(row.fields)}
                for row in self.rows
            ],
            "source": {
                "path": source_path,
                "sha256": self.source_sha256,
                "verified_snapshot": self.verified_snapshot,
                "commit": COMMIT if self.verified_snapshot else None,
                "url": (
                    f"{REPOSITORY}/blob/{COMMIT}/{source_path}"
                    if self.verified_snapshot
                    else None
                ),
                "license": "CC-BY-4.0" if self.verified_snapshot else None,
                "locator": "record_number is 1-based CSV data record; header excluded",
            },
        }


def parse_export(
    kind: str, content: bytes, verified: bool = False
) -> tuple[CosmosRecord, ...]:
    """Validate CSV boundaries and group records; malformed/conflicting input fails."""
    if kind not in FILES or not isinstance(content, bytes) or not content:
        raise ValueError("Expected a known export kind and nonempty CSV bytes")
    try:
        reader = csv.DictReader(
            io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True
        )
        headers = reader.fieldnames or []
        required = set(COMMON[kind]) | {CHILD_KEYS[kind]}
        if kind == "sdtm":
            required |= {"subject", "object", "predicate_term"}
        if kind == "crf":
            required |= {"sdtm_target_variable", "sdtm_annotation"}
        if len(headers) != len(set(headers)) or not required.issubset(headers):
            raise ValueError(f"Invalid {kind} CSV headers")
        groups: dict[str, list[SourceRow]] = defaultdict(list)
        for number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Wrong column count in {kind} record {number}")
            source_id = row[GROUP_KEYS[kind]]
            if not source_id.strip() or not row["short_name"].strip():
                raise ValueError(f"Missing identity/title in {kind} record {number}")
            date.fromisoformat(row["package_date"])
            groups[source_id].append(SourceRow(number, tuple(row.items())))
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise ValueError(f"Cannot parse COSMoS {kind} export: {exc}") from exc
    if not groups:
        raise ValueError(f"No data records in {kind} export")

    records = []
    source_sha256 = hashlib.sha256(content).hexdigest()
    # A latest export must have one coherent version and unique children per ID.
    for source_id, rows in sorted(groups.items()):
        for key in COMMON[kind]:
            if len({row.value(key) for row in rows}) != 1:
                raise ValueError(f"Conflicting {kind}:{source_id} metadata: {key}")
        children = [row.value(CHILD_KEYS[kind]) for row in rows]
        nonempty = [child for child in children if child]
        if len(nonempty) != len(set(nonempty)) or (
            kind != "bc" and len(nonempty) != len(children)
        ):
            raise ValueError(f"Duplicate/missing child in {kind}:{source_id}")
        if kind in ("sdtm", "crf"):
            prefix = "sdtmig" if kind == "sdtm" else "standard"
            start = version_tuple(rows[0].value(f"{prefix}_start_version"))
            end = rows[0].value(f"{prefix}_end_version")
            if end and version_tuple(end) < start:
                raise ValueError(f"Reversed standards version range: {source_id}")
        records.append(
            CosmosRecord(
                kind,
                source_id,
                rows[0].value("package_date"),
                tuple(rows),
                source_sha256,
                verified,
            )
        )
    return tuple(records)
