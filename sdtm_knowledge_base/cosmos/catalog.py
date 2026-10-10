"""Offline COSMoS lookup and complete-record evidence for RAG consumers."""

import json
from collections.abc import Mapping

from .source import FILES, CosmosRecord, parse_export, version_tuple


class CosmosCatalog:
    """Expose structured templates without changing any mapper or ranking policy."""

    def __init__(self, records: tuple[CosmosRecord, ...]):
        """Index already parsed immutable records; reject duplicate scoped identities."""
        self._records = records
        self._by_key = {(record.kind, record.source_id): record for record in records}
        if len(self._by_key) != len(records):
            raise ValueError("Duplicate COSMoS record identity")

    @classmethod
    def from_exports(cls, exports: Mapping[str, bytes]) -> "CosmosCatalog":
        """Parse supplied BC/SDTM/CRF CSV bytes; these are not verified snapshots."""
        if set(exports) != set(FILES):
            raise ValueError("Provide exactly the bc, sdtm, and crf exports")
        return cls(
            tuple(
                record for kind in FILES for record in parse_export(kind, exports[kind])
            )
        )

    @staticmethod
    def _check_filters(kind: str | None, sdtmig_version: str | None) -> None:
        """Reject unknown kinds and ambiguous version labels at the API boundary."""
        if kind is not None and kind not in FILES:
            raise ValueError(f"Unknown COSMoS kind: {kind}")
        if sdtmig_version is not None:
            version_tuple(sdtmig_version)

    def records(
        self,
        *,
        kind: str | None = None,
        domain: str | None = None,
        sdtmig_version: str | None = "3.4",
        include_drafts: bool = False,
    ) -> tuple[CosmosRecord, ...]:
        """Select complete records; SDTMIG ranges apply to SDTM, not CDASH CRFs."""
        self._check_filters(kind, sdtmig_version)
        target = version_tuple(sdtmig_version) if sdtmig_version is not None else None
        selected = []
        for record in self._records:
            if (kind is not None and kind != record.kind) or (
                domain is not None and domain != record.value("domain")
            ):
                continue
            if record.status == "draft" and not include_drafts:
                continue
            # COSMoS defines start as earliest and end as last applicable version.
            if record.kind == "sdtm" and target is not None:
                start, end = record.value("sdtmig_start_version"), record.value(
                    "sdtmig_end_version"
                )
                if target < version_tuple(start) or (
                    end and target > version_tuple(end)
                ):
                    continue
            selected.append(record)
        return tuple(selected)

    def lookup(
        self,
        kind: str,
        source_id: str,
        *,
        sdtmig_version: str | None = "3.4",
        include_drafts: bool = False,
    ) -> CosmosRecord:
        """Return one available record; unknown, draft-disabled or incompatible IDs fail."""
        self._check_filters(kind, sdtmig_version)
        record = self._by_key.get((kind, source_id))
        if record is None or record not in self.records(
            kind=kind, sdtmig_version=sdtmig_version, include_drafts=include_drafts
        ):
            raise LookupError(
                f"COSMoS record unavailable for selected filters: {kind}:{source_id}"
            )
        return record

    def evidence_documents(
        self,
        *,
        kind: str | None = None,
        domain: str | None = None,
        sdtmig_version: str | None = "3.4",
        include_drafts: bool = False,
    ) -> list[dict]:
        """Return whole templates compatible with id/text retrieval; preserve structured rows."""
        documents = []
        for record in self.records(
            kind=kind,
            domain=domain,
            sdtmig_version=sdtmig_version,
            include_drafts=include_drafts,
        ):
            structured = record.to_dict()
            documents.append(
                {
                    "id": record.id,
                    "kind": f"cosmos_{record.kind}",
                    "text": json.dumps(
                        {
                            "metadata": structured["metadata"],
                            "rows": [dict(row.fields) for row in record.rows],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "source": structured["source"],
                    "record": structured,
                }
            )
        return documents

    def _references(self) -> tuple[list[dict], list[dict]]:
        """Resolve explicit entity references and report gaps without inferred repairs."""
        edges: list[dict] = []
        gaps: list[dict] = []
        for record in self._records:
            references = (
                [("bc", "parent_bc_id", "child_of")]
                if record.kind == "bc"
                else [("bc", "bc_id", "specializes")]
            )
            if record.kind == "crf":
                references.append(("sdtm", "vlm_group_id", "collects_as"))
            for target_kind, field, relation in references:
                target_id = record.value(field)
                if not target_id:
                    continue
                target = self._by_key.get((target_kind, target_id))
                if target is None:
                    gaps.append(
                        {
                            "source": record.id,
                            "target": target_id,
                            "field": field,
                            "reason": "unresolved_entity_reference",
                        }
                    )
                else:
                    edges.append(
                        {
                            "source": record.id,
                            "target": target.id,
                            "relation": relation,
                            "source_field": field,
                        }
                    )
            if record.kind == "sdtm":
                variables = {row.value("sdtm_variable") for row in record.rows}
                for row in record.rows:
                    for field in ("subject", "object"):
                        target_id = row.value(field)
                        if target_id and target_id not in variables:
                            gaps.append(
                                {
                                    "source": record.id,
                                    "record_number": row.number,
                                    "target": target_id,
                                    "field": field,
                                    "reason": "unresolved_in_specialization_scope",
                                }
                            )
        return edges, gaps

    def links(
        self, *, include_drafts: bool = False, sdtmig_version: str | None = "3.4"
    ) -> list[dict]:
        """Return entity edges only when both endpoints pass the same visibility filters."""
        available = {
            record.id
            for record in self.records(
                include_drafts=include_drafts, sdtmig_version=sdtmig_version
            )
        }
        return [
            edge
            for edge in self._references()[0]
            if edge["source"] in available and edge["target"] in available
        ]

    def issues(self) -> list[dict]:
        """Report unresolved source references, including drafts; no standard-error claim."""
        return self._references()[1]

    def summary(self) -> dict:
        """Count distinct entities and original CSV records independently of filters."""
        return {
            "records": {
                kind: sum(record.kind == kind for record in self._records)
                for kind in FILES
            },
            "csv_records": {
                kind: sum(
                    len(record.rows) for record in self._records if record.kind == kind
                )
                for kind in FILES
            },
            "unresolved_references": len(self.issues()),
            "verified_snapshot": bool(self._records)
            and all(record.verified_snapshot for record in self._records),
        }
