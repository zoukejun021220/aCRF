"""Complete verified COSMoS retrieval and versioned rule fallback using real SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from sdtm_knowledge_base.cosmos.catalog import CosmosCatalog
from sdtm_knowledge_base.cosmos.source import (COMMIT, FILES, HASHES,
                                               CosmosRecord, parse_export)

from ..graphrag.field_contracts import FrozenModel, StandardProfile
from .common import WorkflowError, canonical, digest, parse, unique
from .understanding import MappingQuery


class TemplateRow(FrozenModel):
    """Retain original structured fields as immutable JSON with exact row identity."""

    variable: str
    source_locator: str
    values_json: str

    def values(self) -> dict[str, str]:
        """Return a detached original row; malformed serialized rows fail closed."""
        data = json.loads(self.values_json)
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise WorkflowError("invalid_template_row")
        return cast(dict[str, str], data)


class Candidate(FrozenModel):
    """A backend-owned evidence bundle, not a model-created or approved mapping."""

    ref: str
    kind: Literal["released_template", "rule_evidence_bundle"]
    title: str
    domain: str
    sdtmig: str
    source_url: str
    source_sha256: str
    source_version: str
    excerpt: str = ""
    rows: tuple[TemplateRow, ...]
    coverage_limits: tuple[str, ...] = ()

    def row(self, variable: str) -> TemplateRow:
        """Resolve a variable inside this occurrence only, never across templates."""
        for row in self.rows:
            if row.variable == variable:
                return row
        raise WorkflowError("variable_not_in_candidate")

    def wire(self) -> dict[str, object]:
        """Expose ordinary rows to the model without making it decode nested JSON."""
        result = self.model_dump(mode="json", exclude={"rows"})
        result["rows"] = [
            {"source_locator": row.source_locator, **row.values()} for row in self.rows
        ]
        return result


class StandardRule(FrozenModel):
    """A narrowly curated published rule with original excerpt and adapter metadata."""

    ref: str
    title: str
    concept_terms: tuple[str, ...]
    property_terms: tuple[str, ...]
    sdtmig_versions: tuple[str, ...]
    source_version: str
    source_url: str
    source_locator: str
    excerpt: str
    excerpt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: str
    variables: tuple[str, ...]
    interpretation: str


class RuleCorpus(FrozenModel):
    """Explicitly limited public rule inventory; not a study-rule repository."""

    schema_version: Literal[1] = 1
    rules: tuple[StandardRule, ...]


def tokens(text: str) -> set[str]:
    """Normalize lexical tokens only; clinical synonym facts come from the resource."""
    # Preserve clinical words/numbers, but do not penalize a richer source phrase
    # for ordinary inflection or connecting words absent from a short KB label.
    words = {
        (
            "walk"
            if word == "walked"
            else word[:-1] if word.endswith("s") and len(word) > 4 else word
        )
        for word in re.findall(r"[a-z0-9]+", text.lower())
    }
    return words - {
        "the",
        "of",
        "a",
        "an",
        "was",
        "is",
        "for",
        "and",
        "what",
        "this",
        "at",
        "during",
        "test",
    }


def timepoints(text: str) -> set[tuple[str, str]]:
    """Read explicit numeric timepoints only; never infer a clinical timepoint."""
    return {
        (number, unit.rstrip("s"))
        for number, unit in re.findall(
            r"\b(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?|weeks?)\b",
            text.lower(),
        )
    }


class Knowledge:
    """Query an isolated in-memory SQLite index over verified published templates."""

    def __init__(
        self,
        snapshot: Path,
        profile: StandardProfile,
        rules: Path | None = None,
        *,
        include_templates: bool = True,
    ) -> None:
        """Verify exact snapshot bytes before indexing; never download or mutate KB."""
        if profile.cosmos_commit != COMMIT or profile.include_drafts:
            raise WorkflowError("unsupported_knowledge_profile")
        records: list[CosmosRecord] = []
        try:
            for name, expected in HASHES.items():
                if digest((snapshot / "raw" / name).read_bytes()) != expected:
                    raise WorkflowError("knowledge_hash_mismatch")
            for kind, name in FILES.items():
                records.extend(
                    parse_export(
                        kind, (snapshot / "raw" / name).read_bytes(), verified=True
                    )
                )
            corpus = (
                parse(RuleCorpus, rules.read_text()) if rules else RuleCorpus(rules=())
            )
        except OSError as exc:
            raise WorkflowError("knowledge_read_failed") from exc
        unique(tuple(rule.ref for rule in corpus.rules), "duplicate_rule_ref")
        self.profile = profile
        self.manifest = {
            "cosmos_commit": COMMIT,
            "raw_sha256": dict(HASHES),
            "rule_corpus_sha256": digest(
                canonical(corpus.model_dump(mode="json")).encode()
            ),
            "include_templates": include_templates,
        }
        self._catalog = CosmosCatalog(tuple(records))
        self._candidates: dict[str, Candidate] = {}
        self._terms: dict[str, set[str]] = {}
        self._rules: dict[str, StandardRule] = {}
        self._db = sqlite3.connect(":memory:")
        self._db.execute(
            "CREATE VIRTUAL TABLE candidates USING fts5(ref UNINDEXED, text)"
        )
        try:
            # Resolve immutable metadata once; repeated full-catalog scans are quadratic.
            concepts = {
                record.source_id: record for record in self._catalog.records(kind="bc")
            }
            source_issues = self._catalog.issues()
            if include_templates:
                for record in self._catalog.records(
                    kind="sdtm", sdtmig_version=profile.sdtmig
                ):
                    text = record.value("short_name")
                    bc = concepts.get(record.value("bc_id"))
                    if bc is not None:
                        text += " " + " ".join(
                            row.value(key)
                            for row in bc.rows
                            for key in ("short_name", "synonyms", "categories")
                        )
                    rows = tuple(
                        TemplateRow(
                            variable=row.value("sdtm_variable"),
                            source_locator=f"csv:{row.number}",
                            values_json=canonical(dict(row.fields)),
                        )
                        for row in record.rows
                    )
                    issues = tuple(
                        str(issue)
                        for issue in source_issues
                        if issue.get("source") == record.id
                    )
                    candidate = Candidate(
                        ref=record.id,
                        kind="released_template",
                        title=record.value("short_name"),
                        domain=record.value("domain"),
                        sdtmig=profile.sdtmig,
                        source_url=f"https://github.com/cdisc-org/COSMoS/blob/{COMMIT}/{FILES['sdtm']}",
                        source_sha256=record.source_sha256,
                        source_version=record.package_date,
                        rows=rows,
                        coverage_limits=issues,
                    )
                    self._add(candidate, text)
            for rule in corpus.rules:
                if digest(rule.excerpt.encode()) != rule.excerpt_sha256:
                    raise WorkflowError("rule_hash_mismatch")
                if profile.sdtmig not in rule.sdtmig_versions:
                    continue
                candidate = Candidate(
                    ref=rule.ref,
                    kind="rule_evidence_bundle",
                    title=rule.title,
                    domain=rule.domain,
                    sdtmig=profile.sdtmig,
                    source_url=rule.source_url,
                    source_sha256=rule.excerpt_sha256,
                    source_version=rule.source_version,
                    excerpt=rule.excerpt
                    + "\nAdapter interpretation: "
                    + rule.interpretation,
                    rows=tuple(
                        TemplateRow(
                            variable=variable,
                            source_locator=rule.source_locator,
                            values_json=canonical(
                                {"sdtm_variable": variable, "origin_type": "Collected"}
                            ),
                        )
                        for variable in rule.variables
                    ),
                )
                self._rules[rule.ref] = rule
                self._add(
                    candidate, " ".join((*rule.concept_terms, *rule.property_terms))
                )
        except (ValueError, sqlite3.Error, WorkflowError):
            self._db.close()
            raise

    def _add(self, candidate: Candidate, text: str) -> None:
        """Index one immutable candidate with parameterized values, rejecting collisions."""
        if candidate.ref in self._candidates:
            raise WorkflowError("duplicate_candidate_ref")
        self._candidates[candidate.ref] = candidate
        self._terms[candidate.ref] = tokens(text)
        self._db.execute(
            "INSERT INTO candidates(ref,text) VALUES (?,?)", (candidate.ref, text)
        )

    def close(self) -> None:
        """Release only this index; shared KB files remain read-only."""
        self._db.close()

    def search(self, query: MappingQuery, limit: int = 3) -> tuple[Candidate, ...]:
        """Rank actual SQL candidates by concept, property and scoped modifiers.

        Lexical similarity proposes candidates only; the model still decides
        applicability. Rules additionally require an appropriate property match.
        """
        if not 1 <= limit <= 10:
            raise WorkflowError("invalid_retrieval_limit")
        concept, prop = tokens(query.concept), tokens(query.collected_property)
        modifiers = tokens(" ".join(item.value for item in query.context))
        # A source timepoint is an applicability constraint, not a weak tie-break:
        # a four-minute observation must not receive a one-minute topic template.
        requested_timepoints = timepoints(
            query.concept
            + " "
            + " ".join(
                item.value
                for item in query.context
                if item.name in {"source_label", "timepoint", "section"}
            )
        )
        terms = concept | prop | modifiers
        if not terms:
            return ()
        expression = " OR ".join('"' + word + '"' for word in sorted(terms))
        rows = self._db.execute(
            "SELECT ref, rank FROM candidates WHERE text MATCH ? "
            "ORDER BY rank, ref LIMIT 100",
            (expression,),
        ).fetchall()
        ranked: list[tuple[float, float, str]] = []
        for ref, fts_rank in rows:
            candidate = self._candidates[ref]
            candidate_timepoints = timepoints(candidate.title)
            if (
                requested_timepoints
                and candidate_timepoints
                and not requested_timepoints & candidate_timepoints
            ):
                continue
            words = self._terms[ref]
            overlap = len(concept & words) / max(1, len(concept))
            if overlap < 0.5:
                continue
            rule = self._rules.get(ref)
            if rule and not prop & tokens(" ".join(rule.property_terms)):
                continue
            score = 10 * overlap + 2 * len(prop & words) + len(modifiers & words)
            if tokens(candidate.title) == concept:
                score += 20
            ranked.append((score, fts_rank, ref))
        # Equal semantic scores must retain FTS relevance before the top-k cut.
        # SQLite ranks better matches lower; refs break only exact remaining ties.
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        return tuple(self._candidates[ref] for _, _, ref in ranked[:limit])

    def resolve(self, ref: str) -> Candidate:
        """Read a real candidate, not a guessed target or another snapshot's ID."""
        if ref not in self._candidates:
            raise WorkflowError("candidate_not_found")
        return self._candidates[ref]
