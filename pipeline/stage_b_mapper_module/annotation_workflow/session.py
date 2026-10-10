"""Two-decision orchestration, targeted evidence routing and preserved partial results."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import Field

from ..graphrag.field_contracts import FrozenModel
from .checks import check_decision
from .common import (
    DecisionExplanation,
    Issue,
    Model,
    WorkflowError,
    canonical,
    digest,
    parse,
)
from .decision import MappingDecision
from .evidence_actions import (
    DecisionTurn,
    QuestionLedger,
    ReadMaterial,
    SearchStandard,
    SubmitDecision,
)
from .knowledge import Candidate, Knowledge
from .proposal import Proposal, compile_proposal, proposal_context
from .review import explanation_lines, needs_consistency, render
from .runtime import LocalClient, ModelClient, Phase, prompt, schema_for
from .source import Material, MaterialStore, SourcePacket
from .store import RunStore
from .understanding import (
    CollectionUnderstanding,
    ContextValue,
    MappingQuery,
    Unknown,
    queries,
    validate_understanding,
)
from .understanding_proposal import (
    UnderstandingProposal,
    compile_understanding,
    understanding_context,
)


class Budget(FrozenModel):
    """Declared ceilings, independent of provider defaults and score outcomes."""

    max_calls: int = Field(default=8, ge=2, le=16)
    max_output_tokens: int = Field(default=4800, ge=128, le=4800)
    max_total_tokens: int = Field(default=60000, ge=1000)
    max_seconds: float = Field(default=900, gt=0)
    max_evidence_bytes: int = Field(default=64000, ge=1000)


class Outcome(FrozenModel):
    """A partial/failed run never becomes study-approved because its JSON is valid."""

    status: str
    stop_reason: str
    understanding: CollectionUnderstanding | None
    decision: MappingDecision | None
    issues: tuple[Issue, ...]
    calls: int
    measured_tokens: int
    usage_complete: bool
    consistency_reviewed: bool
    semantic_review: str = "pending"
    study_approval: str = "pending"


def _source_refs(value: object) -> set[str]:
    if not isinstance(value, dict):
        return set()
    refs = value.get("source", [])
    return (
        {ref for ref in refs if isinstance(ref, str)}
        if isinstance(refs, list)
        else set()
    )


def _record_refs(record: dict[str, Any]) -> set[str]:
    return set().union(*(_source_refs(item) for item in _groups(record, "bindings")))


def _revision_scope(refs: set[str], packet: SourcePacket | None) -> set[str]:
    """A changed single-choice parent affects its authorized option branches too.

    This locates revisions only: it does not merge annotations or authorize
    sibling changes when feedback names just one option.
    """
    result = set(refs)
    if packet is not None:
        parents = {
            item.ref
            for item in packet.objects
            if item.ref in refs
            and item.ref in packet.output_scope
            and item.datatype in {"radio", "single_choice"}
        }
        result.update(
            item.ref
            for item in packet.objects
            if item.kind == "option"
            and item.parent_ref in parents
            and item.ref in packet.output_scope
        )
    return result


def _groups(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    entries = value.get(key, [])
    return (
        [item for item in entries if isinstance(item, dict)]
        if isinstance(entries, list)
        else []
    )


def _record_target(proposal: dict[str, Any], position: object) -> object:
    records = proposal.get("records", [])
    if (
        isinstance(position, int)
        and isinstance(records, list)
        and 0 < position <= len(records)
    ):
        record = records[position - 1]
        if isinstance(record, dict):
            # Resolve ordinal references before comparing revisions. The target's
            # own outgoing links are excluded to keep cyclic relations finite.
            return {
                key: value
                for key, value in record.items()
                if key not in {"reason", "links"}
            }
    return {"invalid_record_position": position}


def _business(value: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    result = {key: item for key, item in value.items() if key != "reason"}
    if "links" in result:
        result["links"] = [
            {**link, "record": _record_target(proposal, link.get("record"))}
            for link in _groups(value, "links")
        ]
    if "parent_record" in result:
        result["parent_record"] = _record_target(proposal, result["parent_record"])
    return result


def _freeze_conditions(
    value: dict[str, Any], understanding: CollectionUnderstanding
) -> dict[str, Any]:
    """Persist explicit C handles as predicates before source evidence can reorder them."""
    choices = {
        f"C{index + 1}": item.predicate.model_dump(mode="json")
        for index, item in enumerate(understanding.conditions.entries)
    }

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(entry) for entry in item]
        if isinstance(item, dict):
            return {
                key: (
                    choices[entry]
                    if key == "when" and isinstance(entry, str) and entry in choices
                    else visit(entry)
                )
                for key, entry in item.items()
            }
        return item

    return dict(visit(value))


def _preserve_unaffected(
    previous: dict[str, Any],
    current: dict[str, Any],
    affected: set[str],
    packet: SourcePacket | None = None,
) -> None:
    """Protect actual binding details, including selectors/constants, during repair.

    Adding an omitted field to an existing record is permitted. Splitting,
    removing or changing an unrelated binding or record constraint is not.
    """
    affected = _revision_scope(affected, packet)
    feedback: list[dict[str, object]] = []
    for index, old in enumerate(_groups(previous, "records")):
        protected = [
            item
            for item in _groups(old, "bindings")
            if _source_refs(item) and not _source_refs(item) & affected
        ]
        if not protected:
            continue
        metadata = {
            key: value
            for key, value in _business(old, previous).items()
            if key not in {"bindings", "reason"}
        }
        if any(
            isinstance(new, dict)
            and all(item in new.get("bindings", []) for item in protected)
            and metadata
            == {
                key: value
                for key, value in _business(new, current).items()
                if key not in {"bindings", "reason"}
            }
            for new in _groups(current, "records")
        ):
            continue
        feedback.append(
            {
                "path": f"/records/{index}",
                "code": "unaffected_judgment_changed",
                "message": "Restore these previously supplied bindings and their record constraints. The reported error does not authorize changing them.",
                "affected_refs": sorted(
                    set().union(*(_source_refs(item) for item in protected))
                ),
            }
        )
    for index, old in enumerate(_groups(previous, "other_outcomes")):
        if _source_refs(old) and not _source_refs(old) & affected:
            if not any(
                _business(old, previous) == _business(new, current)
                for new in _groups(current, "other_outcomes")
            ):
                feedback.append(
                    {
                        "path": f"/other_outcomes/{index}",
                        "code": "unaffected_judgment_changed",
                        "message": "Restore this unaffected source disposition and its supporting details.",
                        "affected_refs": sorted(_source_refs(old)),
                    }
                )
    if feedback:
        raise WorkflowError("unaffected_judgment_changed", feedback=tuple(feedback))


def _revision_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
    packet: SourcePacket | None = None,
) -> list[dict[str, object]]:
    """Require an explicit new reason when an accepted business group is revised."""
    changes: list[dict[str, object]] = []
    for section in ("records", "other_outcomes"):
        for old in previous.get(section, []):
            if not isinstance(old, dict):
                continue
            scope = _record_refs(old) if section == "records" else _source_refs(old)
            if any(
                _business(old, previous) == _business(new, current)
                for new in current.get(section, [])
            ):
                continue
            replacements = [
                item
                for item in current.get("records", [])
                + current.get("other_outcomes", [])
                if _revision_scope(scope, packet)
                & _revision_scope(
                    _record_refs(item) if "bindings" in item else _source_refs(item),
                    packet,
                )
            ]
            reasons = [item.get("reason", "") for item in replacements]
            if not replacements or not all(
                reason and reason != old.get("reason") for reason in reasons
            ):
                raise WorkflowError(
                    "revision_reason_required",
                    feedback=(
                        {
                            "path": f"/{section}",
                            "code": "revision_reason_required",
                            "message": "Explain why the new evidence or review changes this previously accepted judgment. Keep every affected source explicitly mapped or otherwise disposed.",
                            "affected_refs": sorted(scope),
                        },
                    ),
                )
            changes.append(
                {"source": sorted(scope), "previous": old, "replacement": replacements}
            )
    for question in previous.get("open_questions", []):
        if any(
            question.get("missing") == other.get("missing")
            and set(question.get("affected_refs", []))
            == set(other.get("affected_refs", []))
            for other in current.get("open_questions", [])
        ):
            continue
        scope = set(question.get("affected_refs", []))
        old_reasons = {
            item.get("reason")
            for item in previous.get("records", []) + previous.get("other_outcomes", [])
            if scope
            & (_record_refs(item) if "bindings" in item else _source_refs(item))
        }
        replacements = [
            item
            for item in current.get("records", []) + current.get("other_outcomes", [])
            if scope
            & (_record_refs(item) if "bindings" in item else _source_refs(item))
        ]
        if not replacements or all(
            item.get("reason") in old_reasons for item in replacements
        ):
            raise WorkflowError(
                "evidence_resolution_reason_required",
                feedback=(
                    {
                        "path": "/open_questions",
                        "code": "evidence_resolution_reason_required",
                        "message": "Explain in the affected judgment why this evidence question can now be closed, including how an unavailable result limits the decision. Do not silently drop an unanswered question.",
                        "affected_refs": sorted(scope),
                    },
                ),
            )
        changes.append(
            {
                "source": sorted(scope),
                "closed_question": question,
                "replacement": replacements,
            }
        )
    return changes


def _understanding_refs(section: str, value: dict[str, Any]) -> set[str]:
    """Identify the judgment's subjects, separately from its supporting evidence."""
    if section == "items":
        source = value.get("source_ref")
        return {source} if isinstance(source, str) else set()
    if section == "bindings":
        refs = value.get("target_source_refs", [])
        source = value.get("source_ref")
        return (
            {ref for ref in refs if isinstance(ref, str)}
            if isinstance(refs, list)
            else set()
        ) | ({source} if isinstance(source, str) else set())
    refs = value.get("affected_refs", [])
    return (
        {ref for ref in refs if isinstance(ref, str)}
        if isinstance(refs, list)
        else set()
    )


def _understanding_business(value: Any) -> Any:
    """Compare semantic content while allowing wording and omitted optional defaults."""
    if isinstance(value, list):
        return [_understanding_business(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: (
            sorted(item)
            if key in {"source_refs", "target_source_refs", "affected_refs"}
            and isinstance(item, list)
            and all(isinstance(ref, str) for ref in item)
            else _understanding_business(item)
        )
        for key, item in value.items()
        if key != "reason"
        and not (key == "concept_context" and item == [])
        and not (key in {"source_context_ref", "material_ref"} and item is None)
    }


def _preserve_understanding_unaffected(
    previous: dict[str, Any],
    current: dict[str, Any],
    affected: set[str],
    *,
    condition_status_affected: bool = False,
) -> None:
    """Keep unrelated source judgments during a correction with a known scope."""
    feedback: list[dict[str, object]] = []
    for section in ("items", "bindings", "material_unknowns", "conditions"):
        old_parent = (
            previous.get("conditions", {}) if section == "conditions" else previous
        )
        new_parent = (
            current.get("conditions", {}) if section == "conditions" else current
        )
        if not isinstance(old_parent, dict) or not isinstance(new_parent, dict):
            continue
        key = "entries" if section == "conditions" else section
        for index, old in enumerate(_groups(old_parent, key)):
            scope = _understanding_refs(section, old)
            if not scope or scope & affected:
                continue
            if any(
                _understanding_business(old) == _understanding_business(new)
                for new in _groups(new_parent, key)
            ):
                continue
            feedback.append(
                {
                    "path": (
                        f"/conditions/entries/{index}"
                        if section == "conditions"
                        else f"/{section}/{index}"
                    ),
                    "code": "unaffected_understanding_changed",
                    "message": "Restore this unaffected source interpretation and its supporting details. Repairing another item does not authorize removing or changing it.",
                    "affected_refs": sorted(scope),
                }
            )
        if (
            section == "conditions"
            and not condition_status_affected
            and old_parent.get("status") != new_parent.get("status")
        ):
            entries = _groups(old_parent, "entries") + _groups(new_parent, "entries")
            if not any(
                _understanding_refs(section, item) & affected for item in entries
            ):
                feedback.append(
                    {
                        "path": "/conditions/status",
                        "code": "unaffected_understanding_changed",
                        "message": "Preserve the unaffected source condition status; do not turn unknown conditions into an assertion that no conditions exist.",
                        "affected_refs": [],
                    }
                )
    if feedback:
        raise WorkflowError(
            "unaffected_understanding_changed", feedback=tuple(feedback)
        )


def _understanding_closed_questions(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    """Require an explicit model rationale before an accepted evidence question closes."""
    previous_summary = previous.get("explanation", {}).get("summary")
    summary = current.get("explanation", {}).get("summary")
    for index, question in enumerate(_groups(previous, "material_unknowns")):
        if any(
            _understanding_business(question) == _understanding_business(item)
            for item in _groups(current, "material_unknowns")
        ):
            continue
        scope = _understanding_refs("material_unknowns", question)
        old_reasons = {
            item.get("basis", {}).get("reason")
            for item in _groups(previous, "items")
            if _understanding_refs("items", item) & scope
        }
        revised_reason = any(
            item.get("basis", {}).get("reason")
            and item.get("basis", {}).get("reason") not in old_reasons
            for item in _groups(current, "items")
            if _understanding_refs("items", item) & scope
        )
        if revised_reason or (summary and summary != previous_summary):
            continue
        raise WorkflowError(
            "understanding_evidence_resolution_reason_required",
            feedback=(
                {
                    "path": f"/material_unknowns/{index}",
                    "code": "understanding_evidence_resolution_reason_required",
                    "message": "Explain why this previously accepted evidence question can now close, in the affected item's basis.reason or explanation.summary. State how any unavailable lookup limits the interpretation; do not silently erase an unanswered question.",
                    "affected_refs": sorted(scope),
                },
            ),
        )


class Session:
    """Run a task serially; each formal response is checked before atomic publication."""

    def __init__(
        self,
        packet: SourcePacket,
        knowledge: Knowledge,
        model: ModelClient,
        output: Path,
        materials: MaterialStore | None = None,
        budget: Budget | None = None,
    ) -> None:
        """Bind one immutable study/profile to an exclusive new run directory."""
        if packet.standard != knowledge.profile:
            raise WorkflowError("session_profile_mismatch")
        self.packet, self.knowledge, self.model = packet, knowledge, model
        self.store = RunStore(output)
        self.materials = materials or MaterialStore()
        self.budget = budget or Budget()
        self.calls, self.tokens = 0, 0
        self.usage_complete = True
        self.started = time.monotonic()
        self.understanding: CollectionUnderstanding | None = None
        self.understanding_proposal: UnderstandingProposal | None = None
        self.decision: MappingDecision | None = None
        self.proposal: Proposal | None = None
        self.delivered: tuple[Candidate, ...] = ()
        self.documents: tuple[Material, ...] = ()
        self.checked_requests: set[str] = set()
        self.consistency_reviewed = False
        self.issues: tuple[Issue, ...] = ()
        self.source_changed = False
        self.lookup_results: list[dict[str, object]] = []
        self.questions = QuestionLedger()
        self.evidence_action_counts: dict[str, int] = {}
        self.repair_draft: dict[str, Any] | None = None
        self.repair_scope: set[str] = set()
        self.lookup_affected: set[str] = set()
        self.consistency_contexts: set[str] = set()
        self.decision_notes: list[dict[str, object]] = []
        self.note_lines = [
            "# Decision explanations",
            "",
            "Brief evidence-based explanations; not private model reasoning. Backend actions are labelled separately.",
            "",
        ]

    def _note(
        self,
        stage: str,
        author: str,
        explanation: DecisionExplanation,
        details: tuple[str, ...] = (),
    ) -> None:
        """Persist each completed stage explanation before the next model request."""
        entry = {
            "stage": stage,
            "author": author,
            "attempt": self.calls,
            "revision": self.store.revision,
            "explanation": explanation.model_dump(mode="json"),
            "details": details,
        }
        self.decision_notes.append(entry)
        self.note_lines.extend(
            [
                f"## {len(self.decision_notes)}. {stage} ({author}, attempt {self.calls}, revision {self.store.revision})",
                "",
                *explanation_lines(explanation),
                *("- " + detail for detail in details),
                "",
            ]
        )
        self.store.write("decision-notes.json", self.decision_notes)
        self.store.write_text("decision-notes.md", "\n".join(self.note_lines))

    def _structured(
        self, phase: Phase, payload: Mapping[str, object], model: type[Model]
    ) -> Model:
        """Allow one schema-only repair; semantic check failures are never retried here."""
        text = self._generate(phase, payload)
        try:
            return parse(model, text)
        except WorkflowError as exc:
            if exc.code != "invalid_structured_output":
                raise
            feedback = exc.feedback
        self._note(
            "format_repair",
            "backend",
            DecisionExplanation(
                summary="The formal response failed the output contract. Requesting the single allowed format repair; the invalid reply remains in the trace.",
                evidence_refs=(),
                uncertainties=(
                    "The response has not been accepted as a valid decision.",
                ),
            ),
        )
        repair = {
            **payload,
            "format_repair": {
                "invalid_formal_response": text,
                "errors": feedback,
                "instruction": "Correct schema/JSON only, preserve all facts. This is the sole format repair.",
            },
        }
        return parse(model, self._generate(phase, repair))

    def _generate(self, phase: Phase, payload: object) -> str:
        """Persist request identity before sending and diagnostic metadata before parsing."""
        if (self.store.path / "STOP").exists():
            raise WorkflowError("operator_stop")
        if (
            self.calls >= self.budget.max_calls
            or self.tokens >= self.budget.max_total_tokens
            or time.monotonic() - self.started >= self.budget.max_seconds
        ):
            raise WorkflowError("budget_exhausted")
        messages, schema = prompt(phase, payload), schema_for(phase, payload)
        index = self.calls + 1
        request: dict[str, object] = {
            "phase": phase,
            "messages": messages,
            "schema": schema,
            "max_output_tokens": self.budget.max_output_tokens,
        }
        request_timeout: float | None = None
        if isinstance(self.model, LocalClient):
            # A slow local model may use the larger request cap only while the
            # existing whole-run budget still has time remaining.
            request_timeout = self.model.bounded_timeout(
                self.budget.max_seconds - (time.monotonic() - self.started)
            )
            request["transport"] = self.model.request_body(
                messages, schema, self.budget.max_output_tokens
            )
            request["base_url"] = self.model.base_url
            request["timeout_seconds"] = request_timeout
        self.store.write(f"{index:02d}.request.json", request)
        self.store.write(
            "inflight.json",
            {
                "attempt": index,
                "status": "pending",
                "request_sha256": digest(canonical(request).encode()),
            },
        )
        self.calls += 1
        if isinstance(self.model, LocalClient):
            reply = self.model.generate(
                messages,
                schema,
                self.budget.max_output_tokens,
                timeout_seconds=request_timeout,
            )
        else:
            reply = self.model.generate(messages, schema, self.budget.max_output_tokens)
        if reply.prompt_tokens is not None and reply.completion_tokens is not None:
            self.tokens += reply.prompt_tokens + reply.completion_tokens
        else:
            self.usage_complete = False
        self.store.write(f"{index:02d}.reply.json", reply.model_dump(mode="json"))
        self.store.write(
            "inflight.json",
            {
                "attempt": index,
                "status": "returned",
                "measured_tokens": self.tokens,
                "usage_complete": self.usage_complete,
            },
        )
        if reply.error:
            raise WorkflowError(reply.error)
        if not self.usage_complete:
            raise WorkflowError("usage_missing")
        if not reply.content:
            raise WorkflowError("response_content_empty")
        return reply.content

    def _read(
        self,
        requests: tuple[Unknown, ...],
        *,
        stop_on_repeat: bool = True,
        allow_cached: bool = False,
        query_override: MappingQuery | None = None,
    ) -> bool:
        """Return every attempted lookup to the next turn, including negative results."""
        processed = False
        self.lookup_affected = set()
        for request in requests:
            key = digest(
                canonical(
                    {
                        "kind": request.source_kind,
                        "material": request.material_ref,
                        "scope": sorted(request.affected_refs),
                        "query": request.missing if not request.material_ref else None,
                        "structured_query": (
                            query_override.model_dump(mode="json")
                            if query_override is not None
                            else None
                        ),
                    }
                ).encode()
            )
            if key in self.checked_requests:
                if allow_cached:
                    original = next(
                        item
                        for item in reversed(self.lookup_results)
                        if item.get("request_key") == key
                    )
                    cached_result = {
                        **original,
                        "status": (
                            "unavailable"
                            if original["status"] == "unavailable"
                            else "no_new_evidence"
                        ),
                        "cached": True,
                    }
                    self.lookup_results.append(cached_result)
                    self.lookup_affected.update(request.affected_refs)
                    self.store.write(
                        f"material-{len(self.lookup_results):03d}.json", cached_result
                    )
                    processed = True
                continue
            self.checked_requests.add(key)
            processed = True
            self.lookup_affected.update(request.affected_refs)
            added = False
            document = None
            added_candidates: tuple[Candidate, ...] = ()
            executed_queries: list[MappingQuery] = []
            if request.source_kind == "standard" and self.understanding is not None:
                # Only real pinned evidence is admitted; a failed lookup remains visible.
                if request.material_ref:
                    try:
                        added_candidates = (
                            self.knowledge.resolve(request.material_ref),
                        )
                    except WorkflowError as exc:
                        if exc.code != "candidate_not_found":
                            raise
                        added_candidates = ()
                else:
                    found: dict[str, Candidate] = {}
                    for query in (
                        (query_override,)
                        if query_override is not None
                        else queries(self.understanding, self.packet)
                    ):
                        if set(query.source_refs) & set(request.affected_refs):
                            contextual = query.model_copy(
                                update={
                                    "context": query.context
                                    + (
                                        ContextValue(
                                            name="evidence_question",
                                            value=request.missing,
                                            source_refs=request.affected_refs,
                                        ),
                                    )
                                }
                            )
                            executed_queries.append(contextual)
                            for candidate in self.knowledge.search(contextual):
                                found[candidate.ref] = candidate
                    added_candidates = tuple(found.values())
                new = tuple(
                    candidate
                    for candidate in added_candidates
                    if candidate.ref not in {item.ref for item in self.delivered}
                )
                if new:
                    self.delivered += new
                    if (
                        len(
                            canonical(
                                [candidate.wire() for candidate in self.delivered]
                            ).encode()
                        )
                        > self.budget.max_evidence_bytes
                    ):
                        raise WorkflowError("evidence_capacity_exceeded")
                    added = True
            if request.material_ref and request.source_kind != "standard":
                document = self.materials.read(
                    self.packet.study_id, request.material_ref, request.source_kind
                )
            if document is not None and document.ref not in {
                item.ref for item in self.documents
            }:
                self.documents += (document,)
                added = True
                if document.kind in {"source", "dependency"}:
                    self.source_changed = True
            status = (
                "available"
                if added
                else (
                    "no_new_evidence" if document or added_candidates else "unavailable"
                )
            )
            result: dict[str, object] = {
                "request_key": key,
                "request": request.model_dump(mode="json"),
                "status": status,
                "cached": False,
                "queries": [
                    query.model_dump(mode="json") for query in executed_queries
                ],
                "document": document.model_dump(mode="json") if document else None,
                "candidates": [candidate.wire() for candidate in added_candidates],
            }
            self.lookup_results.append(result)
            self.store.write(f"material-{len(self.lookup_results):03d}.json", result)
            self._note(
                "evidence_lookup",
                "backend",
                DecisionExplanation(
                    summary=f"Processed the model's {request.source_kind} evidence request. Result: "
                    + status
                    + ". Full lookup evidence is saved in the material artifact.",
                    evidence_refs=(document.ref,) if document else (),
                    uncertainties=(
                        ()
                        if document or added_candidates
                        else (
                            "The requested evidence is unavailable in admitted materials.",
                        )
                    ),
                ),
                (f"Requested: {request.missing}", f"Decision impact: {request.impact}"),
            )
        if requests and not processed and stop_on_repeat:
            raise WorkflowError("evidence_no_progress")
        return processed

    def _lookup_context(self) -> list[dict[str, object]]:
        """Reference each delivered body once per prompt; full receipts stay on disk."""
        output = []
        for entry in self.lookup_results:
            document = entry.get("document")
            candidates = entry.get("candidates", [])
            output.append(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"document", "candidates"}
                }
                | {
                    "document_ref": (
                        document.get("ref") if isinstance(document, dict) else None
                    ),
                    "candidate_refs": (
                        [item["ref"] for item in candidates if isinstance(item, dict)]
                        if isinstance(candidates, list)
                        else []
                    ),
                }
            )
        return output

    def _save_questions(self) -> None:
        self.store.write(
            "question-ledger.json",
            [question.model_dump(mode="json") for question in self.questions.records],
        )

    def _perform_action(self, action: ReadMaterial | SearchStandard) -> None:
        """Run an authorized evidence action without committing an annotation."""
        assert self.understanding is not None
        if not set(action.source) <= set(self.packet.output_scope):
            raise WorkflowError(
                "evidence_source_scope_invalid",
                feedback=(
                    {
                        "path": "/action/source",
                        "code": "evidence_source_scope_invalid",
                        "message": "Choose affected sources from output_sources; supporting context is read-only.",
                        "affected_refs": list(action.source),
                        "allowed": list(self.packet.output_scope),
                    },
                ),
            )
        query = None
        if isinstance(action, ReadMaterial):
            material_index = {
                item["ref"]: item["kind"]
                for item in self.materials.index(self.packet.study_id)
            }
            context = proposal_context(
                self.packet,
                self.understanding,
                self.delivered,
                materials=self.documents,
                reserved_refs=tuple(material_index),
            )
            template_choices = context["templates"]
            assert isinstance(template_choices, list)
            aliases = {
                entry["alias"]: candidate.ref
                for entry, candidate in zip(template_choices, self.delivered)
            }
            if action.handle in material_index:
                kind, ref = material_index[action.handle], action.handle
            elif action.handle in aliases:
                kind, ref = "standard", aliases[action.handle]
            else:
                raise WorkflowError(
                    "action_material_unknown",
                    feedback=(
                        {
                            "path": "/action/handle",
                            "code": "action_material_unknown",
                            "message": "Read an indexed material or delivered template handle; use search_standard to search for missing standard knowledge.",
                            "affected_refs": list(action.source),
                            "allowed": list(material_index) + list(aliases),
                        },
                    ),
                )
            request = Unknown.model_validate(
                {
                    "missing": action.question,
                    "affected_refs": action.source,
                    "impact": action.impact,
                    "source_kind": kind,
                    "material_ref": ref,
                }
            )
        else:
            context_values: list[ContextValue] = []
            assert self.understanding is not None
            for item in self.understanding.items:
                if set(item.source_refs) & set(action.source):
                    context_values.extend(item.concept_context)
            query = MappingQuery(
                item_ref="evidence-search",
                source_refs=action.source,
                concept=action.concept,
                collected_property=action.collected_property,
                context=tuple(context_values),
                conditions=(),
            )
            request = Unknown(
                missing=action.question,
                affected_refs=action.source,
                impact=action.impact,
                source_kind="standard",
                material_ref=None,
            )
        signature = canonical(
            {
                key: value
                for key, value in action.model_dump(mode="json").items()
                if key not in {"question", "impact"}
            }
        )
        if self.evidence_action_counts.get(signature, 0) >= 2:
            raise WorkflowError("evidence_no_progress")
        handle = self.questions.observe(action, source_scope=self.packet.output_scope)
        self._save_questions()
        before = len(self.lookup_results)
        self._read((request,), allow_cached=True, query_override=query)
        self.evidence_action_counts[signature] = (
            self.evidence_action_counts.get(signature, 0) + 1
        )
        self.store.write(
            f"{self.calls:02d}.action.json",
            {
                "action": action.model_dump(mode="json"),
                "question": handle,
                "executed": True,
                "lookups": [
                    f"material-{index:03d}.json"
                    for index in range(before + 1, len(self.lookup_results) + 1)
                ],
                "statuses": [entry["status"] for entry in self.lookup_results[before:]],
            },
        )

    def _complete_decision(
        self, phase: Phase = "decision", affected: set[str] | None = None
    ) -> None:
        """Interleave evidence and source refreshes until a valid submission arrives."""
        while not self._decide(phase, affected):
            affected = set(affected or ()) | self.lookup_affected
            if self.source_changed:
                self._understand()
                assert self.understanding is not None
                while self._read(
                    self.understanding.material_unknowns, stop_on_repeat=False
                ):
                    affected.update(self.lookup_affected)
                    self._understand()
                self._retrieve()

    def _understand(self) -> None:
        """Compile source judgments and request at most two scoped corrections."""
        previous = (
            self.understanding_proposal.model_dump(mode="json")
            if self.understanding_proposal is not None
            else None
        )
        revision_scope = set(self.lookup_affected) if previous is not None else set()
        payload: dict[str, object] = {
            "source": self.packet.model_dump(mode="json"),
            "understanding_context": understanding_context(self.packet),
            "material_index": self.materials.index(self.packet.study_id),
            "materials": [item.model_dump(mode="json") for item in self.documents],
            "lookup_results": self._lookup_context(),
            "previous_understanding_proposal": previous,
            "revision_scope": sorted(revision_scope),
            "evidence_followup_instruction": "For an evidence follow-up, preserve interpretations outside revision_scope. Explain a closed evidence question in the affected item's basis.reason or explanation.summary, including the limitation of an unavailable lookup.",
        }
        preserved: dict[str, Any] | None = None
        correction_scope: set[str] = set()
        condition_status_affected = False
        fingerprints: set[str] = set()
        for attempt in range(3):
            # Transport, operator STOP and the shared budget must never become
            # semantic correction requests. Every reply is saved by _generate.
            text = self._generate("understanding", payload)
            raw: dict[str, Any] | None = None
            try:
                decoded = json.loads(text)
                canonical(decoded)
                if isinstance(decoded, dict):
                    raw = decoded
            except ValueError:
                pass  # Strict parsing below supplies actionable format feedback.
            if raw is not None:
                self.store.write(f"{self.calls:02d}.understanding-proposal.json", raw)
            try:
                proposal = parse(UnderstandingProposal, text)
                draft = proposal.model_dump(mode="json")
                if preserved is not None:
                    _preserve_understanding_unaffected(
                        preserved,
                        draft,
                        correction_scope,
                        condition_status_affected=condition_status_affected,
                    )
                if previous is not None and revision_scope:
                    _preserve_understanding_unaffected(previous, draft, revision_scope)
                value = compile_understanding(
                    proposal,
                    self.packet,
                    delivered_material_refs=tuple(item.ref for item in self.documents),
                )
                validate_understanding(value, self.packet)
                allowed = {item.ref for item in self.packet.objects} | {
                    item.ref for item in self.documents
                }
                if not set(value.explanation.evidence_refs) <= allowed:
                    raise WorkflowError(
                        "explanation_evidence_not_delivered",
                        feedback=(
                            {
                                "path": "/explanation/evidence_refs",
                                "code": "explanation_evidence_not_delivered",
                                "message": "Cite only source objects or materials delivered in this request. These references support the explanation; they do not add interpreted subjects.",
                                "affected_refs": [],
                                "allowed_refs": sorted(allowed),
                            },
                        ),
                    )
                if previous is not None and revision_scope:
                    _understanding_closed_questions(previous, draft)
                break
            except WorkflowError as exc:
                feedback = self._understanding_feedback(exc, raw)
                self.store.write(
                    f"{self.calls:02d}.understanding-validation.json",
                    {"code": exc.code, "feedback": feedback, "accepted": False},
                )
                fingerprint = digest(
                    canonical(
                        {"reply": text, "code": exc.code, "feedback": feedback}
                    ).encode()
                )
                if fingerprint in fingerprints:
                    raise WorkflowError(
                        "understanding_correction_no_progress", feedback=tuple(feedback)
                    ) from exc
                fingerprints.add(fingerprint)
                if attempt == 2:
                    raise WorkflowError(
                        "understanding_correction_exhausted", feedback=tuple(feedback)
                    ) from exc
                known_scope = all(item.get("scope_known") for item in feedback)
                if preserved is None and raw is not None and known_scope:
                    if (
                        exc.code == "unaffected_understanding_changed"
                        and previous is not None
                        and revision_scope
                    ):
                        preserved = previous
                        correction_scope.update(revision_scope)
                    else:
                        preserved = raw
                for item in feedback:
                    if item.get("code") == "unaffected_understanding_changed":
                        continue
                    refs = item.get("affected_refs", [])
                    if isinstance(refs, list):
                        correction_scope.update(
                            ref for ref in refs if isinstance(ref, str)
                        )
                    condition_status_affected |= (
                        item.get("path") == "/conditions/status"
                        or item.get("code") == "condition_status_conflict"
                    )
                payload["correction"] = {
                    "attempt": attempt + 1,
                    "remaining_attempts": 2 - attempt,
                    "previous_draft": raw if raw is not None else text,
                    "preserved_draft": preserved,
                    "errors": feedback,
                    "instruction": "Correct the reported interpretation only. Keep unaffected subjects, bindings, conditions and evidence questions. The item source_ref identifies one interpreted object; basis.source_refs and concept_context.source_refs identify supporting evidence. Entry mode and datatype are supplied source facts, not model decisions. Use the supplied input before asking for missing material; keep genuine uncertainty explicit.",
                }
                self._note(
                    "understanding_correction",
                    "backend",
                    DecisionExplanation(
                        summary=f"The source interpretation was not accepted ({exc.code}). Returning concrete validation feedback for bounded correction.",
                        evidence_refs=(),
                        uncertainties=(
                            "The invalid draft is preserved separately from accepted interpretations.",
                        ),
                    ),
                )
        self.store.commit("understanding", value.model_dump(mode="json"))
        self.store.write(
            f"{self.calls:02d}.understanding-validation.json",
            {"accepted": True, "code": None, "feedback": []},
        )
        self.store.write(
            f"{self.calls:02d}.accepted-understanding.json",
            value.model_dump(mode="json"),
        )
        self.store.write("accepted-understanding-proposal.json", raw)
        self.understanding_proposal = proposal
        self.understanding = value
        self.source_changed = False
        details = tuple(
            f"Item [{', '.join(item.source_refs)}]: {item.basis.reason}"
            for item in value.items
        )
        details += tuple(
            f"Binding {binding.source_ref} {binding.relation} [{', '.join(binding.target_source_refs)}]: {binding.basis.reason}"
            for binding in value.bindings
        )
        details += tuple(
            f"Condition [{', '.join(condition.affected_refs)}]: {condition.basis.reason}"
            for condition in value.conditions.entries
        )
        details += tuple(
            f"Needs {gap.source_kind}: {gap.missing}; impact: {gap.impact}"
            for gap in value.material_unknowns
        )
        self._note("understanding", "model", value.explanation, details)

    def _understanding_feedback(
        self, error: WorkflowError, draft: dict[str, Any] | None
    ) -> list[dict[str, object]]:
        """Attach source scope to schema paths without mistaking evidence for subjects."""
        feedback = [dict(item) for item in error.feedback] or [
            {
                "path": "/",
                "code": error.code,
                "message": "Correct the supplied interpretation against the source subjects and visible evidence. Keep genuine uncertainty explicit; do not invent source facts.",
            }
        ]
        for item in feedback:
            refs = item.get("affected_refs", [])
            scope = (
                {ref for ref in refs if isinstance(ref, str)}
                if isinstance(refs, list)
                else set()
            )
            parts = str(item.get("path", "/")).strip("/").split("/")
            known = (
                bool(scope)
                or parts[0]
                in {
                    "items",
                    "bindings",
                    "conditions",
                    "material_unknowns",
                    "explanation",
                }
                or (item.get("code") == "extra_forbidden" and bool(parts[0]))
            )
            if draft and len(parts) >= 2:
                section = parts[0]
                parent: object = draft
                key, index_part = section, parts[1]
                if (
                    section == "conditions"
                    and len(parts) >= 3
                    and parts[1] == "entries"
                ):
                    parent, key, index_part = (
                        draft.get("conditions", {}),
                        "entries",
                        parts[2],
                    )
                if isinstance(parent, dict) and index_part.isdigit():
                    groups = parent.get(key, [])
                    index = int(index_part)
                    if (
                        isinstance(groups, list)
                        and index < len(groups)
                        and isinstance(groups[index], dict)
                    ):
                        scope.update(_understanding_refs(section, groups[index]))
                        known |= bool(scope)
                if section == "conditions" and parts[1] == "status":
                    known = True
            item["affected_refs"] = sorted(scope)
            item["scope_known"] = known
        return feedback

    def _retrieve(self) -> None:
        """Consume the saved interpretation and expose full actual evidence in one turn."""
        assert self.understanding is not None
        requests = queries(self.understanding, self.packet)
        # Existing alias positions remain fixed after source reinterpretation.
        candidates: dict[str, Candidate] = {item.ref: item for item in self.delivered}
        for request in requests:
            for candidate in self.knowledge.search(request):
                candidates[candidate.ref] = candidate
        self.delivered = tuple(candidates.values())
        values = [candidate.wire() for candidate in self.delivered]
        if len(canonical(values).encode()) > self.budget.max_evidence_bytes:
            raise WorkflowError("evidence_capacity_exceeded")
        self.store.write(
            "retrieval.json",
            {
                "queries": [query.model_dump(mode="json") for query in requests],
                "candidates": values,
            },
        )
        self._note(
            "retrieval",
            "backend",
            DecisionExplanation(
                summary=f"Queried the pinned knowledge snapshot using {len(requests)} interpreted source items and delivered {len(self.delivered)} complete candidates. This is a retrieval action, not a mapping decision.",
                evidence_refs=tuple(candidate.ref for candidate in self.delivered)[:24],
                uncertainties=(
                    () if self.delivered else ("No candidate was retrieved.",)
                ),
            ),
            tuple(
                f"{query.item_ref}: {query.concept}; property: {query.collected_property}"
                for query in requests
            ),
        )

    def _decide(
        self, phase: Phase = "decision", affected: set[str] | None = None
    ) -> bool:
        """Compile a compact judgment and return bounded, actionable corrections."""
        assert self.understanding is not None
        previous = (
            self.proposal.model_dump(mode="json", exclude_defaults=True)
            if self.proposal
            else None
        )
        reserved_refs = tuple(
            item["ref"] for item in self.materials.index(self.packet.study_id)
        )
        decision_context = proposal_context(
            self.packet,
            self.understanding,
            self.delivered,
            materials=self.documents,
            reserved_refs=reserved_refs,
        )
        template_choices = decision_context["templates"]
        assert isinstance(template_choices, list)
        question_scope = _revision_scope(
            set(affected if affected is not None else self.packet.output_scope),
            self.packet,
        )
        # A decision view separates write subjects from supporting neighbors.
        # Keep every original object and provenance; the full SourcePacket is
        # retained by the compiler, understanding stage and source.json artifact.
        source_view = self.packet.model_dump(mode="json")
        readonly_context = [
            item
            for item in source_view["objects"]
            if item["ref"] not in self.packet.output_scope
        ]
        source_view["objects"] = [
            item
            for item in source_view["objects"]
            if item["ref"] in self.packet.output_scope
        ]
        payload: dict[str, object] = {
            "source": source_view,
            "readonly_context": readonly_context,
            "understanding": self.understanding.model_dump(mode="json"),
            "decision_context": decision_context,
            "materials": [item.model_dump(mode="json") for item in self.documents],
            "material_index": self.materials.index(self.packet.study_id),
            "previous_proposal": previous,
            "lookup_results": self._lookup_context(),
            "questions": [
                item.model_dump(mode="json") for item in self.questions.records
            ],
            "checks": [item.model_dump(mode="json") for item in self.issues],
            "revision_scope": sorted(question_scope),
        }
        first_draft = self.repair_draft
        correction_scope = self.repair_scope | set(affected or ())
        if first_draft is not None:
            payload["preserved_draft"] = first_draft
        fingerprints: set[str] = set()
        for attempt in range(3):
            text = self._generate(phase, payload)
            raw: dict[str, Any] | None = None
            try:
                decoded = json.loads(text)
                canonical(decoded)
                if isinstance(decoded, dict):
                    raw = decoded.get("action", decoded)
                    if not isinstance(raw, dict):
                        raw = None
            except ValueError:
                pass  # The strict parser below supplies the correction details.
            try:
                action = parse(DecisionTurn, text).action
                if isinstance(action, (ReadMaterial, SearchStandard)):
                    self._perform_action(action)
                    return False
                proposal = Proposal(
                    records=action.records,
                    other_outcomes=action.other_outcomes,
                    open_questions=(),
                )
                evidence = (
                    {item.ref for item in self.packet.objects}
                    | {item.ref for item in self.documents}
                    | {item.ref for item in self.delivered}
                    | {item["alias"] for item in template_choices}
                )
                self.questions.validate_updates(
                    action.question_updates,
                    visible_evidence=evidence,
                    source_scope=question_scope,
                )
                updated = {
                    item.question: item.status for item in action.question_updates
                }
                missing_questions = [
                    item
                    for item in self.questions.records
                    if item.status == "open"
                    and set(item.source) <= question_scope
                    and item.handle not in updated
                ]
                if missing_questions:
                    raise WorkflowError(
                        "question_update_required",
                        feedback=tuple(
                            {
                                "path": "/action/question_updates",
                                "code": "question_update_required",
                                "message": "State whether this checked question is resolved or deferred, with a reason. Deferred preserves the missing evidence without repeating a lookup.",
                                "affected_refs": list(item.source),
                                "allowed": [item.handle],
                            }
                            for item in missing_questions
                        ),
                    )
                draft = _freeze_conditions(
                    proposal.model_dump(mode="json", exclude_defaults=True),
                    self.understanding,
                )
                self.store.write(f"{self.calls:02d}.proposal.json", draft)
                if first_draft is not None:
                    _preserve_unaffected(
                        first_draft, draft, correction_scope, self.packet
                    )
                if previous is not None:
                    _preserve_unaffected(
                        previous,
                        draft,
                        set(
                            affected
                            if affected is not None
                            else self.packet.output_scope
                        ),
                        self.packet,
                    )
                value = compile_proposal(
                    proposal,
                    self.packet,
                    self.understanding,
                    self.delivered,
                    materials=self.documents,
                    reserved_refs=reserved_refs,
                )
                issues = check_decision(
                    value,
                    self.packet,
                    self.understanding,
                    self.delivered,
                    self.documents,
                )
                missing = [item for item in issues if item.code == "source_not_covered"]
                if missing:
                    raise WorkflowError(
                        "source_not_covered",
                        feedback=tuple(
                            {
                                "path": "/",
                                "code": item.code,
                                "message": "This authorized source has no disposition. Add a binding or an explicit context_only, not_submitted, or unresolved outcome with a reason; preserve the other judgments.",
                                "affected_refs": list(item.refs),
                            }
                            for item in missing
                        ),
                    )
                changes = (
                    _revision_changes(previous, draft, self.packet) if previous else []
                )
                issues += tuple(
                    Issue(
                        code="evidence_question_deferred",
                        refs=(*item.source, item.handle),
                    )
                    for item in self.questions.records
                    if updated.get(item.handle, item.status) == "deferred"
                )
                break
            except WorkflowError as exc:
                if exc.code in {
                    "evidence_no_progress",
                    "evidence_capacity_exceeded",
                    "artifact_write_failed",
                }:
                    raise
                feedback = self._correction_feedback(exc, raw)
                self.store.write(
                    f"{self.calls:02d}.validation.json",
                    {"code": exc.code, "feedback": feedback, "accepted": False},
                )
                fingerprint = digest(
                    canonical(
                        {"reply": text, "code": exc.code, "feedback": feedback}
                    ).encode()
                )
                if fingerprint in fingerprints:
                    raise WorkflowError(
                        "correction_no_progress", feedback=tuple(feedback)
                    ) from exc
                fingerprints.add(fingerprint)
                if attempt == 2:
                    raise WorkflowError(
                        "correction_exhausted", feedback=tuple(feedback)
                    ) from exc
                if (
                    first_draft is None
                    and raw is not None
                    and (raw.get("records") or raw.get("other_outcomes"))
                ):
                    if (
                        exc.code == "unaffected_judgment_changed"
                        and previous is not None
                    ):
                        first_draft = previous
                        correction_scope.update(
                            affected
                            if affected is not None
                            else self.packet.output_scope
                        )
                    else:
                        first_draft = _freeze_conditions(raw, self.understanding)
                for item in feedback:
                    if item.get("code") != "unaffected_judgment_changed":
                        refs = item.get("affected_refs", [])
                        if isinstance(refs, list):
                            correction_scope.update(
                                ref for ref in refs if isinstance(ref, str)
                            )
                if first_draft is not None and (
                    first_draft.get("records") or first_draft.get("other_outcomes")
                ):
                    self.repair_draft = first_draft
                    self.repair_scope = set(correction_scope)
                payload["correction"] = {
                    "attempt": attempt + 1,
                    "remaining_attempts": 2 - attempt,
                    "previous_draft": raw if raw is not None else text,
                    "preserved_draft": first_draft,
                    "errors": feedback,
                    "instruction": "Correct the reported judgments only. Preserve unrelated bindings, grouping, conditions, selectors and constants. Do not erase facts to pass validation; give an explicit reason for an unresolved or changed disposition.",
                }
                self._note(
                    "decision_correction",
                    "backend",
                    DecisionExplanation(
                        summary=f"The compact proposal was not accepted ({exc.code}). Returning concrete validation feedback for bounded correction.",
                        evidence_refs=(),
                        uncertainties=(
                            "The invalid draft is preserved separately from accepted decisions.",
                        ),
                    ),
                )
        assert isinstance(action, SubmitDecision)
        accepted_proposal = Proposal.model_validate(draft)
        accepted_questions = deepcopy(self.questions)
        accepted_questions.apply_updates(
            action.question_updates,
            visible_evidence=evidence,
            source_scope=question_scope,
        )
        question_updates = [
            item.model_dump(mode="json") for item in action.question_updates
        ]
        self.store.commit(
            "decision",
            value.model_dump(mode="json"),
            questions=[
                item.model_dump(mode="json") for item in accepted_questions.records
            ],
            question_updates=question_updates,
        )
        # This snapshot is the publication boundary. A later derived-file failure
        # must leave both the in-memory result and question state at this revision.
        self.proposal = accepted_proposal
        self.decision, self.issues = value, issues
        self.questions = accepted_questions
        self.repair_draft, self.repair_scope = None, set()
        self.store.write(
            f"{self.calls:02d}.accepted-decision.json", value.model_dump(mode="json")
        )
        self.store.write("accepted-proposal.json", draft)
        self.store.write(f"{self.calls:02d}.revision-diff.json", changes)
        self._save_questions()
        self.store.write(
            f"{self.calls:02d}.question-updates.json",
            question_updates,
        )
        details = tuple(
            f"Selection {selection.decision} [{', '.join(selection.candidate_refs)}]: {selection.reason}"
            for selection in value.decisions
        )
        details += tuple(
            f"Annotation {annotation.key} [{', '.join(annotation.source_scope)}]: {annotation.reason}"
            for annotation in value.annotations
        )
        details += tuple(
            f"Needs {gap.source_kind}: {gap.missing}; impact: {gap.impact}"
            for gap in value.evidence_requests
        )
        self._note(phase, "model", value.explanation, details)
        return True

    def _correction_feedback(
        self, error: WorkflowError, draft: dict[str, Any] | None
    ) -> list[dict[str, object]]:
        feedback = [dict(item) for item in error.feedback] or [
            {
                "path": "/",
                "code": error.code,
                "message": "Revise the affected judgment to satisfy the supplied source scope, template targets and evidence. If evidence cannot support it, give an explicit unresolved outcome.",
            }
        ]
        relevant: list[dict[str, object]] = []
        union_names = {
            "ProposedRecord",
            "LinkedRecord",
            "Binding",
            "ConditionalBinding",
            "QualifiedBinding",
            "OtherOutcome",
            "QualifiedOutcome",
            "SupplementalOutcome",
        }
        for item in feedback:
            if item.get("affected_refs"):
                relevant.append(item)
                continue
            scope: set[str] = set()
            parts = str(item.get("path", "/")).strip("/").split("/")
            if parts[:2] == ["action", "submit"]:
                parts = parts[2:]
            elif parts[:1] == ["action"]:
                parts = parts[1:]
            if (
                draft
                and len(parts) >= 2
                and parts[0] in {"records", "other_outcomes"}
                and parts[1].isdigit()
            ):
                groups = draft.get(parts[0], [])
                index = int(parts[1])
                if (
                    isinstance(groups, list)
                    and index < len(groups)
                    and isinstance(groups[index], dict)
                ):
                    group = groups[index]
                    # Pydantic reports all untagged union alternatives. Only the
                    # shape actually supplied is actionable; e.g. a plain
                    # binding error must not ask for optional relationship data.
                    if parts[0] == "records":
                        expected = (
                            "LinkedRecord" if "links" in group else "ProposedRecord"
                        )
                        if any(
                            label in parts and label != expected
                            for label in ("ProposedRecord", "LinkedRecord")
                        ):
                            continue
                    else:
                        expected = (
                            "SupplementalOutcome"
                            if group.get("outcome") == "supplemental"
                            else (
                                "QualifiedOutcome"
                                if "details" in group
                                else "OtherOutcome"
                            )
                        )
                        if any(
                            label in parts and label != expected
                            for label in (
                                "SupplementalOutcome",
                                "QualifiedOutcome",
                                "OtherOutcome",
                            )
                        ):
                            continue
                    scope = (
                        _record_refs(group)
                        if parts[0] == "records"
                        else _source_refs(group)
                    )
                    if parts[0] == "records" and "bindings" in parts:
                        offset = parts.index("bindings") + 1
                        bindings = group.get("bindings", [])
                        if (
                            offset < len(parts)
                            and parts[offset].isdigit()
                            and isinstance(bindings, list)
                        ):
                            binding_index = int(parts[offset])
                            if binding_index < len(bindings):
                                binding = bindings[binding_index]
                                if isinstance(binding, dict):
                                    expected = (
                                        "QualifiedBinding"
                                        if "details" in binding
                                        else (
                                            "ConditionalBinding"
                                            if "when" in binding
                                            else "Binding"
                                        )
                                    )
                                    if any(
                                        label in parts and label != expected
                                        for label in (
                                            "Binding",
                                            "ConditionalBinding",
                                            "QualifiedBinding",
                                        )
                                    ):
                                        continue
                                scope = _source_refs(binding) or scope
            item["path"] = "/" + "/".join(
                part for part in parts if part not in union_names
            )
            item["affected_refs"] = sorted(scope or set(self.packet.output_scope))
            relevant.append(item)
        return relevant or feedback

    def run(self) -> Outcome:
        """Execute the normal two-call path plus bounded evidence/consistency work."""
        reason = "completed_for_review"
        try:
            self.store.write("source.json", self.packet.model_dump(mode="json"))
            self.store.write("budget.json", self.budget.model_dump(mode="json"))
            self.store.write("knowledge-manifest.json", self.knowledge.manifest)
            if self.packet.modality != "reviewed_text":
                raise WorkflowError("image_input_not_implemented")
            self._understand()
            assert self.understanding is not None
            while self._read(
                self.understanding.material_unknowns, stop_on_repeat=False
            ):
                self._understand()
            self._retrieve()
            self._complete_decision()
            assert self.decision is not None
            while True:
                if self._read(self.decision.evidence_requests):
                    affected = set(self.lookup_affected)
                    if self.source_changed:
                        self._understand()
                        assert self.understanding is not None
                        while self._read(
                            self.understanding.material_unknowns, stop_on_repeat=False
                        ):
                            affected.update(self.lookup_affected)
                            self._understand()
                        self._retrieve()
                    self._complete_decision(affected=affected)
                    continue
                context = digest(
                    canonical(
                        {
                            "understanding": self.understanding.model_dump(mode="json"),
                            "candidates": [item.ref for item in self.delivered],
                            "materials": [item.ref for item in self.documents],
                        }
                    ).encode()
                )
                if context not in self.consistency_contexts and needs_consistency(
                    self.understanding, self.packet, self.decision
                ):
                    self._complete_decision("consistency")
                    self.consistency_reviewed = True
                    self.consistency_contexts.add(context)
                    continue
                break
            self.store.write_text("review.md", render(self.decision, self.delivered))
            self.store.write(
                "observation-identities.json",
                {
                    item.key: "obs:"
                    + digest(
                        canonical(
                            {
                                "study": self.packet.study_id,
                                "task": self.packet.task_ref,
                                "local_key": item.key,
                            }
                        ).encode()
                    )[:24]
                    for item in self.decision.target_observations
                },
            )
        except WorkflowError as exc:
            reason = exc.code
            if reason != "artifact_write_failed":
                self.store.write(
                    "stop.json", {"reason": reason, "feedback": exc.feedback}
                )
                self._note(
                    "stopped",
                    "backend",
                    DecisionExplanation(
                        summary="The workflow stopped before another validated decision was available: "
                        + reason,
                        evidence_refs=(),
                        uncertainties=(
                            "No model explanation is inferred for the failed step.",
                        ),
                    ),
                )
                if self.decision is not None:
                    self.store.write_text(
                        "review.md", render(self.decision, self.delivered)
                    )
        status = (
            "failed"
            if self.decision is None
            else (
                "partial"
                if self.issues or reason != "completed_for_review"
                else "proposal"
            )
        )
        outcome = Outcome(
            status=status,
            stop_reason=reason,
            understanding=self.understanding,
            decision=self.decision,
            issues=self.issues,
            calls=self.calls,
            measured_tokens=self.tokens,
            usage_complete=self.usage_complete,
            consistency_reviewed=self.consistency_reviewed,
        )
        self.store.write("result.json", outcome.model_dump(mode="json"))
        return outcome
