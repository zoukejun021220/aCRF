"""Explicit evidence actions and backend-owned questions, separate from annotation.

The ledger records the model's question dispositions. A ``resolved`` entry is
not an assertion that its evidence establishes clinical or study correctness.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from ..graphrag.conditions import Identifier
from ..graphrag.field_contracts import FrozenModel
from .common import BriefReason, WorkflowError
from .proposal import (
    LinkedRecord,
    OtherOutcome,
    ProposedRecord,
    QualifiedOutcome,
    SupplementalOutcome,
)
from .understanding import Unknown


class ReadMaterial(FrozenModel):
    """Select an admitted handle; the backend owns its material type."""

    op: Literal["read_material"]
    handle: Identifier
    source: tuple[Identifier, ...] = Field(min_length=1)
    question: Identifier
    impact: Identifier


class SearchStandard(FrozenModel):
    """Search a stated source meaning without inventing a document identifier."""

    op: Literal["search_standard"]
    concept: Identifier
    collected_property: Identifier
    source: tuple[Identifier, ...] = Field(min_length=1)
    question: Identifier
    impact: Identifier


class QuestionUpdate(FrozenModel):
    """Explicitly answer or defer a known question, with an inspectable reason."""

    question: Identifier
    status: Literal["resolved", "deferred"]
    reason: BriefReason
    support: tuple[Identifier, ...] = ()


class SubmitDecision(FrozenModel):
    """Submit annotation judgments plus explicit question dispositions."""

    op: Literal["submit"]
    records: tuple[ProposedRecord | LinkedRecord, ...]
    other_outcomes: tuple[OtherOutcome | QualifiedOutcome | SupplementalOutcome, ...]
    question_updates: tuple[QuestionUpdate, ...] = ()


EvidenceRequest: TypeAlias = ReadMaterial | SearchStandard
DecisionAction: TypeAlias = Annotated[
    ReadMaterial | SearchStandard | SubmitDecision, Field(discriminator="op")
]


class DecisionTurn(FrozenModel):
    """Choose one bounded action instead of submitting a partial annotation."""

    action: DecisionAction


class QuestionRecord(FrozenModel):
    """A stable backend handle and the latest explicit disposition of its question."""

    handle: Identifier
    question: Identifier
    source: tuple[Identifier, ...] = Field(min_length=1)
    impact: Identifier
    status: Literal["open", "resolved", "deferred"] = "open"
    reason: BriefReason | None = None
    support: tuple[Identifier, ...] = ()


class QuestionLedger:
    """Keep question identity distinct from material caching and request wording.

    Identity uses the exact question and its source set. Changing impact or the
    selected material does not create a new question. The first wording stays
    available; repeated observation never silently clears an explicit deferral.
    """

    def __init__(self) -> None:
        self._records: dict[str, QuestionRecord] = {}
        self._identities: dict[tuple[str, tuple[str, ...]], str] = {}

    @property
    def records(self) -> tuple[QuestionRecord, ...]:
        """Return immutable records in backend allocation order."""
        return tuple(self._records.values())

    @property
    def pending(self) -> tuple[QuestionRecord, ...]:
        """Include deferred questions: deferral does not establish an answer."""
        return tuple(item for item in self.records if item.status != "resolved")

    def get(self, handle: str) -> QuestionRecord:
        """Resolve a backend question handle without creating a guessed question."""
        if handle not in self._records:
            self._fail(
                "question_handle_unknown",
                "/question",
                (),
                "Choose a question handle supplied by the backend.",
                tuple(self._records),
            )
        return self._records[handle]

    def observe(
        self,
        request: EvidenceRequest | Unknown,
        *,
        source_scope: Collection[str] | None = None,
    ) -> str:
        """Register a validated action or an explicit legacy material question.

        Callers can enforce their current authorized source scope here. Update
        validation always requires an explicit scope and visible-evidence set.
        """
        if isinstance(request, Unknown):
            question, source = request.missing, request.affected_refs
        else:
            question, source = request.question, request.source
        if len(source) != len(set(source)):
            self._fail(
                "evidence_source_duplicate",
                "/action/source",
                source,
                "List each affected source once.",
            )
        if source_scope is not None and not set(source) <= set(source_scope):
            self._fail(
                "evidence_source_scope_invalid",
                "/action/source",
                source,
                "Ask about sources within the authorized action scope.",
                tuple(source_scope),
            )
        identity = (question, tuple(sorted(source)))
        if identity in self._identities:
            return self._identities[identity]
        handle = f"Q{len(self._records) + 1}"
        self._records[handle] = QuestionRecord(
            handle=handle,
            question=question,
            source=source,
            impact=request.impact,
        )
        self._identities[identity] = handle
        return handle

    def validate_updates(
        self,
        updates: tuple[QuestionUpdate, ...],
        *,
        visible_evidence: Collection[str],
        source_scope: Collection[str],
    ) -> None:
        """Check references and scope, without deciding whether an answer is true."""
        seen: set[str] = set()
        for index, update in enumerate(updates):
            path = f"/action/question_updates/{index}"
            if update.question not in self._records:
                self._fail(
                    "question_handle_unknown",
                    path + "/question",
                    (),
                    "Choose a question handle supplied by the backend.",
                    tuple(self._records),
                )
            record = self._records[update.question]
            if update.question in seen:
                self._fail(
                    "question_update_duplicate",
                    path + "/question",
                    record.source,
                    "Give one disposition for this question in the current turn.",
                )
            seen.add(update.question)
            if not set(record.source) <= set(source_scope):
                self._fail(
                    "question_source_scope_invalid",
                    path + "/question",
                    record.source,
                    "This question affects sources outside the current revision scope.",
                    tuple(source_scope),
                )
            if len(update.support) != len(set(update.support)):
                self._fail(
                    "question_support_duplicate",
                    path + "/support",
                    record.source,
                    "List each supporting evidence reference once.",
                )
            if not set(update.support) <= set(visible_evidence):
                self._fail(
                    "question_support_unknown",
                    path + "/support",
                    record.source,
                    "Cite only source or material evidence already delivered in this context.",
                    tuple(visible_evidence),
                )

    def apply_updates(
        self,
        updates: tuple[QuestionUpdate, ...],
        *,
        visible_evidence: Collection[str],
        source_scope: Collection[str],
    ) -> tuple[QuestionRecord, ...]:
        """Apply an entire valid batch; a later invalid update leaves all records intact."""
        self.validate_updates(
            updates, visible_evidence=visible_evidence, source_scope=source_scope
        )
        changed = []
        for update in updates:
            record = self._records[update.question].model_copy(
                update={
                    "status": update.status,
                    "reason": update.reason,
                    "support": update.support,
                }
            )
            self._records[update.question] = record
            changed.append(record)
        return tuple(changed)

    @staticmethod
    def _fail(
        code: str,
        path: str,
        source: tuple[str, ...],
        message: str,
        allowed: tuple[str, ...] = (),
    ) -> None:
        raise WorkflowError(
            code,
            feedback=(
                {
                    "path": path,
                    "code": code,
                    "message": message,
                    "affected_refs": list(source),
                    "allowed": sorted(allowed),
                },
            ),
        )
