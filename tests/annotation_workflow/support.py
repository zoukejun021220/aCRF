"""Synthetic development fixtures; never imported by the inference path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.stage_b_mapper_module.annotation_workflow.decision import MappingDecision
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import Reply
from pipeline.stage_b_mapper_module.annotation_workflow.source import SourcePacket
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "kb/cosmos/031429b1d14823721991cd23ee88a11616686ce3"
# Keep the pinned test rule with the shipped fixtures so clean clones do not
# depend on the experiment archive; its source text and provenance stay intact.
RULES = ROOT / "tests/fixtures/annotation_workflow/rules.json"
SYSBP = "cosmos:sdtm:2026-05-26:SYSBP"


def packet() -> SourcePacket:
    """An explicitly synthetic measurement whose result, unit and date share scope."""
    source = {
        "source_id": "synthetic:bp",
        "sha256": "a" * 64,
        "locator": "page:1",
        "authority": "generated",
    }
    return SourcePacket.model_validate(
        {
            "task_ref": "task:bp",
            "study_id": "synthetic-study",
            "study_standard_confirmed": False,
            "standard": {
                "sdtm": "2.0",
                "sdtmig": "3.4",
                "ct_package": "sdtmct-2025-03-28",
                "cosmos_commit": SNAPSHOT.name,
            },
            "output_scope": ["bp", "unit", "date"],
            "objects": [
                {
                    "ref": "measurement",
                    "kind": "context",
                    "text": "One measurement entry; unit and date apply to this systolic pressure",
                    "origin": "blank_crf",
                    "source_ref": source,
                },
                {
                    "ref": "bp",
                    "kind": "field",
                    "text": "Systolic blood pressure",
                    "origin": "blank_crf",
                    "source_ref": source,
                    "context_ref": "measurement",
                    "entry_mode": "entered",
                    "datatype": "integer",
                },
                {
                    "ref": "unit",
                    "kind": "field",
                    "text": "mmHg",
                    "origin": "blank_crf",
                    "source_ref": source,
                    "context_ref": "measurement",
                    "entry_mode": "printed_context",
                },
                {
                    "ref": "date",
                    "kind": "field",
                    "text": "Date of measurement",
                    "origin": "blank_crf",
                    "source_ref": source,
                    "context_ref": "measurement",
                    "entry_mode": "entered",
                    "datatype": "date",
                },
            ],
        }
    )


def understanding() -> CollectionUnderstanding:
    """Source-only interpretation independent of SDTM expected targets."""
    items = []
    for ref, concept, prop, role, mode in [
        (
            "bp",
            "Systolic blood pressure",
            "measurement result",
            "measurement_result",
            "entered",
        ),
        ("unit", "Measurement unit", "unit", "unit", "printed_context"),
        ("date", "Measurement date", "date", "time", "entered"),
    ]:
        items.append(
            {
                "source_refs": [ref],
                "concept": concept,
                "collected_property": prop,
                "source_role": role,
                "value_mode": mode,
                "basis": {
                    "kind": "explicit",
                    "source_refs": [ref],
                    "reason": {
                        "bp": "The label asks for a systolic pressure result, without a patient value.",
                        "unit": "The source prints mmHg as context rather than an entered result.",
                        "date": "The date label identifies the measurement time.",
                    }[ref],
                },
            }
        )
    return CollectionUnderstanding.model_validate(
        {
            "explanation": {
                "summary": "The source collects a systolic result with its printed unit and measurement date.",
                "evidence_refs": ["bp", "unit", "date", "measurement"],
                "uncertainties": [],
            },
            "items": items,
            "bindings": [
                {
                    "source_ref": ref,
                    "relation": relation,
                    "target_source_refs": ["bp"],
                    "source_context_ref": "measurement",
                    "basis": {
                        "kind": "explicit",
                        "source_refs": ["measurement"],
                        "reason": "The shared measurement instruction links this qualifier to the systolic result.",
                    },
                }
                for ref, relation in [("unit", "unit_for"), ("date", "time_of")]
            ],
            "conditions": {"status": "none_stated_in_supplied_source", "entries": []},
            "material_unknowns": [],
        }
    )


def understanding_proposal(
    value: CollectionUnderstanding | None = None,
) -> dict[str, Any]:
    """Explicitly build provider fixtures; production never converts old wire output."""
    raw = (value or understanding()).model_dump(mode="json")
    items = []
    for item in raw["items"]:
        refs = item.pop("source_refs")
        item.pop("value_mode")
        items.extend({**item, "source_ref": ref} for ref in refs)
    raw["items"] = items
    return raw


def understanding_json(value: CollectionUnderstanding | None = None) -> str:
    """Serialize an explicitly requested compact understanding test fixture."""
    return json.dumps(understanding_proposal(value))


def decision() -> MappingDecision:
    """Engineering expectation using the real pinned SYSBP target bundle."""
    annotations = []
    for ref, variable in [("bp", "VSORRES"), ("unit", "VSORRESU"), ("date", "VSDTC")]:
        annotations.append(
            {
                "key": ref,
                "source_scope": [ref],
                "source_condition": {"status": "not_applicable", "value": None},
                "disposition": "mapped",
                "observation_key": "measurement:1",
                "variable": variable,
                "record_selector": {
                    "status": "specified",
                    "value": {
                        "op": "eq",
                        "variable_ref": {"kind": "variable", "id": "VSTESTCD"},
                        "value": "SYSBP",
                        "evidence_ids": [SYSBP],
                    },
                },
                "fixed_assignments": (
                    [
                        {
                            "variable": "VSORRESU",
                            "value": "mmHg",
                            "evidence_refs": ["unit", SYSBP],
                        }
                    ]
                    if ref == "unit"
                    else []
                ),
                "evidence_refs": [ref, SYSBP],
                "reason": {
                    "bp": "The collected systolic result maps to VSORRES under the SYSBP topic selector; no patient value is assigned.",
                    "unit": "VSORRESU belongs to the same measurement; the printed mmHg source supports the fixed unit value.",
                    "date": "VSDTC represents the date of this same SYSBP measurement.",
                }[ref],
            }
        )
    return MappingDecision.model_validate(
        {
            "explanation": {
                "summary": "The SYSBP candidate supplies collected result, unit and date targets for one source measurement; derived standardized values are excluded.",
                "evidence_refs": ["bp", "unit", "date", SYSBP],
                "uncertainties": [
                    "Study approval of the declared standard profile is not supplied."
                ],
            },
            "decisions": [
                {
                    "item_refs": ["item:1"],
                    "candidate_refs": [SYSBP],
                    "decision": "apply",
                    "reason": "Source is a systolic pressure measurement.",
                }
            ],
            "target_observations": [
                {
                    "key": "measurement:1",
                    "source_context_refs": ["measurement"],
                    "candidate_ref": SYSBP,
                    "evidence_refs": ["measurement", SYSBP],
                }
            ],
            "annotations": annotations,
            "material_checks": [],
            "evidence_requests": [],
        }
    )


def proposal() -> dict[str, Any]:
    """Independent compact provider response for the synthetic BP source."""
    return {
        "records": [
            {
                "template": "T1",
                "bindings": [
                    {"source": ["bp"], "target": "VSORRES"},
                    {"source": ["unit"], "target": "VSORRESU"},
                    {"source": ["date"], "target": "VSDTC"},
                ],
                "reason": "The result, printed unit and date describe one systolic measurement supported by T1.",
            }
        ],
        "other_outcomes": [],
        "open_questions": [],
    }


class ScriptedModel:
    """A typed provider-edge replacement that records actual host prompt objects."""

    def __init__(self, outputs: list[str | Reply]) -> None:
        """Own a private response queue and retain each independently serialized request."""
        self.outputs = list(outputs)
        self.requests: list[list[dict[str, str]]] = []

    def generate(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int
    ) -> Reply:
        """Return the next formal response or fail on an unexpected host call."""
        self.requests.append(json.loads(json.dumps(messages)))
        item = self.outputs.pop(0)
        if isinstance(item, Reply):
            return item
        if "action" in schema.get("properties", {}):
            # Migrate the existing explicit Proposal fixtures to the new turn
            # envelope at the provider edge. Production accepts DecisionTurn
            # only; it never guesses or repairs this wrapper from model output.
            try:
                raw = json.loads(item)
            except ValueError:
                raw = None
            if (
                isinstance(raw, dict)
                and "action" not in raw
                and any(
                    key in raw
                    for key in ("records", "other_outcomes", "open_questions")
                )
            ):
                if not raw.get("open_questions"):
                    raw.pop("open_questions", None)
                item = json.dumps({"action": {"op": "submit", **raw}})
        return Reply(
            content=item,
            error=None,
            status=200,
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=100,
            reasoning_present=False,
            response_sha256=None,
            seconds=0,
        )
