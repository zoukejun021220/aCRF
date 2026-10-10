"""Nonclinical formal-channel gate using the exact deployed stage schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import WorkflowError, canonical, digest, parse
from .evidence_actions import DecisionTurn
from .runtime import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SCHEMA_TRANSPORT,
    LocalClient,
    ModelClient,
    Phase,
    Reply,
    SchemaTransport,
    schema_for,
)
from .store import RunStore
from .understanding_proposal import UnderstandingProposal


def cases() -> tuple[tuple[Phase, dict[str, Any]], ...]:
    """Use synthetic structural payloads; success proves transport, not annotation skill."""
    understanding = {
        "explanation": {
            "summary": "The synthetic subject has no clinical meaning in this probe.",
            "evidence_refs": ["probe"],
            "uncertainties": [],
        },
        "items": [
            {
                "source_ref": "probe",
                "concept": None,
                "collected_property": None,
                "source_role": "unknown",
                "basis": {
                    "kind": "unknown",
                    "source_refs": ["probe"],
                    "reason": "This nonclinical subject exercises the item contract.",
                },
                "concept_context": [],
            }
        ],
        "bindings": [],
        "conditions": {"status": "unknown", "entries": []},
        "material_unknowns": [
            {
                "missing": "source text",
                "affected_refs": ["probe"],
                "impact": "cannot interpret",
                "source_kind": "source",
                "material_ref": None,
            }
        ],
    }
    decision: dict[str, Any] = {
        "records": [
            {
                "template": "T1",
                "bindings": [{"source": ["probe"], "target": "PROBE"}],
                "reason": "This is a nonclinical structural binding probe.",
            }
        ],
        "other_outcomes": [],
    }
    unresolved: dict[str, Any] = {
        "records": [],
        "other_outcomes": [
            {
                "source": ["probe"],
                "outcome": "unresolved",
                "reason": "No source was supplied.",
            }
        ],
    }
    read_material: dict[str, Any] = {
        "op": "read_material",
        "handle": "source:probe",
        "source": ["probe"],
        "question": "What does the synthetic source label mean?",
        "impact": "Determines the synthetic subject role.",
    }
    search_standard: dict[str, Any] = {
        "op": "search_standard",
        "concept": "synthetic probe",
        "collected_property": "synthetic result",
        "source": ["probe"],
        "question": "Is there a synthetic standard candidate?",
        "impact": "Determines whether a synthetic binding is supported.",
    }
    actions = (
        {"op": "submit", **decision},
        {
            "op": "submit",
            **unresolved,
            "question_updates": [
                {
                    "question": "Q1",
                    "status": "deferred",
                    "reason": "The synthetic source remains unavailable.",
                    "support": [],
                }
            ],
        },
        read_material,
        search_standard,
    )
    return (
        (("probe", {"ok": True}),) * 3
        + (("understanding", understanding),) * 2
        + tuple(
            (
                "decision",
                DecisionTurn.model_validate({"action": action}).model_dump(mode="json"),
            )
            for action in actions
        )
    )


def valid_reply(reply: Reply, phase: Phase, expected: dict[str, Any]) -> bool:
    """Require measured usage, formal completion, and exact locally validated payload."""
    if (
        reply.error
        or not reply.content
        or reply.finish_reason != "stop"
        or reply.prompt_tokens is None
        or reply.completion_tokens is None
    ):
        return False
    try:
        if phase == "probe":
            return json.loads(reply.content) == expected
        model = UnderstandingProposal if phase == "understanding" else DecisionTurn
        return parse(model, reply.content).model_dump(
            mode="json"
        ) == model.model_validate(expected).model_dump(mode="json")
    except (WorkflowError, ValueError):
        return False


def run_preflight(
    client: ModelClient,
    store: RunStore,
    identity: dict[str, str],
    model: str,
    base_url: str,
) -> bool:
    """Stop at the first failed probe and preserve its request, usage and gate result."""
    records = []
    transport = (
        client.schema_transport
        if isinstance(client, LocalClient)
        else DEFAULT_SCHEMA_TRANSPORT
    )
    effort = (
        client.reasoning_effort
        if isinstance(client, LocalClient)
        else DEFAULT_REASONING_EFFORT
    )
    for index, (phase, expected) in enumerate(cases(), 1):
        schema = schema_for(phase)
        messages = [
            {
                "role": "system",
                "content": "Nonclinical schema transport test. Return the supplied JSON exactly, without extra text.",
            },
            {"role": "user", "content": canonical(expected)},
        ]
        request = {
            "phase": phase,
            "messages": messages,
            "schema": schema,
            "max_tokens": 4800,
            "model": model,
            "base_url": base_url,
            "temperature": 0,
            "reasoning_effort": effort,
            "schema_transport": transport,
        }
        if isinstance(client, LocalClient):
            request["transport"] = client.request_body(messages, schema, 4800)
        store.write(f"{index}.request.json", request)
        reply = client.generate(messages, schema, 4800)
        store.write(f"{index}.reply.json", reply.model_dump(mode="json"))
        passed = valid_reply(reply, phase, expected) and (
            effort != "none" or not reply.reasoning_present
        )
        records.append(
            {
                "phase": phase,
                "schema_sha256": digest(canonical(schema).encode()),
                "passed": passed,
                "reply": reply.model_dump(mode="json"),
            }
        )
        if not passed:
            break
    ready = len(records) == len(cases()) and all(record["passed"] for record in records)
    store.write(
        "gate.json",
        {
            "ready": ready,
            "identity": identity,
            "model": model,
            "base_url": base_url,
            "temperature": 0,
            "reasoning_effort": effort,
            "max_output_tokens": 4800,
            "records": records,
            "scope": "formal_transport_only",
            "schema_transport": transport,
        },
    )
    return ready


def validate_gate(
    path: Path,
    identity: dict[str, str],
    model: str,
    base_url: str,
    schema_transport: SchemaTransport = DEFAULT_SCHEMA_TRANSPORT,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> None:
    """Recheck every stored outcome, schema and configuration; a flag is insufficient."""
    try:
        gate = json.loads(path.read_text())
        if (
            not gate["ready"]
            or gate["identity"] != identity
            or gate["model"] != model
            or gate["base_url"] != base_url
            or gate["reasoning_effort"] != reasoning_effort
            or gate["temperature"] != 0
            or gate["max_output_tokens"] != 4800
            or gate["schema_transport"] != schema_transport
        ):
            raise ValueError("gate_identity")
        if len(gate["records"]) != len(cases()):
            raise ValueError("gate_incomplete")
        for record, (phase, expected) in zip(gate["records"], cases()):
            if (
                (
                    reasoning_effort == "none"
                    and Reply.model_validate(record["reply"]).reasoning_present
                )
                or record["phase"] != phase
                or record["schema_sha256"]
                != digest(canonical(schema_for(phase)).encode())
                or not valid_reply(
                    Reply.model_validate(record["reply"]), phase, expected
                )
            ):
                raise ValueError("gate_failed")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise WorkflowError("runtime_gate_not_passed") from exc
