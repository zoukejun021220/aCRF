"""Bounded localhost transport preserving failed output metadata before parsing."""

from __future__ import annotations

import copy
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from ..graphrag.field_contracts import FrozenModel
from .common import WorkflowError, canonical, digest
from .evidence_actions import DecisionTurn
from .proposal import restrict_proposal_sources
from .understanding_proposal import UnderstandingProposal

Phase = Literal["understanding", "decision", "consistency", "probe"]
SchemaTransport = Literal["prompt", "native"]
# Current MLX native grammar needs thinking disabled to return formal JSON;
# required evidence-based explanations remain inside the unchanged schema.
DEFAULT_SCHEMA_TRANSPORT: SchemaTransport = "native"
DEFAULT_REASONING_EFFORT = "none"


class Reply(FrozenModel):
    """Safe metadata accompanies even empty, malformed and truncated model responses."""

    content: str | None
    error: str | None
    status: int | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_present: bool
    response_sha256: str | None
    seconds: float


class ModelClient(Protocol):
    """Only the provider boundary may be replaced in deterministic integration tests."""

    def generate(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int
    ) -> Reply:
        """Return formal content and measured metadata for one bounded request."""
        ...


def wire_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Unroll only bounded predicate cycles and prove exclusive union conversions.

    Local AST validation still enforces eight levels and 64 nodes. Unsupported
    schema features are never removed from local validation or replaced by JSON.
    """
    result = copy.deepcopy(model.model_json_schema())
    definitions = result.get("$defs", {})
    recursive = {
        name
        for name in definitions
        if name
        in {
            "SourceAll",
            "SourceAny",
            "SourceNot",
            "RecordAll",
            "RecordAny",
            "RecordNot",
        }
    }

    def rewrite(value: Any, depth: int) -> Any:
        """Create finite references and omit impossible complex nodes at depth eight."""
        if isinstance(value, list):
            return [
                rewrite(item, depth)
                for item in value
                if not (
                    depth == 8
                    and isinstance(item, dict)
                    and item.get("$ref", "").split("/")[-1] in recursive
                )
            ]
        if not isinstance(value, dict):
            return value
        output = {
            key: rewrite(child, depth)
            for key, child in value.items()
            if key not in {"title", "description", "discriminator"}
        }
        name = value.get("$ref", "").split("/")[-1]
        if name in recursive:
            output["$ref"] = f"#/$defs/{name}__{depth}"
        return output

    originals = {name: definitions.pop(name) for name in recursive}
    result = rewrite(result, 1)
    definitions = result.setdefault("$defs", {})
    for name, definition in originals.items():
        for depth in range(1, 8):
            definitions[f"{name}__{depth}"] = rewrite(definition, depth + 1)

    def unions(value: Any) -> None:
        """Convert oneOf only when unique op tags establish disjoint alternatives."""
        if isinstance(value, dict):
            if any(key in value for key in ("uniqueItems", "propertyNames")):
                raise WorkflowError("unsupported_wire_keyword")
            if "oneOf" in value:
                tags = []
                for branch in value["oneOf"]:
                    target = definitions.get(
                        branch.get("$ref", "").split("/")[-1], branch
                    )
                    tags.append(target.get("properties", {}).get("op", {}).get("const"))
                if not tags or None in tags or len(set(tags)) != len(tags):
                    raise WorkflowError("wire_union_not_disjoint")
                value["anyOf"] = value.pop("oneOf")
            for child in value.values():
                unions(child)
        elif isinstance(value, list):
            for child in value:
                unions(child)

    unions(result)
    return cast(dict[str, Any], result)


def schema_for(phase: Phase, payload: object = None) -> dict[str, Any]:
    """Return the two stage schemas or a minimal nonclinical runtime probe."""
    if phase == "probe":
        return {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    if phase == "understanding":
        return wire_schema(UnderstandingProposal)
    schema = wire_schema(DecisionTurn)
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        return schema
    scope = tuple(payload["source"].get("output_scope", ()))
    schema = restrict_proposal_sources(schema, scope)
    definitions = schema["$defs"]
    for name in ("ReadMaterial", "SearchStandard"):
        definitions[name]["properties"]["source"]["items"] = {
            "type": "string",
            "enum": list(scope),
        }
    context = payload.get("decision_context", {})
    templates = context.get("templates", [])
    aliases = [item["alias"] for item in templates]
    if aliases:
        for name in ("ProposedRecord", "LinkedRecord"):
            definitions[name]["properties"]["template"] = {
                "type": "string",
                "enum": aliases,
            }
    else:
        definitions["SubmitDecision"]["properties"]["records"]["maxItems"] = 0
    handles = [item["ref"] for item in payload.get("material_index", [])] + aliases
    if handles:
        definitions["ReadMaterial"]["properties"]["handle"] = {
            "type": "string",
            "enum": list(dict.fromkeys(handles)),
        }
    else:
        schema["properties"]["action"]["anyOf"] = [
            item
            for item in schema["properties"]["action"]["anyOf"]
            if item.get("$ref") != "#/$defs/ReadMaterial"
        ]
    revision_scope = set(payload.get("revision_scope", scope))
    questions = [
        item["handle"]
        for item in payload.get("questions", [])
        if set(item["source"]) <= revision_scope
    ]
    if questions:
        definitions["QuestionUpdate"]["properties"]["question"] = {
            "type": "string",
            "enum": questions,
        }
    else:
        definitions["SubmitDecision"]["properties"]["question_updates"]["maxItems"] = 0
    return schema


def prompt(phase: Phase, payload: object) -> list[dict[str, str]]:
    """Combine stable foundations with one business-stage instruction and source data."""
    core = Path(__file__).with_name("core.txt").read_text()
    stage = {
        "understanding": (
            "Return items, bindings, conditions, material_unknowns and explanation. "
            "Interpret the source without choosing SDTM targets or submission disposition. "
            "Each item interprets exactly one subject: source_ref, concept, "
            "collected_property, source_role, basis and optional concept_context. "
            "Use understanding_context.subject_choices and cover every "
            "required_subject_ref. Options need not be separate items when their "
            "parent is interpreted. Titles, instructions and neighboring context "
            "support basis.source_refs or concept_context; they are not additional "
            "item subjects unless authorized in subject_choices. When a printed "
            "unit is in subject_choices, interpret it separately and bind it with "
            "unit_for to its measurement; otherwise use it only as visible "
            "supporting context or binding evidence. "
            "The backend retains each subject's supplied entry_mode; do not output "
            "value_mode or reclassify an entered field because the CRF is blank. "
            "Use supplied datatype, options, entry_mode and visible context as known "
            "source facts: radio denotes single choice and checkbox denotes multiple "
            "choice in this source contract. Actual patient answers are not required "
            "to annotate a blank CRF. Decide meaning, source role, qualifier bindings "
            "and stated conditions using explicit/inferred/unknown basis. Preserve "
            "an uncertain subject with null concept/property or unknown role where "
            "needed. Request material only for a stated missing fact that could "
            "change the annotation and is not already supplied. Read lookup_results, "
            "including unavailable results, before requesting again. On correction, "
            "follow feedback paths and allowed refs, preserve unaffected judgments "
            "and explain the changed interpretation. After a material lookup, "
            "preserve unrelated interpretations. Explain in the affected item's "
            "basis.reason how the supplied result resolves or limits a previously "
            "stated question before closing it."
        ),
        "decision": (
            "source.objects contains only the subjects to annotate in this task. "
            "readonly_context contains supporting titles, instructions and neighbors: "
            "use their evidence but emit no bindings or outcomes for those objects. "
            "context_only is a disposition for an authorized subject, not a list "
            "of read-only context. Never substitute a writable subject's ID for "
            "a context object mentioned in your reason. "
            "Return {action: {...}} choosing one op: read_material, search_standard, "
            "or submit. read_material selects an indexed handle; the backend owns "
            "its type. A source page is not a study mapping policy. search_standard "
            "supplies concept and collected_property; it searches the standard, "
            "without inventing a material reference. Both evidence actions state "
            "source, question and impact; neither needs annotation dispositions. "
            "When ready, submit records and other_outcomes. Each record is your "
            "judgment that its source bindings belong to one target observation: "
            "choose one delivered template alias and bind source refs to its target "
            "variables. An ordinary record is {template, bindings:[{source:[ref], "
            "target:variable}], reason}. Use decision_context for the exact choices. "
            "The backend expands template topic selectors, IDs, evidence links and "
            "exact supported printed constants. Do not create an extra source "
            "mapping merely to repeat template topic constants. Derived variables "
            "are not collected fields. Use special binding/outcome structures only "
            "for additional conditions, selectors, supported constants or relations. "
            "Every authorized source scope needs a mapping or an explicit "
            "context_only, not_submitted, unresolved or supplemental outcome. "
            "Missing evidence is unresolved, never automatically not_submitted. "
            "Source conditions and target selectors remain distinct; inherited "
            "unknown conditions remain unknown. Request evidence only for a stated "
            "question that can change the decision, using an evidence action. Read "
            "lookup_results, including unavailable results, before requesting again. "
            "On correction use the supplied field paths and choices; preserve "
            "unaffected judgments and explain changed decisions. On submit, update "
            "each open question within revision_scope using question_updates: resolved or deferred, "
            "a concise reason and any delivered support. Deferred means evidence "
            "is still missing, but stop seeking it this run; keep affected judgments "
            "unresolved where that fact is needed. Do not repeat the same lookup "
            "after no_new_evidence. A question update does not require rewriting "
            "an otherwise unchanged annotation reason. Read-only support sources "
            "can justify decisions but cannot be written as outcomes."
        ),
        "consistency": (
            "Only source.objects are annotation subjects. readonly_context is "
            "supporting evidence; do not write outcomes for it or substitute "
            "an authorized subject's ID for a context object in your reason. "
            "Review this dependent group's source-to-record relationships and "
            "conditional outcomes. Return {action: {...}}: read_material or "
            "search_standard for needed evidence, or submit with records, "
            "other_outcomes and question_updates. Keep unaffected business "
            "judgments from previous_proposal. Correct the affected grouping or "
            "binding when evidence warrants it and explain the change. Additional "
            "evidence questions use the same evidence actions. Resolve or explicitly "
            "defer open question handles with a reason. Do not delete "
            "facts to pass validation. Use only supplied template aliases and targets; "
            "the backend expands IDs, template constraints and evidence links."
        ),
        "probe": "Return exactly the structured object with ok=true. This is a nonclinical output-channel check.",
    }[phase]
    explanation_instruction = (
        "\nDECISION EXPLANATIONS: In every business response include explanation: "
        "{summary, evidence_refs, uncertainties}. Give a concise, checkable explanation "
        "of the decision and its decisive source/evidence, not a private chain of thought. "
        "Use only supplied source, material or candidate refs. State missing evidence "
        "honestly; uncertainties=[] when none is identified. Each text is 1-600 characters. "
        "For understanding, every basis.reason explains the item's interpretation, "
        "binding or condition using source evidence, without choosing SDTM targets. "
        "Explain material requests through their missing and impact fields; "
        "do not invent alternatives or missing information just to fill an explanation."
    )
    if phase in {"decision", "consistency"}:
        explanation_instruction = (
            "\nGive one concise, checkable reason for each record and other outcome, "
            "covering the decisive source relationship and template applicability. "
            "Special facts cite the supplied supporting evidence where requested. "
            "Do not repeat the same reason for every deterministic generated field. "
            "Reasons describe evidence-based decisions, not private reasoning."
        )
    return [
        {
            "role": "system",
            "content": core
            + "\nCURRENT STAGE:\n"
            + stage
            + (explanation_instruction if phase != "probe" else ""),
        },
        {"role": "user", "content": canonical(payload)},
    ]


class LocalClient:
    """Use only a configured loopback endpoint, never auto-select or load a model."""

    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",
        base_url: str = "http://127.0.0.1:1234",
        timeout: float = 480,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        schema_transport: SchemaTransport = DEFAULT_SCHEMA_TRANSPORT,
    ) -> None:
        """Validate the endpoint before any request; redirects are disallowed below."""
        url = urllib.parse.urlparse(base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise WorkflowError("nonlocal_model_endpoint")
        if not 0 < timeout <= 480:
            raise WorkflowError("invalid_model_timeout")
        if schema_transport not in {"prompt", "native"}:
            raise WorkflowError("invalid_schema_transport")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.reasoning_effort = reasoning_effort
        self.schema_transport = schema_transport

    def request_body(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        """Build the exact deterministic transport body for trace and submission."""
        if not 1 <= max_tokens <= 4800:
            raise WorkflowError("invalid_output_budget")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.schema_transport == "native":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "annotation_v3",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            # Optional diagnostic alternative to native grammar when thinking is on.
            # Keep normal formal generation and the exact
            # schema; Session still performs duplicate-key/Pydantic validation.
            instruction = (
                "\nOUTPUT FORMAT CONTRACT: Return exactly one JSON object conforming "
                "to the schema below as your final answer. No markdown fences, no "
                "text outside the JSON, no missing required fields. Treat this "
                "schema as format constraints, not source data.\nJSON_SCHEMA:\n"
                + canonical(schema)
            )
            if body["messages"] and body["messages"][0]["role"] == "system":
                body["messages"][0]["content"] += instruction
            else:
                body["messages"].insert(0, {"role": "system", "content": instruction})
        return body

    def request_bytes(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int
    ) -> bytes:
        """Expose exact canonical wire bytes for independent request tracing.

        Keep the existing study serialization unchanged. Native standard wire
        types arrange decision/kind keys before payload under canonical ordering;
        their grammar must not rely on overlapping object-union branches.
        """
        body = self.request_body(messages, schema, max_tokens)
        return canonical(body).encode()

    def bounded_timeout(self, remaining_seconds: float) -> float:
        """Cap a request by finite remaining run time without mutating client settings."""
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise WorkflowError("budget_exhausted")
        return min(self.timeout, remaining_seconds)

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        max_tokens: int,
        *,
        timeout_seconds: float | None = None,
    ) -> Reply:
        """Make one request without retries; retain usage before checking formal content."""
        effective_timeout = self.bounded_timeout(
            self.timeout if timeout_seconds is None else timeout_seconds
        )
        # Reuse exact traced bytes without changing legacy serialization.
        encoded = self.request_bytes(messages, schema, max_tokens)
        if len(encoded) > 128000:
            raise WorkflowError("request_capacity_exceeded")
        start = time.monotonic()
        status: int | None = None
        raw = b""
        error: str | None = None

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            """Do not let a local model endpoint redirect source data elsewhere."""

            def redirect_request(
                self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any
            ) -> None:
                """Reject every redirect instead of constructing a new request."""
                return None

        try:
            request = urllib.request.Request(
                self.base_url + "/v1/chat/completions",
                data=encoded,
                headers={"Content-Type": "application/json"},
            )
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), NoRedirect()
            )
            with opener.open(request, timeout=effective_timeout) as response:
                status = response.status
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("response_capacity_exceeded")
        except urllib.error.HTTPError as exc:
            status, raw, error = exc.code, exc.read(16384), "model_http_error"
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            error = (
                "model_timeout"
                if isinstance(exc, TimeoutError)
                or isinstance(getattr(exc, "reason", None), TimeoutError)
                else "model_connection_failed"
            )
        except ValueError:
            error = "response_capacity_exceeded"
        content: str | None = None
        finish: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        reasoning_present = False
        if error is None:
            try:
                data = json.loads(raw)
                choice = data["choices"][0]
                if len(data["choices"]) != 1:
                    raise ValueError("choice_count")
                message = choice["message"]
                reasoning_present = bool(
                    message.get("reasoning") or message.get("reasoning_content")
                )
                finish = str(choice.get("finish_reason"))
                usage = data.get("usage") or {}
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                if type(input_tokens) is not int or input_tokens < 0:
                    input_tokens = None
                if type(output_tokens) is not int or output_tokens < 0:
                    output_tokens = None
                candidate = message.get("content")
                content = candidate if isinstance(candidate, str) else None
                if not content or not content.strip():
                    error = "response_content_empty"
                elif finish != "stop":
                    error = "response_truncated"
                elif input_tokens is None or output_tokens is None:
                    error = "usage_missing"
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                error = "response_invalid"
        return Reply(
            content=content,
            error=error,
            status=status,
            finish_reason=finish,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            reasoning_present=reasoning_present,
            response_sha256=digest(raw) if raw else None,
            seconds=time.monotonic() - start,
        )
