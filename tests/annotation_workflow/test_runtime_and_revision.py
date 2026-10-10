"""Exercise transport, bounded revisions and comparison through observable outcomes."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from support import (
    RULES,
    SNAPSHOT,
    ScriptedModel,
    decision,
    packet,
    proposal,
    understanding,
    understanding_json,
    understanding_proposal,
)

from pipeline.stage_b_mapper_module.annotation_workflow.common import (
    WorkflowError,
    canonical,
)
from pipeline.stage_b_mapper_module.annotation_workflow.decision import MappingDecision
from pipeline.stage_b_mapper_module.annotation_workflow.evaluation import compare
from pipeline.stage_b_mapper_module.annotation_workflow.knowledge import Knowledge
from pipeline.stage_b_mapper_module.annotation_workflow.preflight import (
    cases,
    run_preflight,
    validate_gate,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import (
    LocalClient,
    Reply,
    schema_for,
)
from pipeline.stage_b_mapper_module.annotation_workflow.session import Budget, Session
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    Material,
    MaterialStore,
)
from pipeline.stage_b_mapper_module.annotation_workflow.store import RunStore


class RevisionTests(unittest.TestCase):
    """New evidence and invalid revisions cross the real retrieval/persistence boundary."""

    def setUp(self) -> None:
        """Give every test its own disk location and public SQLite index."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
        self.addCleanup(self.knowledge.close)

    def test_one_format_repair_preserves_facts_and_stops_if_still_invalid(self) -> None:
        """Repair is stage-local, bounded and cannot bypass schema validation."""
        outputs = [
            "{invalid",
            understanding_json(),
            canonical(proposal()),
        ]
        result = Session(
            packet(), self.knowledge, ScriptedModel(outputs), self.output
        ).run()
        self.assertEqual(
            [a.variable for a in result.decision.annotations],
            ["VSORRES", "VSORRESU", "VSDTC"],
        )
        self.assertEqual(result.calls, 3)
        self.assertEqual(
            json.loads((self.output / "01.reply.json").read_text())["content"],
            "{invalid",
        )
        second = Session(
            packet(),
            self.knowledge,
            ScriptedModel(["bad", "bad", "bad"]),
            self.output.parent / "invalid",
        ).run()
        self.assertEqual(second.stop_reason, "understanding_correction_no_progress")
        self.assertEqual(second.calls, 2)
        self.assertIsNone(second.understanding)

    def test_budget_and_stop_preserve_partial_state_without_next_request(self) -> None:
        """Known exhausted usage and operator stop cannot silently trigger another call."""
        reply = Reply(
            content=understanding_json(),
            error=None,
            status=200,
            finish_reason="stop",
            prompt_tokens=900,
            completion_tokens=200,
            reasoning_present=False,
            response_sha256=None,
            seconds=0,
        )
        result = Session(
            packet(),
            self.knowledge,
            ScriptedModel([reply]),
            self.output,
            budget=Budget(max_total_tokens=1000),
        ).run()
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.measured_tokens, 1100)
        self.assertEqual(result.understanding, understanding())
        session = Session(
            packet(), self.knowledge, ScriptedModel([]), self.output.parent / "stop"
        )
        (session.store.path / "STOP").write_text("operator requested stop")
        stopped = session.run()
        self.assertEqual(stopped.stop_reason, "operator_stop")
        self.assertEqual(stopped.calls, 0)

    def test_missing_usage_stops_with_explicit_uncertainty(self) -> None:
        """Unknown usage cannot be presented as a measured zero or allow further work."""
        reply = Reply(
            content=understanding_json(),
            error="usage_missing",
            status=200,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_present=False,
            response_sha256=None,
            seconds=0,
        )
        result = Session(
            packet(), self.knowledge, ScriptedModel([reply]), self.output
        ).run()
        self.assertEqual(result.stop_reason, "usage_missing")
        self.assertFalse(result.usage_complete)
        self.assertEqual(result.calls, 1)
        self.assertIsNone(result.understanding)

    def test_late_source_evidence_reinterprets_before_rebinding(self) -> None:
        """A source document affects the next understanding payload and candidate query."""
        read = {
            "action": {
                "op": "read_material",
                "handle": "guide",
                "source": ["bp"],
                "question": "What is the measurement instruction?",
                "impact": "Verify source interpretation before binding a target.",
            }
        }
        material = Material(
            ref="guide",
            kind="source",
            study_id=packet().study_id,
            text="The entry is systolic pressure.",
            source_ref=packet().objects[0].source_ref,
            origin="completion_guidance",
        )
        model = ScriptedModel(
            [
                understanding_json(),
                canonical(read),
                understanding_json(),
                canonical(
                    {
                        "action": {
                            "op": "submit",
                            "records": [
                                {
                                    **proposal()["records"][0],
                                    "reason": "The retrieved completion guide confirms the systolic measurement and supports these source bindings.",
                                }
                            ],
                            "other_outcomes": [],
                            "question_updates": [
                                {
                                    "question": "Q1",
                                    "status": "resolved",
                                    "reason": "The completion guide identifies the entry as systolic pressure.",
                                    "support": ["guide"],
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        result = Session(
            packet(), self.knowledge, model, self.output, MaterialStore((material,))
        ).run()
        self.assertEqual(
            [a.variable for a in result.decision.annotations],
            ["VSORRES", "VSORRESU", "VSDTC"],
        )
        self.assertEqual(result.calls, 4)
        self.assertEqual(result.stop_reason, "completed_for_review")
        payload = json.loads(model.requests[2][1]["content"])
        self.assertEqual(payload["materials"][0]["text"], material.text)
        self.assertNotIn("candidates", payload)
        self.assertTrue((self.output / "02.action.json").exists())
        self.assertTrue((self.output / "03.accepted-understanding.json").exists())
        self.assertFalse((self.output / "02.accepted-decision.json").exists())
        self.assertEqual(len(list(self.output.glob("*.accepted-decision.json"))), 1)
        self.assertIsNone(
            json.loads(model.requests[3][1]["content"])["previous_proposal"]
        )
        snapshot = json.loads((self.output / "snapshot.json").read_text())
        self.assertEqual(snapshot["revision"], 3)
        self.assertEqual(snapshot["value"], result.decision.model_dump(mode="json"))
        ledger = json.loads((self.output / "question-ledger.json").read_text())
        self.assertEqual(
            (ledger[0]["status"], ledger[0]["support"]), ("resolved", ["guide"])
        )

    def test_unexplained_deletion_retains_previous_committed_decision(self) -> None:
        """A valid-looking shorter repair cannot erase an existing annotation."""
        first = {
            "action": {
                "op": "submit",
                "records": proposal()["records"],
                "other_outcomes": [],
            }
        }
        read = {
            "action": {
                "op": "read_material",
                "handle": "study-rule",
                "source": ["bp"],
                "question": "Which study instruction applies to this measurement?",
                "impact": "Verify the study convention for the accepted mapping.",
            }
        }
        material = Material(
            ref="study-rule",
            kind="study",
            study_id=packet().study_id,
            text="Use the declared standard profile.",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        bad = proposal()
        bad["records"][0]["bindings"] = bad["records"][0]["bindings"][:1]
        bad_turn = {
            "action": {
                "op": "submit",
                "records": bad["records"],
                "other_outcomes": [],
                "question_updates": [
                    {
                        "question": "Q1",
                        "status": "resolved",
                        "reason": "The study rule confirms the declared standard profile.",
                        "support": ["study-rule"],
                    }
                ],
            }
        }
        source = packet().model_copy(
            update={
                "objects": tuple(
                    (
                        obj.model_copy(update={"parent_ref": "measurement"})
                        if obj.ref == "bp"
                        else obj
                    )
                    for obj in packet().objects
                ),
            }
        )
        result = Session(
            source,
            self.knowledge,
            ScriptedModel(
                [
                    understanding_json(),
                    canonical(first),
                    canonical(read),
                    canonical(bad_turn),
                    canonical(bad_turn),
                ]
            ),
            self.output,
            MaterialStore((material,)),
        ).run()
        self.assertEqual(result.stop_reason, "correction_no_progress")
        self.assertEqual(result.calls, 5)
        self.assertEqual(len(result.decision.annotations), 3)
        self.assertEqual(
            result.decision.model_dump(mode="json"),
            json.loads((self.output / "02.accepted-decision.json").read_text()),
        )
        self.assertEqual(len(list(self.output.glob("*.accepted-decision.json"))), 1)
        self.assertEqual(
            json.loads((self.output / "04.validation.json").read_text())["code"],
            "unaffected_judgment_changed",
        )
        self.assertEqual(
            json.loads((self.output / "question-ledger.json").read_text())[0]["status"],
            "open",
        )
        self.assertEqual(
            len(
                json.loads((self.output / "snapshot.json").read_text())["value"][
                    "annotations"
                ]
            ),
            3,
        )

    def test_evaluator_rejects_plausible_semantic_mutations(self) -> None:
        """Missing/extra facts, source conditions and wrong selectors survive comparison."""
        for mutation in ("selector", "source_condition", "missing", "extra"):
            value = decision().model_dump(mode="json")
            if mutation == "selector":
                value["annotations"][0]["record_selector"]["value"]["value"] = "DIABP"
            elif mutation == "source_condition":
                value["annotations"][0]["source_condition"] = {
                    "status": "specified",
                    "value": {"op": "eq", "source_id": "bp", "value": 100},
                }
            elif mutation == "missing":
                value["annotations"].pop()
            else:
                value["annotations"].append(
                    {**value["annotations"][0], "key": "duplicate"}
                )
            with self.subTest(mutation=mutation):
                result = compare(MappingDecision.model_validate(value), decision())
                self.assertFalse(result.exact)
                self.assertTrue(result.missing or result.extra)
        self.assertTrue(compare(decision(), decision()).exact)
        self.assertEqual(compare(decision(), decision()).expert_adjudication, "pending")

    def test_gate_requires_all_actual_schemas_and_current_identity(self) -> None:
        """A ready flag, missing attempts or changed code cannot authorize live mapping."""
        outputs = [canonical(expected) for _, expected in cases()]
        self.assertTrue(
            run_preflight(
                ScriptedModel(outputs),
                RunStore(self.output),
                {"code": "abc"},
                "local-model",
                "http://localhost:1234",
            )
        )
        gate = self.output / "gate.json"
        validate_gate(gate, {"code": "abc"}, "local-model", "http://localhost:1234")
        with self.assertRaises(WorkflowError):
            validate_gate(
                gate, {"code": "changed"}, "local-model", "http://localhost:1234"
            )
        with self.assertRaises(WorkflowError):
            validate_gate(
                gate, {"code": "abc"}, "local-model", "http://localhost:1234", "prompt"
            )
        with self.assertRaises(WorkflowError):
            validate_gate(
                gate,
                {"code": "abc"},
                "local-model",
                "http://localhost:1234",
                "native",
                "low",
            )
        value = json.loads(gate.read_text())
        changed_effect = json.loads(gate.read_text())
        changed_effect["records"][0]["reply"]["reasoning_present"] = True
        gate.write_text(canonical(changed_effect))
        with self.assertRaises(WorkflowError):
            validate_gate(gate, {"code": "abc"}, "local-model", "http://localhost:1234")
        missing_mode = {
            key: entry for key, entry in value.items() if key != "schema_transport"
        }
        gate.write_text(canonical(missing_mode))
        with self.assertRaises(WorkflowError):
            validate_gate(gate, {"code": "abc"}, "local-model", "http://localhost:1234")
        value["records"].pop()
        gate.write_text(canonical(value))
        with self.assertRaises(WorkflowError):
            validate_gate(gate, {"code": "abc"}, "local-model", "http://localhost:1234")


class HttpTests(unittest.TestCase):
    """Use real loopback HTTP; only response generation is synthetic."""

    def serve(
        self, responses: list[dict[str, Any]], release: threading.Event | None = None
    ) -> tuple[LocalClient, list[dict[str, Any]]]:
        """Own and close a loopback server without emitting source-bearing logs."""
        queue, requests = list(responses), []

        class Handler(BaseHTTPRequestHandler):
            """A provider-edge fixture responding with queued OpenAI-compatible envelopes."""

            def do_POST(self) -> None:
                """Capture actual wire JSON and return one finite response."""
                requests.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                if release is not None:
                    # A finite provider-edge stall proves the HTTP timeout is
                    # applied, without asserting wall-clock durations.
                    release.wait(timeout=2)
                payload = canonical(queue.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    # Expected when the timeout test closes the client first.
                    return

            def log_message(self, format: str, *args: object) -> None:
                """Suppress HTTP logging; tests inspect outcomes directly."""
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def close() -> None:
            """Stop the owned socket and join the worker after assertions or failures."""
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

        self.addCleanup(close)
        return LocalClient(base_url=f"http://127.0.0.1:{server.server_port}"), requests

    def envelope(self, content: str, finish: str = "stop") -> dict[str, Any]:
        """Construct only the external provider envelope; internal seams remain real."""
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning": "private channel fixture",
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 62, "completion_tokens": 6},
        }

    def test_native_schema_property_order_reaches_real_http_boundary(self) -> None:
        """Keep decision/discriminator fields before payload under canonical JSON.

        Native generation uses property order. The flat wire types deliberately
        preserve decision/kind-first order despite legacy canonical serialization.
        """
        from pipeline.stage_b_mapper_module.annotation_workflow.runtime import (
            wire_schema,
        )
        from pipeline.stage_b_mapper_module.annotation_workflow.standard_wire import (
            WireAnnotation,
        )

        schema = wire_schema(WireAnnotation)
        client, requests = self.serve([self.envelope("{}")])
        client.generate([], schema, 64)
        delivered = requests[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(delivered["properties"])[0], "decision")
        self.assertEqual(list(delivered["$defs"]["WireValue"]["properties"])[0], "kind")
        self.assertEqual(
            list(delivered["$defs"]["WireRecord"]["properties"])[0], "kind"
        )

    def test_http_source_retrieval_decision_storage_chain(self) -> None:
        """Full workflow consumes real wire responses and persists complete facts."""
        client, requests = self.serve(
            [
                self.envelope(understanding_json()),
                self.envelope(
                    canonical(
                        {
                            "action": {
                                "op": "submit",
                                "records": proposal()["records"],
                                "other_outcomes": [],
                            }
                        }
                    )
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
            self.addCleanup(knowledge.close)
            output = Path(directory) / "run"
            result = Session(packet(), knowledge, client, output).run()
            self.assertEqual(
                [a.variable for a in result.decision.annotations],
                ["VSORRES", "VSORRESU", "VSDTC"],
            )
            self.assertEqual(
                result.decision.annotations[1].fixed_assignments[0].value, "mmHg"
            )
            self.assertEqual(result.measured_tokens, 136)
            saved = json.loads((output / "02.request.json").read_text())
            self.assertEqual(saved["transport"], requests[1])
            self.assertEqual(requests[1]["reasoning_effort"], "none")
            delivered = requests[1]["response_format"]["json_schema"]["schema"]
            self.assertEqual(
                delivered,
                schema_for(
                    "decision", json.loads(requests[1]["messages"][1]["content"])
                ),
            )
            self.assertEqual(
                len(
                    json.loads(requests[1]["messages"][1]["content"])[
                        "decision_context"
                    ]["templates"][0]["rows"]
                ),
                11,
            )
            self.assertIn("mmHg", (output / "review.md").read_text())

    def test_empty_truncated_missing_usage_and_malformed_http_responses(self) -> None:
        """Runtime faults remain distinct and empty content retains measured usage."""
        missing = self.envelope('{"ok": true}')
        missing.pop("usage")
        client, _ = self.serve(
            [
                self.envelope(""),
                self.envelope('{"ok": true}', "length"),
                missing,
                {"unexpected": True},
            ]
        )
        for expected in (
            "response_content_empty",
            "response_truncated",
            "usage_missing",
            "response_invalid",
        ):
            with self.subTest(expected=expected):
                reply = client.generate([], schema_for("probe"), 512)
                self.assertEqual(reply.error, expected)
                if expected == "response_content_empty":
                    self.assertEqual(reply.prompt_tokens + reply.completion_tokens, 68)
                    self.assertTrue(reply.reasoning_present)

    def test_request_deadline_reaches_the_real_http_boundary(self) -> None:
        """A stalled provider times out under the per-call budget instead of the client cap."""
        release = threading.Event()
        client, _ = self.serve([self.envelope('{"ok": true}')], release)
        try:
            reply = client.generate([], schema_for("probe"), 512, timeout_seconds=0.05)
            self.assertEqual(reply.error, "model_timeout")
            self.assertIsNone(reply.content)
            self.assertIsNone(reply.completion_tokens)
        finally:
            release.set()

    def test_gate_rejects_ineffective_thinking_configuration(self) -> None:
        """Valid formal JSON alone cannot pass a probe that still emits separate thinking."""
        client, _ = self.serve([self.envelope('{"ok": true}')])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate"
            passed = run_preflight(
                client, RunStore(output), {"code": "abc"}, client.model, client.base_url
            )
            self.assertFalse(passed)
            gate = json.loads((output / "gate.json").read_text())
            self.assertFalse(gate["ready"])
            self.assertEqual(len(gate["records"]), 1)
            self.assertTrue(gate["records"][0]["reply"]["reasoning_present"])

    def test_reasoning_json_is_never_promoted_to_formal_output(self) -> None:
        """A schema-valid reasoning object still fails before retrieval or publication."""
        envelope = self.envelope("")
        envelope["choices"][0]["message"]["reasoning_content"] = understanding_json()
        client, requests = self.serve([envelope])
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
            self.addCleanup(knowledge.close)
            output = Path(directory) / "run"
            result = Session(packet(), knowledge, client, output).run()
            self.assertEqual(result.stop_reason, "response_content_empty")
            self.assertIsNone(result.understanding)
            self.assertIsNone(result.decision)
            self.assertFalse((output / "snapshot.json").exists())
            self.assertEqual(len(requests), 1)

    def test_transport_cannot_publish_schema_invalid_content(self) -> None:
        """Plausible output with an extra property fails after bounded correction."""
        invalid = understanding_proposal()
        invalid["unsupported_target"] = "VS"
        client, _ = self.serve([self.envelope(canonical(invalid))] * 3)
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Knowledge(SNAPSHOT, packet().standard, RULES)
            self.addCleanup(knowledge.close)
            output = Path(directory) / "run"
            result = Session(packet(), knowledge, client, output).run()
            self.assertEqual(result.stop_reason, "understanding_correction_no_progress")
            self.assertEqual(result.calls, 2)
            self.assertEqual(
                json.loads((output / "02.understanding-validation.json").read_text())[
                    "code"
                ],
                "invalid_structured_output",
            )
            self.assertIsNone(result.understanding)
            self.assertFalse((output / "snapshot.json").exists())

    def test_external_endpoint_is_rejected_before_data_transfer(self) -> None:
        """A caller cannot silently move source data to another provider."""
        with self.assertRaises(WorkflowError):
            LocalClient(base_url="https://example.com")
