"""Transport contracts preserve the complete schema without promoting reasoning."""

import json
import tempfile
import unittest
from pathlib import Path

from support import ScriptedModel

from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.preflight import (
    cases,
    run_preflight,
    valid_reply,
    validate_gate,
)
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import (
    LocalClient,
    Reply,
    schema_for,
)
from pipeline.stage_b_mapper_module.annotation_workflow.store import RunStore


class TransportTests(unittest.TestCase):
    """Check exact request behavior before model calls or subjective scores."""

    def test_decision_gate_exercises_actions_and_rejects_old_unwrapped_wire(
        self,
    ) -> None:
        schema = schema_for("decision")
        self.assertIn("action", schema["required"])
        self.assertNotIn("records", schema["properties"])
        probes = [expected for phase, expected in cases() if phase == "decision"]
        self.assertEqual(len(probes), 4)
        self.assertEqual(
            {expected["action"]["op"] for expected in probes},
            {"submit", "read_material", "search_standard"},
        )
        for expected in probes:
            reply = Reply(
                content=json.dumps(expected),
                error=None,
                status=200,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
                reasoning_present=False,
                response_sha256=None,
                seconds=0,
            )
            self.assertTrue(valid_reply(reply, "decision", expected))
            if expected["action"]["op"] == "submit":
                self.assertNotIn("open_questions", expected["action"])
                old = {
                    key: value
                    for key, value in expected["action"].items()
                    if key not in {"op", "question_updates"}
                }
                self.assertFalse(
                    valid_reply(
                        reply.model_copy(update={"content": json.dumps(old)}),
                        "decision",
                        expected,
                    )
                )

    def test_gate_checks_missing_action_probe_and_altered_raw_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "gate")
            model = ScriptedModel([json.dumps(expected) for _, expected in cases()])
            self.assertTrue(
                run_preflight(
                    model, store, {"code": "test"}, "local", "http://localhost:1234"
                )
            )
            path = store.path / "gate.json"
            validate_gate(path, {"code": "test"}, "local", "http://localhost:1234")
            gate = json.loads(path.read_text())
            self.assertEqual(len(gate["records"]), 9)
            missing = {**gate, "records": gate["records"][:-1]}
            path.write_text(json.dumps(missing))
            with self.assertRaises(WorkflowError):
                validate_gate(path, {"code": "test"}, "local", "http://localhost:1234")
            gate["records"][-1]["reply"]["content"] = json.dumps(
                {"action": {"op": "submit"}}
            )
            path.write_text(json.dumps(gate))
            with self.assertRaises(WorkflowError):
                validate_gate(path, {"code": "test"}, "local", "http://localhost:1234")

    def test_understanding_wire_and_probe_exercise_one_subject_without_mode(
        self,
    ) -> None:
        """A transport gate must exercise the changed nonempty item contract."""
        schema = schema_for("understanding")
        item = schema["$defs"]["InterpretedItem"]
        self.assertIn("source_ref", item["required"])
        self.assertNotIn("source_refs", item["properties"])
        self.assertNotIn("value_mode", item["properties"])
        probes = [expected for phase, expected in cases() if phase == "understanding"]
        self.assertEqual(len(probes), 2)
        for expected in probes:
            self.assertTrue(expected["items"])
            reply = Reply(
                content=json.dumps(expected),
                error=None,
                status=200,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
                reasoning_present=False,
                response_sha256=None,
                seconds=0,
            )
            self.assertTrue(valid_reply(reply, "understanding", expected))
            assert reply.content is not None
            raw = json.loads(reply.content)
            raw["items"][0]["value_mode"] = "entered"
            self.assertFalse(
                valid_reply(
                    reply.model_copy(update={"content": json.dumps(raw)}),
                    "understanding",
                    expected,
                )
            )

    def test_prompt_preserves_every_schema_and_does_not_mutate_messages(self) -> None:
        """Every deployed schema reaches normal generation intact; callers retain ownership."""
        for phase in ("probe", "understanding", "decision"):
            messages = [
                {"role": "system", "content": "Source interpretation."},
                {"role": "user", "content": "Synthetic input."},
            ]
            original = [dict(message) for message in messages]
            schema = schema_for(phase)
            body = LocalClient(schema_transport="prompt").request_body(
                messages, schema, 512
            )
            self.assertNotIn("response_format", body)
            delivered = json.loads(
                body["messages"][0]["content"].split("\nJSON_SCHEMA:\n", 1)[1]
            )
            self.assertEqual(delivered, schema)
            self.assertEqual(messages, original)
            self.assertEqual(body["messages"][1], original[1])

    def test_missing_system_message_gets_a_separate_format_contract(self) -> None:
        """User source text stays intact when the caller supplies no system message."""
        messages = [{"role": "user", "content": "Synthetic input."}]
        body = LocalClient(schema_transport="prompt").request_body(
            messages, schema_for("probe"), 512
        )
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1:], messages)
        self.assertEqual(len(messages), 1)

    def test_native_opt_in_retains_original_wire_contract(self) -> None:
        """The explicit native mode remains available for future provider diagnostics."""
        messages = [{"role": "system", "content": "Source interpretation."}]
        schema = schema_for("understanding")
        body = LocalClient(schema_transport="native").request_body(
            messages, schema, 512
        )
        self.assertEqual(body["messages"], messages)
        self.assertEqual(
            body["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "annotation_v3",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

    def test_unknown_mode_is_rejected(self) -> None:
        """Runtime configuration cannot silently fall back to a different transport."""
        with self.assertRaises(WorkflowError) as caught:
            LocalClient(schema_transport="invalid")  # type: ignore[arg-type]
        self.assertEqual(caught.exception.code, "invalid_schema_transport")

    def test_timeout_uses_remaining_budget_and_rejects_exhaustion(self) -> None:
        """Slow generation has headroom but never extends the caller's remaining budget."""
        client = LocalClient()
        self.assertEqual(client.bounded_timeout(900), 480)
        self.assertEqual(client.bounded_timeout(23), 23)
        for remaining in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(WorkflowError):
                client.bounded_timeout(remaining)
        for invalid in (0, 481, float("nan")):
            with self.assertRaises(WorkflowError):
                LocalClient(timeout=invalid)

    def test_default_disables_separate_thinking_without_dropping_explanations(
        self,
    ) -> None:
        """Final JSON keeps its required explanation despite disabling private thinking."""
        body = LocalClient().request_body([], schema_for("understanding"), 512)
        self.assertEqual(body["reasoning_effort"], "none")
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema, schema_for("understanding"))
        self.assertIn("explanation", schema["required"])
        self.assertEqual(
            LocalClient(reasoning_effort="low").request_body([], schema, 512)[
                "reasoning_effort"
            ],
            "low",
        )
