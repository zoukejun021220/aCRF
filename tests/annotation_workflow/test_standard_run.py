"""Check bounded correction and exact semantic export with public-only inputs.

The model stub replaces only inference. The real retriever, validator, exporter
and trace store execute against a pinned public collection without gold files.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.stage_b_mapper_module.annotation_workflow.runtime import Reply
from pipeline.stage_b_mapper_module.annotation_workflow.standard_annotation import (
    run_standard,
)
from tests.annotation_workflow.test_standard_context import (
    ROOT,
    StandardsCorpus,
    source_for,
)


class StandardRunTests(unittest.TestCase):
    """A wrong but plausible answer cannot be committed as a valid mapping."""

    def setUp(self):
        self.corpus = StandardsCorpus(
            ROOT / "kb/standard_mapping_cdash23_sdtmig34_20261004"
        )
        group = next(
            g
            for g in self.corpus.groups
            if any(f.get("name") == "AEACNOYN" for f in g["fields"])
        )
        self.case = {"case_id": "synthetic", "source": source_for(group, "AEACNOYN")}
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "trace"

    def client(self, responses):
        class Stub:
            """Record exactly what the real workflow delivers to the provider."""

            def __init__(self):
                self.requests = []

            def generate(self, messages, schema, max_tokens):
                self.requests.append((messages, schema, max_tokens))
                return Reply(
                    content=responses[min(len(self.requests) - 1, len(responses) - 1)],
                    error=None,
                    status=200,
                    finish_reason="stop",
                    prompt_tokens=1200,
                    completion_tokens=80,
                    reasoning_present=False,
                    response_sha256=None,
                    seconds=0.01,
                )

        return Stub()

    def positive(self):
        return json.dumps(
            {
                "decision": "not_submitted",
                "facts": [],
                "missing": [],
                "not_submitted_when": [],
                "reason": "The public mapping instruction explicitly states NOT SUBMITTED.",
                "evidence_refs": ["E1"],
            }
        )

    def test_correction_preserves_real_traces_and_exact_export(self):
        client = self.client(['{"status":"invented"}', self.positive()])
        result = run_standard(self.case, self.path, client, self.corpus)
        assert result["outcome"]["calls"] == 2
        assert result["outcome"]["measured_tokens"] == 2560
        assert (self.path / "01.validation.json").exists()
        answer = json.loads((self.path / "standard-answer.json").read_text())
        assert answer["status"] == "not_submitted" and not answer["facts"]
        assert "evidence_refs" not in answer
        first = json.loads(client.requests[0][0][1]["content"])
        assert first["context"]["source"] == self.case["source"]
        assert first["context"]["evidence"][0]["collection_field"] == "AEACNOYN"
        assert first["task_mode"] == "standards_only"
        assert "accepted" not in first and "reference" not in first

    def test_invalid_replies_stop_without_fabricating_facts(self):
        client = self.client(['{"invalid":"plausible text is not a proposal"}'])
        result = run_standard(self.case, self.path, client, self.corpus)
        assert result["outcome"]["calls"] == 3
        assert result["outcome"]["stop_reason"] == "correction_no_progress"
        assert not result["prediction"]["facts"]
        assert (
            json.loads((self.path / "standard-answer.json").read_text())["status"]
            == "abstain"
        )

    def test_correction_delivers_validation_path_without_expected_answer(self):
        """Reproduce the live repeated-failure cause using a synthetic bad value.

        A generic invalid-output label hides which slot failed. Send the existing
        parser's safe field path/message; never add reference-derived facts.
        """
        invalid = {
            "decision": "mapped",
            "evidence_refs": ["E1"],
            "facts": [
                {
                    "domain": "AE",
                    "variable": "AEACN",
                    "record": {
                        "kind": "collection_record",
                        "name": None,
                        "test_code": None,
                        "test_source_field": None,
                    },
                    "value": {
                        "kind": "ct_lookup",
                        "literal": None,
                        "source_field": None,
                        "source_fields": [],
                    },
                    "when": None,
                }
            ],
            "missing": [],
            "not_submitted_when": [],
            "reason": "Synthetic invalid payload exercises correction feedback.",
        }
        client = self.client([json.dumps(invalid), self.positive()])
        run_standard(self.case, self.path, client, self.corpus)
        correction = json.loads(client.requests[1][0][-1]["content"])
        # The path is available from schema validation independently of any
        # benchmark, so its presence proves an actionable host-side correction.
        assert "/facts/0/value" in json.dumps(correction["correction"])
        assert "expected_answer" not in correction


if __name__ == "__main__":
    unittest.main()
