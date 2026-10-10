"""Prove native-wire compilation preserves facts and all uncertainty states.

Unused payloads cannot silently be discarded as a plausible different mapping.
These probes use synthetic target strings without any clinical reference files.
"""

import copy
import unittest

from pydantic import ValidationError

from pipeline.stage_b_mapper_module.annotation_workflow.standard_preflight import (
    examples,
)
from pipeline.stage_b_mapper_module.annotation_workflow.standard_wire import (
    WireAnnotation,
    WireRecord,
    WireValue,
)


class StandardWireTests(unittest.TestCase):
    """Validate output representation independently of retrieval or model success."""

    def test_every_value_record_and_decision_variant_compiles_without_new_facts(self):
        for raw in examples():
            with self.subTest(status=raw["decision"], facts=raw["facts"]):
                wire = WireAnnotation.model_validate(raw)
                semantic = wire.represented().model_dump(mode="json")
                assert semantic["status"] == raw["decision"]
                assert semantic["missing"] == raw["missing"]
                assert semantic["explanation"] == raw["reason"]
                assert len(semantic["facts"]) == len(raw["facts"])
                if raw["facts"]:
                    a, b = raw["facts"][0], semantic["facts"][0]
                    assert a["domain"] == b["domain"] and a["variable"] == b["variable"]
                    assert a["value"]["kind"] == b["value"]["kind"]
                    assert a["record"]["kind"] == b["record"]["kind"]
                    assert a["when"] == b["when"]
                    if a["value"]["kind"].startswith("iso8601"):
                        assert b["value"]["field_ids"] == a["value"]["source_fields"]
                    elif a["value"]["kind"] == "constant":
                        assert b["value"]["value"] == a["value"]["literal"]
                    else:
                        assert b["value"]["field_id"] == a["value"]["source_field"]

    def test_inactive_payload_is_rejected_instead_of_ignored(self):
        raw = copy.deepcopy(examples()[0]["facts"][0]["value"])
        raw["literal"] = "invented constant"
        with self.assertRaises(ValidationError):
            WireValue.model_validate(raw)
        raw = copy.deepcopy(examples()[0]["facts"][0]["record"])
        raw["test_code"] = "invented fixed test"
        with self.assertRaises(ValidationError):
            WireRecord.model_validate(raw)

    def test_mapped_without_facts_fails_business_shape_during_parse(self):
        raw = copy.deepcopy(examples()[0])
        raw["facts"] = []
        with self.assertRaises(ValidationError):
            WireAnnotation.model_validate(raw)

    def test_missing_lookup_source_reports_actionable_slot_constraint(self):
        # Correction must explain the broken variant without suggesting any
        # clinical target or expected answer from the benchmark.
        raw = copy.deepcopy(examples()[1]["facts"][0]["value"])
        raw["source_field"] = None
        with self.assertRaises(ValidationError) as caught:
            WireValue.model_validate(raw)
        assert "ct_lookup requires source_field" in str(caught.exception)


if __name__ == "__main__":
    unittest.main()
