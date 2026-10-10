"""Test source/decision meaning boundaries with deliberately plausible wrong values."""

import unittest
from pathlib import Path

from pydantic import ValidationError
from support import ROOT, decision, packet, understanding

from pipeline.stage_b_mapper_module.annotation_workflow.common import (
    WorkflowError, parse)
from pipeline.stage_b_mapper_module.annotation_workflow.decision import \
    MappingDecision
from pipeline.stage_b_mapper_module.annotation_workflow.runtime import \
    schema_for
from pipeline.stage_b_mapper_module.annotation_workflow.source import (
    Material, MaterialStore, SourcePacket, load_stage_a)
from pipeline.stage_b_mapper_module.annotation_workflow.understanding import (
    CollectionUnderstanding, queries, validate_understanding)


class SourceTests(unittest.TestCase):
    """Exercise actual typed source boundaries instead of accepting hand-shaped mock rows."""

    def test_no_target_or_annotation_origin_in_understanding(self) -> None:
        """A target disposition or answer-origin source must fail at the boundary."""
        value = understanding().model_dump(mode="json")
        value["items"][0]["disposition"] = "not_submitted"
        with self.assertRaises(ValidationError):
            CollectionUnderstanding.model_validate(value)
        source = packet().model_dump(mode="json")
        source["objects"][0]["origin"] = "annotation"
        with self.assertRaises(ValidationError):
            SourcePacket.model_validate(source)

    def test_duplicate_scope_and_wrong_mode_rejected(self) -> None:
        """Repeated IDs and converting printed context into entered data cannot pass."""
        source = packet().model_dump(mode="json")
        source["objects"].append(source["objects"][1])
        with self.assertRaises(ValidationError):
            SourcePacket.model_validate(source)
        value = understanding().model_dump(mode="json")
        value["items"][1]["value_mode"] = "entered"
        with self.assertRaises(WorkflowError):
            validate_understanding(
                CollectionUnderstanding.model_validate(value), packet()
            )

    def test_unknown_can_trigger_work_without_seed_target(self) -> None:
        """A partial item still produces a real semantic query without target fields."""
        value = understanding()
        result = queries(value, packet())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].concept, "Systolic blood pressure")
        self.assertEqual(
            {item.name: item.value for item in result[0].context},
            # Preserve the actual source label even when the semantic concept
            # is shorter, so numeric/timepoint modifiers cannot be discarded.
            {
                "unit_for": "mmHg",
                "time_of": "Date of measurement",
                "source_label": "Systolic blood pressure",
            },
        )
        missing = CollectionUnderstanding(
            explanation={
                "summary": "The measurement concept cannot be determined without the missing title.",
                "evidence_refs": ["bp"],
                "uncertainties": ["The title is missing."],
            },
            items=(),
            bindings=(),
            conditions=value.conditions,
            material_unknowns=(
                {
                    "missing": "title",
                    "affected_refs": ("bp",),
                    "impact": "measurement concept",
                    "source_kind": "source",
                    "material_ref": "source:page",
                },
            ),
        )
        self.assertEqual(queries(missing, packet()), ())

    def test_real_source_loader_verifies_bytes_and_keeps_options(self) -> None:
        """The actual reviewed Stage A JSON/PDF/lock path produces the expected source."""
        folder = ROOT / "tests/fixtures/graphrag/vitals-date"
        result = load_stage_a(
            folder / "date-source.json",
            folder / "date-source.pdf",
            folder / "input-lock.json",
            ROOT,
            "study",
            packet().standard,
        )
        self.assertEqual(result[0].task_ref, "assessment-date")
        self.assertEqual(
            result[0].object("assessment-date").text, "Date of vital signs assessment"
        )
        self.assertFalse(result[0].study_standard_confirmed)

    def test_material_read_denies_other_study_and_kind(self) -> None:
        """Both material text and the index obey study isolation, including missing reads."""
        doc = Material(
            ref="rule:1",
            kind="study",
            study_id="A",
            text="private synthetic policy",
            source_ref=packet().objects[0].source_ref,
            origin="study_rule",
            approved=True,
        )
        store = MaterialStore((doc,))
        self.assertEqual(store.read("A", "rule:1", "study"), doc)
        self.assertEqual(store.index("B"), ())
        self.assertIsNone(store.read("A", "missing", "study"))
        for study, kind in [("B", "study"), ("A", "source")]:
            with self.assertRaises(WorkflowError):
                store.read(study, "rule:1", kind)


class DecisionTests(unittest.TestCase):
    """Business schema can represent branches and rejects ambiguous target grouping."""

    def test_condition_and_selector_are_not_interchangeable(self) -> None:
        """Source ID expressions cannot masquerade as target-variable predicates."""
        value = decision().model_dump(mode="json")
        value["annotations"][0]["record_selector"]["value"] = {
            "op": "eq",
            "source_id": "bp",
            "value": "SYSBP",
        }
        with self.assertRaises(ValidationError):
            MappingDecision.model_validate(value)

    def test_dangling_observation_duplicate_keys_and_states_fail(self) -> None:
        """Legal-looking records with incorrect grouping identities cannot be saved."""
        for mutate in ("dangling", "duplicate", "state"):
            value = decision().model_dump(mode="json")
            if mutate == "dangling":
                value["annotations"][0]["observation_key"] = "other"
            elif mutate == "duplicate":
                value["annotations"].append(value["annotations"][0])
            else:
                value["annotations"][0]["source_condition"]["status"] = "specified"
            with self.subTest(mutate=mutate), self.assertRaises(ValidationError):
                MappingDecision.model_validate(value)

    def test_nonfinite_and_duplicate_json_rejected(self) -> None:
        """JSON normalization cannot silently repair duplicate values or NaN."""
        for raw in ('{"items":[],"items":[]}', '{"items":NaN}'):
            with self.assertRaises(WorkflowError):
                parse(CollectionUnderstanding, raw)

    def test_finite_wire_has_resolvable_refs_without_cycles(self) -> None:
        """Walk every actual schema reference; dangling or recursive refs fail."""

        def walk(value: object, schema: dict, stack: tuple[str, ...]) -> None:
            """Follow references against the exported definitions and check provider keys."""
            if isinstance(value, list):
                for item in value:
                    walk(item, schema, stack)
            elif isinstance(value, dict):
                self.assertFalse(set(value) & {"oneOf", "uniqueItems", "propertyNames"})
                if "$ref" in value:
                    name = value["$ref"].split("/")[-1]
                    self.assertNotIn(name, stack)
                    walk(schema["$defs"][name], schema, (*stack, name))
                for key, child in value.items():
                    if key not in {"$defs", "$ref"}:
                        walk(child, schema, stack)

        for phase in ("understanding", "decision"):
            with self.subTest(phase=phase):
                schema = schema_for(phase)
                walk(schema, schema, ())
