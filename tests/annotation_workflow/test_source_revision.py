"""Single-choice revision identity must not broaden unrelated write permission.

Synthetic drafts exercise repair bookkeeping only; accepting a revision is not
evidence that its selected template is clinically applicable.
"""

from __future__ import annotations

import copy
import unittest

from test_source_identity_regression import choice_packet, draft

from pipeline.stage_b_mapper_module.annotation_workflow.common import WorkflowError
from pipeline.stage_b_mapper_module.annotation_workflow.session import (
    _preserve_unaffected,
    _revision_changes,
)
from pipeline.stage_b_mapper_module.annotation_workflow.source import SourcePacket


def child_draft() -> dict:
    """Keep siblings distinct so permission on one cannot hide edits to the other."""
    raw = draft("option-a")
    raw["records"].append(draft("option-b")["records"][0])
    for record in raw["records"]:
        record["bindings"][0]["details"] = {"support": ["measurement"]}
    return raw


def changed_child(previous: dict, index: int = 1) -> dict:
    current = copy.deepcopy(previous)
    current["records"][index]["bindings"][0]["details"]["support"] = [
        "question",
        "measurement",
    ]
    current["records"][index][
        "reason"
    ] = "The reviewed question supplies additional support for this option."
    return current


class SourceRevisionTests(unittest.TestCase):
    def assert_protected(
        self,
        before: dict,
        after: dict,
        affected: set[str],
        source: SourcePacket,
        protected_ref: str = "option-b",
    ) -> None:
        with self.assertRaises(WorkflowError) as raised:
            _preserve_unaffected(before, after, affected, source)
        self.assertEqual(raised.exception.code, "unaffected_judgment_changed")
        self.assertEqual(
            {
                ref
                for item in raised.exception.feedback
                for refs in (item["affected_refs"],)
                if isinstance(refs, list)
                for ref in refs
            },
            {protected_ref},
        )

    def test_parent_to_all_options_with_changed_reason_is_audited(self) -> None:
        for datatype in ("radio", "single_choice"):
            with self.subTest(datatype=datatype):
                before = draft("question")
                after = draft("option-a", "option-b")
                after["records"][0][
                    "reason"
                ] = "The complete option set is the reviewed representation of this question."
                originals = copy.deepcopy((before, after))
                changes = _revision_changes(before, after, choice_packet(datatype))
                self.assertEqual(
                    changes,
                    [
                        {
                            "source": ["question"],
                            "previous": before["records"][0],
                            "replacement": after["records"],
                        }
                    ],
                )
                self.assertEqual((before, after), originals)

    def test_family_replacement_still_requires_a_changed_reason(self) -> None:
        before = draft("question")
        after = draft("option-a", "option-b")
        with self.assertRaises(WorkflowError) as raised:
            _revision_changes(before, after, choice_packet())
        self.assertEqual(raised.exception.code, "revision_reason_required")

    def test_without_source_graph_no_parent_option_identity_is_invented(self) -> None:
        before = draft("question")
        after = draft("option-a", "option-b")
        after["records"][0]["reason"] = "The proposed representation changed."
        with self.assertRaises(WorkflowError) as raised:
            _revision_changes(before, after)
        self.assertEqual(raised.exception.code, "revision_reason_required")

    def test_affected_parent_allows_child_support_revision(self) -> None:
        for datatype in ("radio", "single_choice"):
            with self.subTest(datatype=datatype):
                before = child_draft()
                after = changed_child(before)
                _preserve_unaffected(
                    before, after, {"question"}, choice_packet(datatype)
                )
                changes = _revision_changes(before, after, choice_packet(datatype))
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0]["source"], ["option-b"])
                self.assertEqual(after["records"][0], before["records"][0])

    def test_one_option_can_change_itself_but_not_its_sibling(self) -> None:
        before = child_draft()
        _preserve_unaffected(
            before, changed_child(before, 0), {"option-a"}, choice_packet()
        )
        self.assert_protected(
            before, changed_child(before), {"option-a"}, choice_packet()
        )

    def test_checkbox_parent_does_not_authorize_option_revision(self) -> None:
        before = child_draft()
        self.assert_protected(
            before, changed_child(before), {"question"}, choice_packet("checkbox")
        )

    def test_readonly_context_or_option_cannot_authorize_sibling_revision(self) -> None:
        source = choice_packet()
        readonly = source.object("option-a").model_copy(
            update={"ref": "readonly-option"}
        )
        source = source.model_copy(update={"objects": (*source.objects, readonly)})
        before = child_draft()
        for affected in ({"measurement"}, {"readonly-option"}):
            with self.subTest(affected=affected):
                self.assert_protected(before, changed_child(before), affected, source)

    def test_parent_expansion_excludes_readonly_option(self) -> None:
        """An invalid earlier draft does not grant permission to edit an excluded ref."""
        source = choice_packet().model_copy(
            update={"output_scope": ("question", "option-a")}
        )
        before = child_draft()
        self.assert_protected(before, changed_child(before), {"question"}, source)

    def test_readonly_parent_does_not_authorize_writable_options(self) -> None:
        source = choice_packet().model_copy(
            update={"output_scope": ("option-a", "option-b")}
        )
        before = child_draft()
        self.assert_protected(before, changed_child(before), {"question"}, source)


if __name__ == "__main__":
    unittest.main()
