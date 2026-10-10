"""Answer-separated source packets and authorized material reads."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..graphrag.conditions import Identifier
from ..graphrag.field_contracts import FrozenModel, SourceRef, StandardProfile
from ..graphrag.source_input import load_input_lock, load_verified_source_stage_a
from .common import WorkflowError, unique


class SourceObject(FrozenModel):
    """One original object; semantic roles and target choices are deliberately absent."""

    ref: Identifier
    kind: Literal["question", "field", "option", "context", "instruction"]
    text: str
    origin: Literal["blank_crf", "edc", "completion_guidance"]
    source_ref: SourceRef
    parent_ref: str | None = None
    context_ref: str | None = None
    entry_mode: Literal[
        "entered", "printed_context", "prefilled", "instructional_example", "unknown"
    ] = "unknown"
    datatype: str = "unknown"
    stored_code: str | None = None


class SourcePacket(FrozenModel):
    """Read visibility is broader than write permission; approval is never inferred."""

    schema_version: Literal["source-packet-v3"] = "source-packet-v3"
    task_ref: Identifier
    study_id: Identifier
    objects: tuple[SourceObject, ...] = Field(min_length=1)
    output_scope: tuple[Identifier, ...] = Field(min_length=1)
    standard: StandardProfile
    study_standard_confirmed: bool = False
    modality: Literal["reviewed_text", "text_and_image"] = "reviewed_text"

    @model_validator(mode="after")
    def references(self) -> SourcePacket:
        """Validate source graph membership without making clinical associations."""
        refs = tuple(item.ref for item in self.objects)
        unique(refs, "duplicate_source_ref")
        unique(self.output_scope, "duplicate_output_scope")
        if not set(self.output_scope) <= set(refs):
            raise WorkflowError("unknown_output_scope")
        for item in self.objects:
            if any(
                ref is not None and ref not in refs
                for ref in (item.parent_ref, item.context_ref)
            ):
                raise WorkflowError("unknown_source_relationship")
        return self

    def object(self, ref: str) -> SourceObject:
        """Resolve a visible original object; unknown or other-study handles fail."""
        for item in self.objects:
            if item.ref == ref:
                return item
        raise WorkflowError("unknown_source_ref")


class Material(FrozenModel):
    """Admitted source or study text, with explicit approval and source provenance."""

    ref: Identifier
    kind: Literal["source", "study", "dependency"]
    study_id: Identifier
    text: str
    source_ref: SourceRef
    origin: Literal["blank_crf", "edc", "completion_guidance", "study_rule"]
    approved: bool = False


class MaterialStore:
    """Hold caller-admitted materials; callers cannot request arbitrary paths."""

    def __init__(self, materials: tuple[Material, ...] = ()) -> None:
        """Reject duplicate handles and retain immutable documents."""
        unique(tuple(item.ref for item in materials), "duplicate_material_ref")
        self._materials = materials

    def read(self, study_id: str, ref: str, kind: str) -> Material | None:
        """Return an admitted same-study document; foreign documents are never disclosed."""
        for material in self._materials:
            if material.ref == ref:
                if material.study_id != study_id or material.kind != kind:
                    raise WorkflowError("material_scope_denied")
                return material
        return None

    def index(self, study_id: str) -> tuple[dict[str, str], ...]:
        """Expose only same-study handles; document text is delivered on demand."""
        return tuple(
            {"ref": item.ref, "kind": item.kind}
            for item in self._materials
            if item.study_id == study_id
        )


def group_packets(task_ref: str, packets: tuple[SourcePacket, ...]) -> SourcePacket:
    """Combine explicitly selected source tasks without inferring hidden relationships.

    Identical shared source objects remain single objects. Conflicting objects or
    study/profile boundaries are rejected; no annotation source enters the group.
    """
    if not packets:
        raise WorkflowError("empty_source_group")
    first = packets[0]
    objects: dict[str, SourceObject] = {}
    scope: list[str] = []
    for packet in packets:
        if (
            packet.study_id != first.study_id
            or packet.standard != first.standard
            or packet.modality != first.modality
        ):
            raise WorkflowError("source_group_scope_denied")
        for item in packet.objects:
            if item.ref in objects and item != objects[item.ref]:
                raise WorkflowError("source_group_object_conflict")
            objects[item.ref] = item
        scope.extend(ref for ref in packet.output_scope if ref not in scope)
    return SourcePacket(
        task_ref=task_ref,
        study_id=first.study_id,
        objects=tuple(objects.values()),
        output_scope=tuple(scope),
        standard=first.standard,
        study_standard_confirmed=all(
            packet.study_standard_confirmed for packet in packets
        ),
        modality=first.modality,
    )


def load_stage_a(
    page: Path,
    pdf: Path,
    lock: Path,
    root: Path,
    study_id: str,
    standard: StandardProfile,
) -> tuple[SourcePacket, ...]:
    """Adapt verified tagged Stage A fields without importing target annotations.

    Unsupported fallback shapes are rejected by the existing source parser; no
    guessed conversion, target-standard approval or model call occurs here.
    """
    fields = load_verified_source_stage_a(
        page, pdf, study_id=study_id, lock=load_input_lock(lock), input_root=root
    )
    packets = []
    for field in fields:
        objects = [
            SourceObject(
                ref=field.field_id,
                kind="field",
                text=field.label,
                origin="blank_crf",
                source_ref=field.source_refs[0],
                datatype=field.input_type,
                entry_mode="entered",
            )
        ]
        for option in field.options:
            objects.append(
                SourceObject(
                    ref=option.id,
                    kind="option",
                    text=option.label,
                    parent_ref=field.field_id,
                    origin="blank_crf",
                    source_ref=field.source_refs[0],
                    entry_mode="entered",
                )
            )
        for related in field.related_fields:
            objects.append(
                SourceObject(
                    ref=related.id,
                    kind="context",
                    text=related.label,
                    origin="blank_crf",
                    source_ref=related.source_ref,
                )
            )
        for index, text in enumerate(
            tuple(filter(None, (field.form, *field.section_path, *field.units)))
        ):
            objects.append(
                SourceObject(
                    ref=f"{field.field_id}:context:{index}",
                    kind="context",
                    text=text,
                    origin="blank_crf",
                    source_ref=field.source_refs[0],
                    entry_mode="printed_context",
                )
            )
        packets.append(
            SourcePacket(
                task_ref=field.field_id,
                study_id=study_id,
                objects=tuple(objects),
                output_scope=(field.field_id, *(item.id for item in field.options)),
                standard=standard,
            )
        )
    return tuple(packets)
