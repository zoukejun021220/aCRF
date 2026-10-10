"""Validate reviewed Stage A JSON and project it into source-safe field contexts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import NoReturn, cast

from pydantic import ValidationError

from .field_contracts import (
    SAFE_ID_PATTERN,
    FieldContext,
    GraphRagError,
    InputLock,
    InputType,
    Option,
    PageFileHash,
    RelatedField,
    SourceFieldContext,
    SourceRef,
    StudyProfile,
    is_answer_path_part,
)

PAGE_KEYS = frozenset({"page_id", "items"})
ITEM_KEYS = frozenset(
    {
        "bbox_ref",
        "bbox_xyxy",
        "form",
        "input_type",
        "page",
        "parent_qid",
        "qid",
        "question_below_text",
        "section",
        "seq",
        "subsection",
        "tag",
        "text",
    }
)
SUPPORTED_TAGS = frozenset({"<Q>", "<INPUT>", "<SH>", "<FORM>"})
INPUT_TYPES: dict[str, InputType] = {
    "text": "text",
    "free": "text",
    "textarea": "text",
    "number": "number",
    "numeric": "number",
    "date": "date",
    "radio": "radio",
    "option": "radio",
    "checkbox": "checkbox",
    "mixed": "mixed",
    "unknown": "unknown",
}
UNIT_LABELS = frozenset(
    {
        "%",
        "bpm",
        "c",
        "cm",
        "deg c",
        "f",
        "g",
        "kg",
        "l",
        "lb",
        "mcg",
        "mg",
        "ml",
        "mm",
        "mmhg",
    }
)


def _safe_item_id(value: object, index: int) -> str:
    """Return a source ID only when safe for errors; otherwise return its position."""
    if isinstance(value, str) and SAFE_ID_PATTERN.fullmatch(value):
        return value
    return f"item:{index}"


def _fail(*ids: str) -> NoReturn:
    """Raise the stable input error while excluding source text and unsafe values."""
    raise GraphRagError("input_invalid", tuple(ids))


def _validate_bbox(value: object, identifier: str) -> tuple[float, float, float, float]:
    """Return one finite normalized ordered bounding box or fail with a safe ID."""
    if not isinstance(value, list) or len(value) != 4:
        _fail(identifier)
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            _fail(identifier)
        if coordinate < 0.0 or coordinate > 1.0:
            _fail(identifier)
        number = float(coordinate)
        if not math.isfinite(number):
            _fail(identifier)
        coordinates.append(number)
    x1, y1, x2, y2 = coordinates
    if x1 > x2 or y1 > y2:
        _fail(identifier)
    return x1, y1, x2, y2


def _validate_location(item: dict[str, object], identifier: str) -> tuple[int, str]:
    """Validate a 1-based page and matched bbox reference for honest source locators."""
    page = item.get("page")
    bbox_ref = item.get("bbox_ref")
    if (
        isinstance(page, bool)
        or not isinstance(page, int)
        or page < 1
        or not isinstance(bbox_ref, str)
        or not bbox_ref
        or bbox_ref.casefold() in {"unmatched", "orphan", "unknown"}
    ):
        _fail(identifier)
    return page, bbox_ref


def _validated_items(page: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Validate the explicit Stage A allowlist and detach every accepted item."""
    if set(page) - PAGE_KEYS or not isinstance(page.get("page_id"), int):
        _fail("page")
    raw_items = page.get("items")
    if not isinstance(raw_items, list):
        _fail("page")
    validated: list[dict[str, object]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict) or not all(
            isinstance(key, str) for key in raw_item
        ):
            _fail(f"item:{index}")
        item = cast(dict[str, object], raw_item)
        identifier = _safe_item_id(item.get("qid") or item.get("parent_qid"), index)
        if set(item) - ITEM_KEYS:
            _fail(identifier)
        tag = item.get("tag")
        text = item.get("text")
        seq = item.get("seq")
        if (
            tag not in SUPPORTED_TAGS
            or not isinstance(text, str)
            or isinstance(seq, bool)
            or not isinstance(seq, int)
        ):
            _fail(identifier)
        for optional_text in ("form", "section", "subsection"):
            if optional_text in item and not isinstance(
                item[optional_text], (str, type(None))
            ):
                _fail(identifier)
        if tag in {"<Q>", "<INPUT>"}:
            _validate_location(item, identifier)
            _validate_bbox(item.get("bbox_xyxy"), identifier)
        validated.append(dict(item))
    return tuple(validated)


def _locator(item: dict[str, object], identifier: str) -> str:
    """Build a path-free locator from a validated page, bbox, and Stage A reference."""
    page, bbox_ref = _validate_location(item, identifier)
    bbox = _validate_bbox(item.get("bbox_xyxy"), identifier)
    serialized_bbox = ",".join(format(value, ".8g") for value in bbox)
    return f"page={page};bbox={serialized_bbox};bbox_ref={bbox_ref}"


def _source_ref(item: dict[str, object], identifier: str, lock: InputLock) -> SourceRef:
    """Create an unverified extraction reference bound to original source bytes."""
    return SourceRef(
        source_id=lock.source_sha256,
        sha256=lock.source_sha256,
        locator=_locator(item, identifier),
        authority="unverified",
    )


def _section_path(item: dict[str, object]) -> tuple[str, ...]:
    """Split explicit Stage A section hierarchy and append a declared subsection."""
    section = item.get("section")
    subsection = item.get("subsection")
    parts = (
        tuple(part.strip() for part in section.split(">") if part.strip())
        if isinstance(section, str)
        else ()
    )
    if isinstance(subsection, str) and subsection.strip():
        return (*parts, subsection.strip())
    return parts


def _is_unit_control(item: dict[str, object]) -> bool:
    """Recognize only a standalone known unit label as a source qualifier."""
    label = item.get("text")
    raw_type = item.get("input_type")
    return (
        isinstance(label, str)
        and isinstance(raw_type, str)
        and raw_type.casefold() in {"text", "free", "textarea", "unknown"}
        and label.strip().casefold() in UNIT_LABELS
    )


def _input_type(items: tuple[dict[str, object], ...], field_id: str) -> InputType:
    """Combine explicit non-unit control types without inferring from labels or IDs."""
    types: list[InputType] = []
    for index, item in enumerate(items):
        if _is_unit_control(item):
            continue
        raw_type = item.get("input_type")
        if not isinstance(raw_type, str) or raw_type.casefold() not in INPUT_TYPES:
            _fail(_safe_item_id(item.get("parent_qid"), index))
        normalized = INPUT_TYPES[raw_type.casefold()]
        if normalized not in types:
            types.append(normalized)
    if not types:
        return "unknown"
    if len(types) == 1:
        return types[0]
    return "mixed"


def _options(items: tuple[dict[str, object], ...], field_id: str) -> tuple[Option, ...]:
    """Project explicit option controls into deterministic source-derived identities."""
    options: list[Option] = []
    for index, item in enumerate(items):
        raw_type = item.get("input_type")
        label = item.get("text")
        if (
            isinstance(raw_type, str)
            and raw_type.casefold() in {"option", "radio", "checkbox"}
            and isinstance(label, str)
            and label.strip()
        ):
            seq = item.get("seq")
            options.append(Option(id=f"{field_id}:option:{seq}", label=label))
    return tuple(options)


def _related_fields(
    items: tuple[dict[str, object], ...], field_id: str, lock: InputLock
) -> tuple[RelatedField, ...]:
    """Project unit labels and retain other reviewed text as navigation context."""
    related: list[RelatedField] = []
    for index, item in enumerate(items):
        label = cast(str, item["text"])
        raw_type = cast(str, item["input_type"]).casefold()
        if not label.strip() or raw_type in {"option", "radio", "checkbox"}:
            continue
        seq = item.get("seq")
        identifier = _safe_item_id(item.get("parent_qid"), index)
        related.append(
            RelatedField(
                id=f"{field_id}:related:{seq}",
                label=label,
                relation="unit" if _is_unit_control(item) else "navigation_context",
                source_ref=_source_ref(item, identifier, lock),
            )
        )
    return tuple(related)


def _source_contexts(
    page: dict[str, object], study_id: str, lock: InputLock
) -> tuple[SourceFieldContext, ...]:
    """Validate grouping globally and return only explicitly reviewed field IDs."""
    if not lock.answer_free_reviewed:
        raise GraphRagError("input_not_reviewed")
    items = _validated_items(page)
    if not items:
        raise GraphRagError("input_empty")

    questions: dict[str, dict[str, object]] = {}
    question_order: list[str] = []
    inputs: dict[str, list[dict[str, object]]] = {}
    for index, item in enumerate(items):
        if item["tag"] != "<Q>":
            continue
        field_id = item.get("qid")
        identifier = _safe_item_id(field_id, index)
        if (
            not isinstance(field_id, str)
            or SAFE_ID_PATTERN.fullmatch(field_id) is None
            or field_id in questions
            or not cast(str, item["text"]).strip()
        ):
            _fail(identifier)
        questions[field_id] = item
        question_order.append(field_id)
        inputs[field_id] = []

    for index, item in enumerate(items):
        if item["tag"] != "<INPUT>":
            continue
        parent = item.get("parent_qid")
        identifier = _safe_item_id(parent, index)
        if (
            not isinstance(parent, str)
            or SAFE_ID_PATTERN.fullmatch(parent) is None
            or parent == "orphan"
            or parent not in questions
        ):
            _fail(identifier)
        raw_type = item.get("input_type")
        if not isinstance(raw_type, str) or raw_type.casefold() not in INPUT_TYPES:
            _fail(identifier)
        inputs[parent].append(item)

    allowed_ids = {
        field_id for page_hash in lock.page_hashes for field_id in page_hash.field_ids
    }
    selected = [field_id for field_id in question_order if field_id in allowed_ids]
    if not selected:
        raise GraphRagError("input_empty")

    contexts: list[SourceFieldContext] = []
    for field_id in selected:
        question = questions[field_id]
        grouped_inputs = tuple(inputs[field_id])
        question_ref = _source_ref(question, field_id, lock)
        input_refs = tuple(
            _source_ref(item, _safe_item_id(item.get("parent_qid"), index), lock)
            for index, item in enumerate(grouped_inputs)
        )
        related = _related_fields(grouped_inputs, field_id, lock)
        form = question.get("form")
        contexts.append(
            SourceFieldContext(
                study_id=study_id,
                source_sha256=lock.source_sha256,
                page_number=cast(int, question["page"]),
                field_id=field_id,
                label=cast(str, question["text"]),
                form=cast(str | None, form),
                section_path=_section_path(question),
                input_type=_input_type(grouped_inputs, field_id),
                options=_options(grouped_inputs, field_id),
                units=tuple(
                    field.label for field in related if field.relation == "unit"
                ),
                related_fields=related,
                source_refs=(question_ref, *input_refs),
            )
        )
    return tuple(contexts)


def parse_source_stage_a(
    page: dict[str, object], *, study_id: str, lock: InputLock
) -> tuple[SourceFieldContext, ...]:
    """Parse reviewed Stage A data without assigning an unresolved target standard."""
    if not isinstance(study_id, str) or not study_id.strip():
        raise GraphRagError("study_identity_invalid")
    try:
        return _source_contexts(page, study_id, lock)
    except GraphRagError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise GraphRagError("input_invalid") from exc


def parse_stage_a(
    page: dict[str, object], profile: StudyProfile, lock: InputLock
) -> tuple[FieldContext, ...]:
    """Parse reviewed Stage A data only after study target-version evidence exists."""
    if profile.standard is None or not any(
        evidence.authority == "study_approved" for evidence in profile.version_evidence
    ):
        raise GraphRagError("study_version_unverified")
    source_fields = parse_source_stage_a(page, study_id=profile.study_id, lock=lock)
    return tuple(
        FieldContext(
            **field.model_dump(mode="python"), target_standard=profile.standard
        )
        for field in source_fields
    )


def _relative_page_path(page_path: Path, input_root: Path) -> str:
    """Return a normalized relative path or fail without disclosing filesystem paths."""
    try:
        relative = page_path.resolve(strict=True).relative_to(
            input_root.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise GraphRagError("page_path_unapproved") from exc
    if any(is_answer_path_part(part) for part in relative.parts):
        raise GraphRagError("page_path_unapproved")
    return relative.as_posix()


def _locked_page(path: str, lock: InputLock) -> PageFileHash:
    """Resolve an approved path to its single byte hash and field allowlist record."""
    matches = tuple(
        page_hash for page_hash in lock.page_hashes if page_hash.path == path
    )
    if path not in lock.allowed_page_files or len(matches) != 1:
        raise GraphRagError("page_path_unapproved")
    return matches[0]


def _verified_page(
    page_path: Path, source_path: Path, lock: InputLock, input_root: Path
) -> tuple[dict[str, object], InputLock]:
    """Read each file once, verify exact hashes, and return page-specific lock scope."""
    relative_path = _relative_page_path(page_path, input_root)
    page_lock = _locked_page(relative_path, lock)
    try:
        if any(is_answer_path_part(part) for part in source_path.resolve().parts):
            raise GraphRagError("source_path_unapproved")
        page_bytes = page_path.read_bytes()
        source_bytes = source_path.read_bytes()
    except GraphRagError:
        raise
    except OSError as exc:
        raise GraphRagError("source_read_failed") from exc
    if hashlib.sha256(page_bytes).hexdigest() != page_lock.sha256:
        raise GraphRagError("page_hash_mismatch")
    if hashlib.sha256(source_bytes).hexdigest() != lock.source_sha256:
        raise GraphRagError("source_hash_mismatch")
    try:
        decoded = json.loads(page_bytes, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GraphRagError("input_invalid") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise GraphRagError("input_invalid")
    scoped_lock = InputLock(
        source_sha256=lock.source_sha256,
        answer_free_reviewed=lock.answer_free_reviewed,
        review_record_id=lock.review_record_id,
        allowed_page_files=(page_lock.path,),
        page_hashes=(page_lock,),
    )
    return cast(dict[str, object], decoded), scoped_lock


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON NaN and infinity tokens before boundary validation."""
    raise ValueError("nonfinite_json_value")


def load_input_lock(path: Path) -> InputLock:
    """Load and strictly validate a source lock without exposing its path or values."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise GraphRagError("input_lock_read_failed") from exc
    try:
        return InputLock.model_validate_json(content)
    except (ValueError, ValidationError) as exc:
        raise GraphRagError("input_lock_invalid") from exc


def load_verified_source_stage_a(
    page_path: Path,
    source_path: Path,
    *,
    study_id: str,
    lock: InputLock,
    input_root: Path,
) -> tuple[SourceFieldContext, ...]:
    """Verify source/page bytes and return a version-neutral reviewed source view."""
    page, scoped_lock = _verified_page(page_path, source_path, lock, input_root)
    return parse_source_stage_a(page, study_id=study_id, lock=scoped_lock)


def load_verified_stage_a(
    page_path: Path,
    source_path: Path,
    profile: StudyProfile,
    lock: InputLock,
    *,
    input_root: Path,
) -> tuple[FieldContext, ...]:
    """Verify source/page bytes and return fields only with study version evidence."""
    page, scoped_lock = _verified_page(page_path, source_path, lock, input_root)
    return parse_stage_a(page, profile, scoped_lock)
