"""Immutable graph contracts and validation at the offline build boundary."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sdtm_knowledge_base.cosmos.source import COMMIT, version_tuple

ROOT = Path(__file__).resolve().parents[3]
BUILDER_VERSION = "acrf-source-graph-v1"
LEGACY_FILES = ("metadata.json", "domains.json", "variables_all.json", "cdisc_ct.json")


class GraphError(ValueError):
    """Carry a stable failure code without logging source content or patient data."""

    def __init__(self, code: str):
        """Initialize one safe public error code; no source payload is accepted."""
        self.code = code
        super().__init__(code)


def canonical(value: object) -> str:
    """Serialize deterministic JSON; reject unsupported values and nonfinite floats."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (ValueError, TypeError) as exc:
        raise GraphError("invalid_json_value") from exc


def digest(content: bytes) -> str:
    """Return a content hash; callers must supply the exact bytes being retained."""
    return hashlib.sha256(content).hexdigest()


def object_value(value: object) -> dict[str, object]:
    """Require a JSON object with string keys; return a detached top-level mapping."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise GraphError("expected_json_object")
    return dict(value)


def array_value(value: object) -> list[object]:
    """Require a JSON array; return a detached list without inventing missing data."""
    if not isinstance(value, list):
        raise GraphError("expected_json_array")
    return list(value)


def text_value(value: object, *, empty: bool = False) -> str:
    """Require a string and, for identities, nonempty text; do not coerce types."""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise GraphError("expected_nonempty_string")
    return value


def load_json(content: bytes | str) -> object:
    """Parse finite JSON at the boundary and reject malformed external input."""
    try:
        value: object = json.loads(content)
        canonical(value)
        return value
    except (ValueError, UnicodeError, TypeError) as exc:
        raise GraphError("invalid_json") from exc


@dataclass(frozen=True)
class StandardProfile:
    """An explicit knowledge build scope, never an inferred study configuration."""

    sdtm: str = "2.0"
    sdtmig: str = "3.4"
    ct_package: str = "sdtmct-2025-03-28"
    cosmos_commit: str = COMMIT
    include_drafts: bool = False

    def __post_init__(self) -> None:
        """Reject floating versions, other COSMoS snapshots and coerced draft flags."""
        try:
            text_value(self.sdtm)
            text_value(self.sdtmig)
            version_tuple(self.sdtm)
            version_tuple(self.sdtmig)
        except ValueError as exc:
            raise GraphError("invalid_standard_version") from exc
        if self.cosmos_commit != COMMIT or type(self.include_drafts) is not bool:
            raise GraphError("invalid_graph_profile")
        if not isinstance(self.ct_package, str) or not self.ct_package.startswith(
            "sdtmct-"
        ):
            raise GraphError("invalid_ct_package")

    def to_dict(self) -> dict[str, object]:
        """Return a detached serializable profile for manifests and exact comparisons."""
        return asdict(self)


@dataclass(frozen=True)
class SourceDocument:
    """Exact source bytes, identity and authority; hash validation is not endorsement."""

    path: str
    body: bytes
    authority: str
    license: str
    _id: str = field(init=False, repr=False)
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Hash large source files once per build, not once per node or edge."""
        object.__setattr__(self, "_id", digest(self.path.encode() + b"\0" + self.body))
        object.__setattr__(self, "_sha256", digest(self.body))

    @property
    def id(self) -> str:
        """Scope source identity by path and bytes so two aliases never collide."""
        return self._id

    @property
    def sha256(self) -> str:
        """Hash the preserved bytes, independent of their source path."""
        return self._sha256


@dataclass(frozen=True)
class Node:
    """One immutable query result; nested attributes are decoded into a fresh copy."""

    id: str
    kind: str
    source_key: str
    owner_id: str
    source_id: str
    locator: str
    status: str
    attrs_json: str

    @property
    def attrs(self) -> dict[str, object]:
        """Decode attributes without exposing shared mutable database state."""
        return object_value(load_json(self.attrs_json))
