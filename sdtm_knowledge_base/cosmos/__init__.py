"""Versioned COSMoS metadata, usable offline without ML dependencies."""

from .catalog import CosmosCatalog
from .snapshot import DEFAULT_SNAPSHOT, build_snapshot, load_snapshot
from .source import CosmosRecord

__all__ = [
    "CosmosCatalog",
    "CosmosRecord",
    "DEFAULT_SNAPSHOT",
    "build_snapshot",
    "load_snapshot",
]
