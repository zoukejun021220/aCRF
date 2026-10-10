"""Expose the legacy mapper lazily so offline knowledge tools need no models."""

__all__ = ["UnifiedSDTMMapper"]


def __getattr__(name: str) -> object:
    """Load the legacy public class on demand; propagate its dependency failures."""
    if name == "UnifiedSDTMMapper":
        from .core.mapper import UnifiedSDTMMapper

        return UnifiedSDTMMapper
    raise AttributeError(name)
