"""Exclusive run directories and atomic snapshots; automatic crash resume is absent."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .common import WorkflowError, canonical, digest


class RunStore:
    """A newly created directory is the exclusive session claim, never a resume hint."""

    def __init__(self, path: Path) -> None:
        """Refuse existing runs; create private artifacts without touching siblings."""
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=False)
        except OSError as exc:
            raise WorkflowError("run_directory_unavailable") from exc
        self.path = path
        self.revision = 0

    def write(self, name: str, value: object) -> None:
        """Atomically write one finite JSON artifact; failure leaves its previous bytes."""
        self.write_text(name, canonical(value) + "\n")

    def write_text(self, name: str, text: str) -> None:
        """Publish a private sibling temporary file only after flush/fsync succeeds."""
        if Path(name).name != name or name in {"", ".", ".."}:
            raise WorkflowError("invalid_artifact_name")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path, delete=False
            ) as stream:
                temporary = stream.name
                os.chmod(temporary, 0o600)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path / name)
        except OSError as exc:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            raise WorkflowError("artifact_write_failed") from exc

    def commit(
        self,
        phase: str,
        value: object,
        *,
        questions: object | None = None,
        question_updates: object | None = None,
    ) -> dict[str, object]:
        """Publish authoritative state before memory or derived artifacts advance.

        Decision snapshots include the full question ledger and this turn's
        updates in the same atomic write. Separate accepted/ledger artifacts
        are derived views and may lag if their later publication fails.
        ``value_sha256`` continues to identify the decision value alone.
        """
        snapshot = {
            "revision": self.revision + 1,
            "phase": phase,
            "value": value,
            "value_sha256": digest(canonical(value).encode()),
        }
        if questions is not None:
            snapshot["questions"] = questions
        if question_updates is not None:
            snapshot["question_updates"] = question_updates
        self.write("snapshot.json", snapshot)
        self.revision += 1
        return snapshot
