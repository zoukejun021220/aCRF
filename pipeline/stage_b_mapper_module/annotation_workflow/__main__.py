"""Explicit V3 CLI; legacy mapper defaults and historical experiments remain intact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import WorkflowError, canonical, digest, parse
from .knowledge import Knowledge
from .preflight import run_preflight, validate_gate
from .runtime import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SCHEMA_TRANSPORT,
    LocalClient,
    schema_for,
)
from .session import Session
from .source import SourcePacket
from .store import RunStore
from .understanding import CollectionUnderstanding, queries


def identity(root: Path) -> dict[str, str]:
    """Freeze actual source dependencies and raw metadata, excluding run artifacts."""
    paths: list[Path] = []
    for name in (
        "pipeline/stage_b_mapper_module/annotation_workflow",
        "pipeline/stage_b_mapper_module/graphrag",
        "sdtm_knowledge_base/cosmos",
    ):
        paths.extend((root / name).rglob("*.py"))
    paths.extend(
        (root / "pipeline/stage_b_mapper_module/annotation_workflow").glob("*.txt")
    )
    paths.extend(
        (root / "pipeline/stage_b_mapper_module/annotation_workflow").glob("*.json")
    )
    paths.extend(root.glob("kb/cosmos/*/raw/**/*"))
    return {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in sorted(set(paths))
        if path.is_file()
    }


def main() -> int:
    """Expose bounded explicit commands with nonzero failure and no hidden model loads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("export-schemas", "freeze", "probe", "retrieve", "map", "evaluate"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--understanding", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--runtime-gate", type=Path)
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument(
        "--schema-transport",
        choices=("prompt", "native"),
        default=DEFAULT_SCHEMA_TRANSPORT,
    )
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low"), default=DEFAULT_REASONING_EFFORT
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    try:
        if args.command == "evaluate":
            # Expected answers are only loaded by this explicit offline command.
            from .decision import MappingDecision
            from .evaluation import compare

            if args.actual is None or args.expected is None:
                raise WorkflowError("actual_and_expected_required")
            comparison = compare(
                parse(MappingDecision, args.actual.read_text()),
                parse(MappingDecision, args.expected.read_text()),
            )
            RunStore(args.output).write(
                "comparison.json", comparison.model_dump(mode="json")
            )
            return 0 if comparison.exact else 2
        if args.command == "export-schemas":
            store = RunStore(args.output)
            for phase in ("understanding", "decision"):
                store.write(phase + ".schema.json", schema_for(phase))
            return 0
        if args.command == "freeze":
            store = RunStore(args.output)
            store.write("identity.json", identity(root))
            return 0
        if args.command == "probe":
            store = RunStore(args.output)
            client = LocalClient(
                model=args.model,
                base_url=args.base_url,
                timeout=90,
                schema_transport=args.schema_transport,
                reasoning_effort=args.reasoning_effort,
            )
            ready = run_preflight(
                client, store, identity(root), args.model, args.base_url
            )
            return 0 if ready else 2
        if args.source is None or args.snapshot is None:
            raise WorkflowError("source_and_snapshot_required")
        packet = parse(SourcePacket, args.source.read_text())
        if args.command == "map":
            if args.runtime_gate is None:
                raise WorkflowError("runtime_gate_required")
            validate_gate(
                args.runtime_gate,
                identity(root),
                args.model,
                args.base_url,
                args.schema_transport,
                args.reasoning_effort,
            )
        knowledge = Knowledge(args.snapshot, packet.standard, args.rules)
        try:
            if args.command == "retrieve":
                if args.understanding is None:
                    raise WorkflowError("understanding_required")
                value = parse(CollectionUnderstanding, args.understanding.read_text())
                store = RunStore(args.output)
                output = [
                    {
                        "query": query.model_dump(mode="json"),
                        "candidates": [
                            candidate.wire() for candidate in knowledge.search(query)
                        ],
                    }
                    for query in queries(value, packet)
                ]
                store.write("retrieval.json", output)
                return 0
            session = Session(
                packet,
                knowledge,
                LocalClient(
                    args.model,
                    args.base_url,
                    schema_transport=args.schema_transport,
                    reasoning_effort=args.reasoning_effort,
                ),
                args.output,
            )
            session.store.write("identity.json", identity(root))
            result = session.run()
            print(
                canonical(
                    {
                        "status": result.status,
                        "stop_reason": result.stop_reason,
                        "calls": result.calls,
                    }
                )
            )
            return 0 if result.stop_reason == "completed_for_review" else 2
        finally:
            knowledge.close()
    except (WorkflowError, OSError, ValueError) as exc:
        print(
            canonical(
                {
                    "status": "failed",
                    "code": (
                        exc.code
                        if isinstance(exc, WorkflowError)
                        else "invalid_cli_input"
                    ),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
