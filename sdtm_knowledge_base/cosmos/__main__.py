"""Inspect, import and export COSMoS knowledge locally; no model or network calls."""

import argparse
import logging
import sys
from pathlib import Path

from .snapshot import DEFAULT_SNAPSHOT, build_snapshot, json_bytes, load_snapshot
from .source import FILES

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run an offline command; emit JSON on success and exit 2 on invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser(
        "import", help="Import previously downloaded pinned source files"
    )
    importer.add_argument("--source-directory", type=Path, required=True)
    importer.add_argument("--output-directory", type=Path, default=DEFAULT_SNAPSHOT)
    verifier = commands.add_parser(
        "verify", help="Verify every source and generated artifact"
    )
    verifier.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    show = commands.add_parser(
        "show", help="Look up a complete concept or specialization"
    )
    show.add_argument("kind", choices=FILES)
    show.add_argument("source_id")
    evidence = commands.add_parser(
        "evidence", help="Export whole records for retrieval"
    )
    evidence.add_argument("--kind", choices=FILES)
    evidence.add_argument("--domain")
    for command in (show, evidence):
        command.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
        command.add_argument("--sdtmig-version", default="3.4")
        command.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args(argv)
    result: dict | list
    try:
        if args.command == "import":
            result = build_snapshot(
                args.source_directory, args.output_directory
            ).summary()
        else:
            catalog = load_snapshot(args.snapshot)
            if args.command == "verify":
                result = catalog.summary()
            elif args.command == "show":
                result = catalog.lookup(
                    args.kind,
                    args.source_id,
                    sdtmig_version=args.sdtmig_version,
                    include_drafts=args.include_drafts,
                ).to_dict()
            else:
                result = catalog.evidence_documents(
                    kind=args.kind,
                    domain=args.domain,
                    sdtmig_version=args.sdtmig_version,
                    include_drafts=args.include_drafts,
                )
        sys.stdout.buffer.write(json_bytes(result))
        return 0
    except (ValueError, LookupError, OSError) as exc:
        logger.error("COSMoS command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
