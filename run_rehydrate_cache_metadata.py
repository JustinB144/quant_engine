#!/usr/bin/env python3
"""
Backfill cache metadata sidecars for existing OHLCV cache files.

Examples:
    .venv/bin/python quant_engine/run_rehydrate_cache_metadata.py
    .venv/bin/python quant_engine/run_rehydrate_cache_metadata.py --dry-run
    .venv/bin/python quant_engine/run_rehydrate_cache_metadata.py \
      --root-source "/path/to/cache=ibkr" --force --overwrite-source
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import DATA_CACHE_DIR, FRAMEWORK_DIR
from quant_engine.data.local_cache import rehydrate_cache_metadata


def _parse_root_source(items: list[str]) -> dict[str, str]:
    """Parse repeated --root-source values of the form "<path>=<source>" into a mapping."""
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --root-source '{item}'. Use '<path>=<source>'.")
        root, source = item.split("=", 1)
        root = root.strip()
        source = source.strip().lower()
        if not root or not source:
            raise ValueError(f"Invalid --root-source '{item}'. Use '<path>=<source>'.")
        mapping[root] = source
    return mapping


def main():
    """Backfill cache metadata sidecars for existing cached OHLCV files and print a summary report."""
    parser = argparse.ArgumentParser(description="Backfill quant_engine cache metadata sidecars")
    parser.add_argument(
        "--roots",
        nargs="+",
        help="Optional cache roots to scan. Defaults to standard quant_engine roots.",
    )
    parser.add_argument(
        "--root-source",
        action="append",
        default=[],
        help="Root source override in the form '<path>=<source>' (can repeat).",
    )
    parser.add_argument(
        "--default-source",
        default="unknown",
        help="Default source label for unmapped roots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite metadata even when sidecar already exists.",
    )
    parser.add_argument(
        "--overwrite-source",
        action="store_true",
        help="When rewriting metadata, replace existing source labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not write metadata.",
    )
    args = parser.parse_args()

    default_roots = [
        DATA_CACHE_DIR,
        FRAMEWORK_DIR / "data_cache",
        FRAMEWORK_DIR / "automated_portfolio_system" / "data_cache",
    ]
    roots = [Path(p) for p in (args.roots or default_roots)]
    source_by_root = _parse_root_source(args.root_source)

    summary = rehydrate_cache_metadata(
        cache_roots=roots,
        source_by_root=source_by_root,
        default_source=args.default_source,
        only_missing=not args.force,
        overwrite_source=args.overwrite_source,
        dry_run=args.dry_run,
    )

    print("\n=== CACHE METADATA REHYDRATION ===")
    print(f"  Roots scanned:         {summary['roots_seen']}")
    print(f"  Files scanned:         {summary['files_scanned']}")
    print(f"  Metadata written:      {summary['written']}")
    print(f"  Skipped existing meta: {summary['skipped_existing']}")
    print(f"  Skipped unreadable:    {summary['skipped_unreadable']}")
    print(f"  Missing roots:         {summary['skipped_missing_root']}")
    if args.dry_run:
        print("  Mode:                  DRY RUN")


if __name__ == "__main__":
    main()
