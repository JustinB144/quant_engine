#!/usr/bin/env python3
"""
Run one full autopilot cycle:
discovery -> validation gate -> promotion registry -> paper-trading execution.

Usage:
    python3 run_autopilot.py
    python3 run_autopilot.py --full
    python3 run_autopilot.py --tickers AAPL MSFT NVDA
    python3 run_autopilot.py --horizon 5 --max-candidates 12
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.autopilot.engine import AutopilotEngine
from quant_engine.config import (
    UNIVERSE_FULL, UNIVERSE_QUICK, AUTOPILOT_FEATURE_MODE, AUTOPILOT_DIR,
)
from quant_engine.reproducibility import (
    build_run_manifest,
    write_run_manifest,
    verify_manifest,
)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Run quant_engine autopilot cycle")
    parser.add_argument("--full", action="store_true", help="Use full configured universe")
    parser.add_argument("--tickers", nargs="+", help="Explicit ticker list")
    parser.add_argument("--horizon", type=int, default=10, help="Holding/prediction horizon")
    parser.add_argument("--years", type=int, default=15, help="History window for evaluation")
    parser.add_argument(
        "--version",
        type=str,
        default="champion",
        help="Model version (champion, latest, or explicit version id)",
    )
    parser.add_argument("--max-candidates", type=int, help="Limit candidate count")
    parser.add_argument(
        "--feature-mode",
        choices=["core", "full"],
        default=AUTOPILOT_FEATURE_MODE,
        help="Feature profile for autopilot evaluation",
    )
    parser.add_argument(
        "--no-survivorship",
        action="store_true",
        help="Opt out of survivorship-bias-free point-in-time universe",
    )
    parser.add_argument("--no-walkforward", action="store_true",
                        help="Use single-split training instead of rolling walk-forward")
    parser.add_argument("--allow-in-sample", action="store_true",
                        help="Disable strict out-of-sample filtering")
    parser.add_argument(
        "--verify-manifest", type=str, default=None,
        help="Path to a manifest file to verify environment matches before running",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    # ── Verify manifest (if requested) ──
    if args.verify_manifest:
        verification = verify_manifest(Path(args.verify_manifest), config_snapshot=vars(args))
        if verification["mismatches"]:
            print("WARNING: Manifest verification found mismatches:")
            for m in verification["mismatches"]:
                print(f"  - {m}")
            print("Continuing anyway — results may not be reproducible.\n")
        elif verbose:
            print("Manifest verification passed — environment matches.\n")

    # ── Build reproducibility manifest ──
    manifest = build_run_manifest(
        run_type="autopilot",
        config_snapshot=vars(args),
        script_name="run_autopilot",
    )

    if args.tickers:
        tickers = args.tickers
    elif args.full:
        tickers = UNIVERSE_FULL
    else:
        tickers = UNIVERSE_QUICK

    t0 = time.time()
    engine = AutopilotEngine(
        tickers=tickers,
        horizon=args.horizon,
        years=args.years,
        feature_mode=args.feature_mode,
        model_version=args.version,
        max_candidates=args.max_candidates,
        strict_oos=not args.allow_in_sample,
        survivorship_mode=not args.no_survivorship,
        walk_forward=not args.no_walkforward,
        verbose=not args.quiet,
    )
    report = engine.run_cycle()

    # ── Write reproducibility manifest ──
    manifest["extra"] = {
        "n_candidates": report.get("n_candidates", 0),
        "n_passed": report.get("n_passed", 0),
        "n_promoted": report.get("n_promoted", 0),
        "n_active": report.get("n_active", 0),
    }
    manifest_path = write_run_manifest(manifest, output_dir=AUTOPILOT_DIR)

    if not args.quiet:
        elapsed = time.time() - t0
        print("\n=== AUTOPILOT SUMMARY ===")
        print(f"  Candidates: {report['n_candidates']}")
        print(f"  Passed gate: {report['n_passed']}")
        print(f"  Promoted this cycle: {report['n_promoted']}")
        print(f"  Active strategies: {report['n_active']}")
        print(f"  Paper equity: {report['paper_report']['equity']:.2f}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
