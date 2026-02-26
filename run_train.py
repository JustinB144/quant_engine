#!/usr/bin/env python3
"""
Train the regime-conditional ensemble model.

Usage:
    python3 run_train.py                        # Quick universe, 10d horizon
    python3 run_train.py --full                 # Full universe
    python3 run_train.py --horizon 5 10 20      # Multiple horizons
    python3 run_train.py --years 10             # 10 years of data
    python3 run_train.py --survivorship         # WRDS survivorship-bias-free universe
    python3 run_train.py --recency              # Exponential recency weighting
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import (
    UNIVERSE_FULL, UNIVERSE_QUICK, FORWARD_HORIZONS, LOOKBACK_YEARS,
    FEATURE_MODE_DEFAULT, RESULTS_DIR,
)
from quant_engine.data.loader import load_universe, load_survivorship_universe
from quant_engine.features.pipeline import FeaturePipeline
from quant_engine.regime.detector import RegimeDetector
from quant_engine.models.governance import ModelGovernance
from quant_engine.models.trainer import ModelTrainer
from quant_engine.models.versioning import ModelRegistry
from quant_engine.reproducibility import (
    build_run_manifest,
    write_run_manifest,
    verify_manifest,
)


def main():
    """Train a model for the requested horizon/workflow settings and persist versioned artifacts."""
    parser = argparse.ArgumentParser(
        description="Train regime-conditional gradient boosting ensemble",
    )
    parser.add_argument("--full", action="store_true", help="Use full universe")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to train on")
    parser.add_argument("--horizon", nargs="+", type=int, default=[10],
                        help="Forward return horizons (days)")
    parser.add_argument("--years", type=int, default=LOOKBACK_YEARS,
                        help=f"Years of historical data (default: {LOOKBACK_YEARS})")
    parser.add_argument("--no-interactions", action="store_true",
                        help="Skip interaction features")
    parser.add_argument(
        "--feature-mode",
        choices=["minimal", "core", "full"],
        default=FEATURE_MODE_DEFAULT,
        help="Feature profile: minimal (~20 indicators), core (reduced complexity), or full",
    )
    parser.add_argument("--survivorship", action="store_true",
                        help="Use WRDS survivorship-bias-free universe")
    parser.add_argument("--recency", action="store_true",
                        help="Apply exponential recency weighting")
    parser.add_argument("--no-version", action="store_true",
                        help="Skip model versioning (save flat)")
    parser.add_argument(
        "--verify-manifest", type=str, default=None,
        help="Path to a manifest file to verify environment matches before running",
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    verbose = not args.quiet
    t0 = time.time()

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
        run_type="train",
        config_snapshot=vars(args),
    )

    # ── Select universe ──
    if args.tickers:
        tickers = args.tickers
    elif args.full:
        tickers = UNIVERSE_FULL
    else:
        tickers = UNIVERSE_QUICK

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUANT ENGINE — TRAINING")
        print(f"{'='*60}")
        print(f"  Universe: {len(tickers)} requested symbols")
        print(f"  Horizons: {args.horizon}")
        print(f"  Lookback: {args.years} years")
        print(f"  Interactions: {not args.no_interactions}")

    # ── Step 1: Load data ──
    if verbose:
        print(f"\n── Step 1: Loading data ──")
    if args.survivorship:
        if verbose:
            print(f"  Using survivorship-bias-free WRDS universe")
        data = load_survivorship_universe(years=args.years, verbose=verbose)
    else:
        data = load_universe(tickers, years=args.years, verbose=verbose)

    if not data:
        print("ERROR: No data loaded. Check your data sources.")
        sys.exit(1)

    if verbose:
        print(f"  Loaded {len(data)} PERMNO series")

    # ── Step 2: Compute features ──
    if verbose:
        print(f"\n── Step 2: Computing features ──")
    pipeline = FeaturePipeline(
        feature_mode=args.feature_mode,
        include_interactions=not args.no_interactions,
        verbose=verbose,
    )
    features, targets = pipeline.compute_universe(data, verbose=verbose)

    if verbose:
        print(f"  Feature matrix: {features.shape}")
        print(f"  Target matrix: {targets.shape}")

    # ── Step 3: Detect regimes ──
    if verbose:
        print(f"\n── Step 3: Detecting regimes ──")
    detector = RegimeDetector()

    # Need to detect regimes per PERMNO, then concatenate
    regime_dfs = []
    regime_prob_dfs = []
    for permno in data:
        permno_feats = features.loc[permno]
        regime_df = detector.regime_features(permno_feats)
        regime_df["permno"] = permno
        regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
        regime_dfs.append(regime_df)
        prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
        if prob_cols:
            regime_prob_dfs.append(regime_df[prob_cols])

    regime_data = pd.concat(regime_dfs)
    regimes = regime_data["regime"]
    regime_probs = pd.concat(regime_prob_dfs) if regime_prob_dfs else None

    if verbose:
        regime_counts = regimes.value_counts().sort_index()
        from quant_engine.config import REGIME_NAMES
        for code, count in regime_counts.items():
            name = REGIME_NAMES.get(code, f"regime_{code}")
            print(f"  {name}: {count} ({count/len(regimes):.1%})")

    # ── Step 4: Train models ──
    trainer = ModelTrainer()
    governance = ModelGovernance()
    registry = ModelRegistry()

    for horizon in args.horizon:
        target_col = f"target_{horizon}d"
        if target_col not in targets.columns:
            print(f"  WARNING: {target_col} not in targets, skipping")
            continue

        if verbose:
            print(f"\n── Step 4: Training {horizon}d model ──")

        result = trainer.train_ensemble(
            features=features,
            targets=targets[target_col],
            regimes=regimes,
            regime_probabilities=regime_probs,
            horizon=horizon,
            verbose=verbose,
            versioned=not args.no_version,
            survivorship_mode=args.survivorship,
            recency_weight=args.recency,
        )

        if not args.no_version:
            latest_id = registry.latest_version_id
            if latest_id is not None:
                decision = governance.evaluate_and_update(
                    horizon=horizon,
                    version_id=latest_id,
                    metrics={
                        "oos_spearman": float(np.mean(result.global_model.cv_scores) if result.global_model.cv_scores else 0),
                        "holdout_spearman": float(result.global_model.holdout_correlation),
                        "cv_gap": float(result.global_model.cv_gap),
                    },
                )
                if verbose:
                    print(
                        f"  Governance: promoted={decision['promoted']} "
                        f"reason={decision['reason']} score={decision['score']:.4f}",
                    )

    # ── Write reproducibility manifest ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(manifest, output_dir=RESULTS_DIR)
    if verbose:
        print(f"\n  Manifest: {manifest_path}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — {elapsed:.0f}s total")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
