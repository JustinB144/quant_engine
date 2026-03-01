#!/usr/bin/env python3
"""
Retrain the quant engine model — checks triggers and retrains if needed.

Usage:
    python3 run_retrain.py                    # Check triggers, retrain if needed
    python3 run_retrain.py --force            # Force retrain regardless of triggers
    python3 run_retrain.py --status           # Show retrain status
    python3 run_retrain.py --survivorship     # Use WRDS survivorship-bias-free universe
    python3 run_retrain.py --rollback V_ID    # Rollback to previous model version
    python3 run_retrain.py --versions         # List all model versions
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import (
    UNIVERSE_FULL, UNIVERSE_QUICK, FORWARD_HORIZONS, FEATURE_MODE_DEFAULT,
    RETRAIN_REGIME_CHANGE_DAYS, RESULTS_DIR,
)
from quant_engine.data.loader import load_universe, load_survivorship_universe, warn_if_survivorship_biased
from quant_engine.features.pipeline import FeaturePipeline
from quant_engine.regime.detector import RegimeDetector
from quant_engine.models.governance import ModelGovernance
from quant_engine.models.trainer import ModelTrainer
from quant_engine.models.retrain_trigger import RetrainTrigger
from quant_engine.models.versioning import ModelRegistry
from quant_engine.reproducibility import (
    build_run_manifest,
    write_run_manifest,
    verify_manifest,
)


def _check_regime_change_trigger(predictions_df, trained_regime, days_threshold):
    """
    Check whether the market regime has changed for a sustained period.

    Args:
        predictions_df: DataFrame with a 'regime' column (one row per day/observation).
        trained_regime: The dominant regime at the time the model was last trained.
        days_threshold: Number of consecutive recent rows where the dominant regime
                        must differ from ``trained_regime`` to trigger retraining.

    Returns:
        True if the dominant regime in the most recent ``days_threshold`` rows
        differs from ``trained_regime``; False otherwise.
    """
    if predictions_df is None or predictions_df.empty:
        return False

    if "regime" not in predictions_df.columns:
        return False

    recent = predictions_df["regime"].dropna()

    if len(recent) < days_threshold:
        return False

    tail = recent.iloc[-days_threshold:]
    dominant_regime = tail.value_counts().idxmax()

    return dominant_regime != trained_regime


def main():
    """Evaluate retraining conditions and run controlled retraining/promotion updates when triggered."""
    parser = argparse.ArgumentParser(
        description="Retrain the quant engine model",
    )
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of triggers")
    parser.add_argument("--status", action="store_true", help="Show retrain status and exit")
    parser.add_argument("--versions", action="store_true", help="List all model versions")
    parser.add_argument("--rollback", type=str, help="Rollback to a specific version ID")
    parser.add_argument("--survivorship", action="store_true", help="Use WRDS survivorship-bias-free universe")
    parser.add_argument("--full", action="store_true", help="Use full universe")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--horizon", type=int, nargs="+", default=[10], help="Prediction horizons")
    parser.add_argument("--years", type=int, default=15, help="Years of data")
    parser.add_argument(
        "--feature-mode",
        choices=["core", "full"],
        default=FEATURE_MODE_DEFAULT,
        help="Feature profile: core (reduced complexity) or full",
    )
    parser.add_argument("--recency", action="store_true", help="Apply exponential recency weighting")
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

    # ── Status check ──
    if args.status:
        trigger = RetrainTrigger()
        print(trigger.status())
        return

    # ── List versions ──
    if args.versions:
        registry = ModelRegistry()
        versions = registry.list_versions()
        if not versions:
            print("No model versions found.")
            return
        latest = registry.latest_version_id
        print(f"\nModel Versions ({len(versions)} total):")
        print(f"{'='*60}")
        for v in versions:
            marker = " ← LATEST" if v.version_id == latest else ""
            print(f"  {v.version_id}{marker}")
            print(f"    Trained: {v.training_date}")
            print(f"    Horizon: {v.horizon}d, Samples: {v.n_samples}")
            print(f"    OOS Spearman: {v.oos_spearman:.4f}, CV gap: {v.cv_gap:.4f}")
            if v.survivorship_mode:
                print(f"    Survivorship-free: Yes (as of {v.universe_as_of})")
        return

    # ── Rollback ──
    if args.rollback:
        registry = ModelRegistry()
        if registry.rollback(args.rollback):
            print(f"Rolled back to version: {args.rollback}")
        else:
            print(f"ERROR: Version {args.rollback} not found.")
            sys.exit(1)
        return

    # ── Check triggers ──
    trigger = RetrainTrigger()
    if not args.force:
        should_retrain, reasons = trigger.check()

        # ── Regime-change trigger ──
        trained_regime = trigger.metadata.get("trained_regime")
        if trained_regime is not None:
            for horizon in args.horizon:
                predictions_path = Path(RESULTS_DIR) / f"predictions_{horizon}d.csv"
                if predictions_path.exists():
                    try:
                        predictions_df = pd.read_csv(predictions_path)
                        if _check_regime_change_trigger(
                            predictions_df,
                            trained_regime,
                            RETRAIN_REGIME_CHANGE_DAYS,
                        ):
                            recent_regime = (
                                predictions_df["regime"]
                                .dropna()
                                .iloc[-RETRAIN_REGIME_CHANGE_DAYS:]
                                .mode()
                                .iloc[0]
                            )
                            should_retrain = True
                            reasons.append(
                                f"Regime change: dominant regime is "
                                f"{int(recent_regime)} for the last "
                                f"{RETRAIN_REGIME_CHANGE_DAYS} predictions "
                                f"(trained regime was {trained_regime})."
                            )
                            break  # one horizon triggering is enough
                    except (OSError, ValueError, KeyError, TypeError):
                        pass  # skip if predictions file is malformed

        if not should_retrain:
            if verbose:
                print("No retraining needed.")
                print(trigger.status())
            return
        if verbose:
            print(f"\nRetraining triggered:")
            for r in reasons:
                print(f"  - {r}")

    # ── Build reproducibility manifest ──
    manifest = build_run_manifest(
        run_type="retrain",
        config_snapshot=vars(args),
    )

    # ── Load data ──
    t0 = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUANT ENGINE — RETRAIN")
        print(f"{'='*60}")

    if args.survivorship:
        if verbose:
            print(f"\n── Loading survivorship-bias-free universe ──")
        data = load_survivorship_universe(years=args.years, verbose=verbose)
    elif args.tickers:
        data = load_universe(args.tickers, years=args.years, verbose=verbose)
    elif args.full:
        data = load_universe(UNIVERSE_FULL, years=args.years, verbose=verbose)
    else:
        data = load_universe(UNIVERSE_QUICK, years=args.years, verbose=verbose)

    if not data:
        print("ERROR: No data loaded.")
        sys.exit(1)

    warn_if_survivorship_biased(data, context="retrain")

    # ── Compute features ──
    if verbose:
        print(f"\n── Computing features ──")
    pipeline = FeaturePipeline(feature_mode=args.feature_mode, include_interactions=True)
    features, targets = pipeline.compute_universe(data, verbose=verbose)

    # ── Detect regimes ──
    if verbose:
        print(f"\n── Detecting regimes ──")
    detector = RegimeDetector()
    # Build regimes for each PERMNO
    all_regimes = []
    all_regime_probs = []
    for permno in data:
        permno_features = features.loc[permno]
        regime_df = detector.regime_features(permno_features)
        regime_series = regime_df["regime"]
        regime_series = regime_series.to_frame()
        regime_series["permno"] = permno
        regime_series = regime_series.set_index("permno", append=True).reorder_levels([1, 0])
        all_regimes.append(regime_series["regime"])
        prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
        if prob_cols:
            rp = regime_df[prob_cols].copy()
            rp["permno"] = permno
            rp = rp.set_index("permno", append=True).reorder_levels([1, 0])
            all_regime_probs.append(rp)

    regimes = pd.concat(all_regimes)
    regime_probs = pd.concat(all_regime_probs) if all_regime_probs else None

    # ── Train for each horizon ──
    trainer = ModelTrainer()
    governance = ModelGovernance()
    registry = ModelRegistry()
    for horizon in args.horizon:
        target_col = f"target_{horizon}d"
        if target_col not in targets.columns:
            print(f"  Skipping {horizon}d — target not available")
            continue

        if verbose:
            print(f"\n── Training {horizon}d model ──")

        result = trainer.train_ensemble(
            features, targets[target_col], regimes,
            regime_probabilities=regime_probs,
            horizon=horizon, verbose=verbose,
            versioned=True,
            survivorship_mode=args.survivorship,
            recency_weight=args.recency,
        )

        latest_id = registry.latest_version_id
        if latest_id is not None:
            metrics = {
                "oos_spearman": float(np.mean(result.global_model.cv_scores) if result.global_model.cv_scores else 0),
                "holdout_spearman": float(result.global_model.holdout_correlation),
                "cv_gap": float(result.global_model.cv_gap),
            }
            decision = governance.evaluate_and_update(
                horizon=horizon,
                version_id=latest_id,
                metrics=metrics,
            )
            if verbose:
                print(
                    f"  Governance: promoted={decision['promoted']} "
                    f"reason={decision['reason']} score={decision['score']:.4f}",
                )

        # Record retraining (including dominant regime at training time)
        oos_spearman = float(
            np.mean(result.global_model.cv_scores)
            if result.global_model.cv_scores else 0
        )
        dominant_training_regime = int(regimes.value_counts().idxmax()) if len(regimes) > 0 else None
        trigger.record_retraining(
            n_trades=result.total_samples,
            oos_spearman=oos_spearman,
            notes=f"horizon={horizon}d, survivorship={args.survivorship}",
        )
        # Persist the dominant regime at training time for regime-change trigger
        if dominant_training_regime is not None:
            trigger.metadata["trained_regime"] = dominant_training_regime
            trigger._save_metadata()

    # ── Write reproducibility manifest ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(manifest, output_dir=RESULTS_DIR)
    if verbose:
        print(f"  Manifest: {manifest_path}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Retrain completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
