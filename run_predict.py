#!/usr/bin/env python3
"""
Generate predictions using trained ensemble model.

Usage:
    python3 run_predict.py                     # Quick universe, latest bar
    python3 run_predict.py --full              # Full universe
    python3 run_predict.py --tickers AAPL NVDA # Specific tickers
    python3 run_predict.py --horizon 10        # 10d forward prediction
    python3 run_predict.py --output signals.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import (
    UNIVERSE_FULL, UNIVERSE_QUICK, ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD,
    RESULTS_DIR, REGIME_NAMES, FEATURE_MODE_DEFAULT,
)
from quant_engine.data.loader import load_universe
from quant_engine.features.pipeline import FeaturePipeline
from quant_engine.regime.detector import RegimeDetector
from quant_engine.models.predictor import EnsemblePredictor


def main():
    """Load a trained model version and generate prediction outputs for the configured universe."""
    parser = argparse.ArgumentParser(
        description="Generate predictions from trained ensemble",
    )
    parser.add_argument("--full", action="store_true", help="Use full universe")
    parser.add_argument("--tickers", nargs="+", help="Specific symbols (resolved to PERMNO)")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon (days)")
    parser.add_argument("--version", type=str, default="latest", help="Model version to use")
    parser.add_argument(
        "--feature-mode",
        choices=["core", "full"],
        default=FEATURE_MODE_DEFAULT,
        help="Feature profile: core (reduced complexity) or full",
    )
    parser.add_argument("--output", type=str, help="Save predictions to CSV")
    parser.add_argument("--top", type=int, default=20, help="Show top N signals")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    t0 = time.time()

    if args.tickers:
        tickers = args.tickers
    elif args.full:
        tickers = UNIVERSE_FULL
    else:
        tickers = UNIVERSE_QUICK

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUANT ENGINE — PREDICTIONS ({args.horizon}d)")
        print(f"{'='*60}")

    # ── Load data ──
    if verbose:
        print(f"\n── Loading data ({len(tickers)} tickers) ──")
    data = load_universe(tickers, years=2, verbose=verbose)

    if not data:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # ── Compute features ──
    if verbose:
        print(f"\n── Computing features (universe) ──")
    pipeline = FeaturePipeline(
        feature_mode=args.feature_mode,
        include_interactions=True,
        verbose=verbose,
    )
    universe_features, _ = pipeline.compute_universe(
        data=data,
        verbose=verbose,
        compute_targets_flag=False,
    )
    available_permnos = set(universe_features.index.get_level_values(0))

    # ── Load model ──
    if verbose:
        print(f"\n── Loading {args.horizon}d model ──")
    try:
        predictor = EnsemblePredictor(horizon=args.horizon, version=args.version)
    except FileNotFoundError as e:
        print(f"ERROR: No trained model found for {args.horizon}d horizon.")
        print(f"  Run: python3 run_train.py --horizon {args.horizon}")
        sys.exit(1)

    if verbose:
        print(f"  Global features: {len(predictor.global_features)}")
        print(f"  Regime models: {list(predictor.regime_models.keys())}")

    # ── Generate predictions per PERMNO ──
    detector = RegimeDetector()
    all_predictions = []

    for permno, df in data.items():
        ticker = str(df.attrs.get("ticker", ""))
        if verbose:
            label = f"{permno}/{ticker}" if ticker else str(permno)
            print(f"  Predicting {label}...", end="", flush=True)

        if permno not in available_permnos:
            if verbose:
                print(" no features")
            continue
        features = universe_features.loc[permno]
        regime_df = detector.regime_features(features)
        regimes = regime_df["regime"]
        confidence = regime_df["regime_confidence"]
        regime_probs = regime_df[[c for c in regime_df.columns if c.startswith("regime_prob_")]]

        try:
            preds = predictor.predict(features, regimes, confidence, regime_probabilities=regime_probs)
            preds["permno"] = str(permno)
            preds["ticker"] = ticker
            preds["date"] = features.index
            all_predictions.append(preds)
            if verbose:
                latest = preds.iloc[-1]
                signal = "LONG" if latest["predicted_return"] > ENTRY_THRESHOLD else "---"
                print(f" pred={latest['predicted_return']:+.4f} conf={latest['confidence']:.2f} [{signal}]")
        except (ValueError, KeyError, TypeError) as e:
            if verbose:
                print(f" ERROR: {e}")

    if not all_predictions:
        print("No predictions generated.")
        sys.exit(1)

    predictions = pd.concat(all_predictions, ignore_index=True)

    # ── Display top signals ──
    latest = predictions.groupby("permno").last().reset_index()
    latest = latest.sort_values("predicted_return", ascending=False)

    # Filter to actionable signals
    signals = latest[
        (latest["predicted_return"] > ENTRY_THRESHOLD)
        & (latest["confidence"] > CONFIDENCE_THRESHOLD)
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"TOP SIGNALS — {args.horizon}d forward")
        print(f"{'='*60}")
        print(f"  Threshold: pred > {ENTRY_THRESHOLD:.3f}, conf > {CONFIDENCE_THRESHOLD:.2f}")
        print(f"  Signals: {len(signals)} / {len(latest)} PERMNOs\n")

        if len(signals) > 0:
            display = signals.head(args.top)
            for _, row in display.iterrows():
                regime = REGIME_NAMES.get(int(row["regime"]), "unknown")
                ticker_lbl = str(row.get("ticker", ""))
                label = f"{row['permno']}/{ticker_lbl}" if ticker_lbl else str(row["permno"])
                print(f"  {label:14s}  pred={row['predicted_return']:+.4f}  "
                      f"conf={row['confidence']:.2f}  regime={regime}  "
                      f"alpha={row['blend_alpha']:.2f}")
        else:
            print("  No signals above threshold.")

        print(f"\n  All PERMNOs (by predicted return):")
        for _, row in latest.head(args.top).iterrows():
            regime = REGIME_NAMES.get(int(row["regime"]), "unknown")
            flag = "*" if row["predicted_return"] > ENTRY_THRESHOLD else " "
            ticker_lbl = str(row.get("ticker", ""))
            label = f"{row['permno']}/{ticker_lbl}" if ticker_lbl else str(row["permno"])
            print(f"  {flag} {label:14s}  pred={row['predicted_return']:+.4f}  "
                  f"conf={row['confidence']:.2f}  regime={regime}")

    # ── Save output ──
    if args.output:
        out_path = Path(args.output)
        latest.to_csv(out_path, index=False)
        print(f"\n  Saved to {out_path}")
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"predictions_{args.horizon}d.csv"
        latest.to_csv(out_path, index=False)
        if verbose:
            print(f"\n  Saved to {out_path}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
