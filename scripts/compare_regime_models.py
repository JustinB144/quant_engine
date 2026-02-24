#!/usr/bin/env python3
"""
A/B comparison: Jump Model (PyPI) vs HMM baseline.

Runs identical regime detection on a representative universe and compares
key metrics. This is a diagnostic script that produces data for human
decision-making — it does NOT fail on metric differences.

Usage:
    python3 scripts/compare_regime_models.py
    python3 scripts/compare_regime_models.py --tickers AAPL NVDA MSFT
    python3 scripts/compare_regime_models.py --bars 500

Output:
    results/regime_model_comparison.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quant_engine.config import RESULTS_DIR, REGIME_NAMES, DATA_CACHE_DIR
from quant_engine.regime.detector import RegimeDetector
from quant_engine.regime.hmm import build_hmm_observation_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_representative_features(
    tickers: list[str] | None = None,
    bars: int = 600,
) -> dict[str, pd.DataFrame]:
    """Load cached OHLCV data and compute basic features for comparison.

    Returns dict of ticker -> feature DataFrame.
    """
    from quant_engine.features.pipeline import FeaturePipeline

    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "JPM", "UNH", "TSLA"]

    pipeline = FeaturePipeline(feature_mode="core", verbose=False)
    features_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        # Try WRDS PERMNO-style naming first, then ticker naming
        path = DATA_CACHE_DIR / f"{ticker}_1d.parquet"
        if not path.exists():
            matches = sorted(DATA_CACHE_DIR.glob(f"{ticker}_daily_*.parquet"))
            if matches:
                path = matches[0]
        if not path.exists():
            logger.warning("No cached data for %s, skipping", ticker)
            continue

        try:
            panel = pd.read_parquet(path)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(set(panel.columns)):
                logger.warning("Missing OHLCV columns for %s, skipping", ticker)
                continue

            panel = panel[list(required)].copy()
            panel.index = pd.to_datetime(panel.index)
            panel = panel.sort_index().tail(bars)

            if len(panel) < 200:
                logger.warning("Insufficient data for %s (%d bars), skipping", ticker, len(panel))
                continue

            feats, _ = pipeline.compute(panel, compute_targets_flag=False)
            feats = feats.replace([np.inf, -np.inf], np.nan)
            features_by_ticker[ticker] = feats
        except Exception as e:
            logger.warning("Failed to load %s: %s", ticker, e)
            continue

    return features_by_ticker


def compute_regime_metrics(
    detector: RegimeDetector,
    features: pd.DataFrame,
    returns: pd.Series,
) -> dict:
    """Compute regime detection metrics for a single ticker."""
    out = detector.detect_full(features)

    regime = out.regime
    n = len(regime)

    # Regime transition frequency
    transitions = int(np.sum(np.diff(regime.values) != 0))
    transitions_per_year = transitions / (n / 252) if n > 0 else 0

    # Average regime duration
    if transitions > 0:
        avg_duration = n / (transitions + 1)
    else:
        avg_duration = float(n)

    # Regime distribution
    regime_dist = {}
    for code in range(4):
        frac = float((regime == code).sum()) / n if n > 0 else 0.0
        regime_dist[REGIME_NAMES.get(code, f"regime_{code}")] = round(frac, 4)

    # Simple regime-following strategy performance
    # Long in bull (0), short in bear (1), flat in MR (2) and HV (3)
    strategy_returns = returns.copy()
    strategy_returns = strategy_returns.reindex(regime.index).fillna(0.0)
    regime_aligned = regime.reindex(strategy_returns.index).fillna(2).astype(int)

    signal = pd.Series(0.0, index=strategy_returns.index)
    signal[regime_aligned == 0] = 1.0    # Long in bull
    signal[regime_aligned == 1] = -1.0   # Short in bear
    # Flat in MR and HV

    strat_ret = signal.shift(1).fillna(0.0) * strategy_returns
    strat_ret = strat_ret.fillna(0.0)

    mean_ret = float(strat_ret.mean())
    std_ret = float(strat_ret.std())
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-10 else 0.0

    # Max drawdown
    cum = (1 + strat_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak.replace(0, 1)
    max_dd = float(dd.min())

    # Win rate
    non_zero = strat_ret[strat_ret != 0]
    win_rate = float((non_zero > 0).sum() / len(non_zero)) if len(non_zero) > 0 else 0.0

    # Sortino ratio
    downside = strat_ret[strat_ret < 0]
    downside_std = float(downside.std()) if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 1e-10 else 0.0

    # Confidence stats
    avg_confidence = float(out.confidence.mean())
    avg_uncertainty = float(out.uncertainty.mean()) if out.uncertainty is not None else np.nan

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "transitions_per_year": round(transitions_per_year, 2),
        "avg_regime_duration_days": round(avg_duration, 1),
        "regime_distribution": regime_dist,
        "avg_confidence": round(avg_confidence, 4),
        "avg_uncertainty": round(avg_uncertainty, 4) if np.isfinite(avg_uncertainty) else None,
        "n_bars": n,
    }


def main():
    """Run the comparison."""
    parser = argparse.ArgumentParser(description="Compare Jump Model vs HMM regime detection")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to test")
    parser.add_argument("--bars", type=int, default=600, help="Number of bars per ticker")
    args = parser.parse_args()

    print("=" * 70)
    print("  Regime Model Comparison: Jump Model (PyPI) vs HMM")
    print("=" * 70)
    print()

    # Load data
    logger.info("Loading representative universe...")
    features_by_ticker = load_representative_features(
        tickers=args.tickers,
        bars=args.bars,
    )

    if not features_by_ticker:
        logger.error("No data loaded — cannot compare. Check DATA_CACHE_DIR.")
        sys.exit(1)

    logger.info("Loaded %d tickers: %s", len(features_by_ticker), list(features_by_ticker.keys()))

    # Run both methods
    methods = {
        "jump": RegimeDetector(method="jump"),
        "hmm": RegimeDetector(method="hmm"),
    }

    results = {}

    for method_name, detector in methods.items():
        logger.info("Running %s regime detection...", method_name)
        t0 = time.time()

        all_metrics = []
        for ticker, features in features_by_ticker.items():
            returns = features.get("return_1d", pd.Series(0.0, index=features.index))
            try:
                metrics = compute_regime_metrics(detector, features, returns)
                metrics["ticker"] = ticker
                all_metrics.append(metrics)
            except Exception as e:
                logger.warning("Failed %s for %s: %s", method_name, ticker, e)

        elapsed = time.time() - t0

        if not all_metrics:
            logger.warning("No successful runs for %s", method_name)
            continue

        # Aggregate across tickers
        avg_sharpe = np.mean([m["sharpe"] for m in all_metrics])
        avg_sortino = np.mean([m["sortino"] for m in all_metrics])
        avg_max_dd = np.mean([m["max_drawdown"] for m in all_metrics])
        avg_win_rate = np.mean([m["win_rate"] for m in all_metrics])
        avg_transitions = np.mean([m["transitions_per_year"] for m in all_metrics])
        avg_duration = np.mean([m["avg_regime_duration_days"] for m in all_metrics])
        avg_confidence = np.mean([m["avg_confidence"] for m in all_metrics])

        results[method_name] = {
            "avg_sharpe": round(float(avg_sharpe), 4),
            "avg_sortino": round(float(avg_sortino), 4),
            "avg_max_drawdown": round(float(avg_max_dd), 4),
            "avg_win_rate": round(float(avg_win_rate), 4),
            "avg_transitions_per_year": round(float(avg_transitions), 2),
            "avg_regime_duration_days": round(float(avg_duration), 1),
            "avg_confidence": round(float(avg_confidence), 4),
            "elapsed_seconds": round(elapsed, 2),
            "n_tickers": len(all_metrics),
            "per_ticker": all_metrics,
        }

    # Print comparison table
    print()
    print(f"{'Metric':<35} {'Jump Model':>15} {'HMM':>15}")
    print("-" * 65)

    for metric_key, label in [
        ("avg_sharpe", "Sharpe Ratio"),
        ("avg_sortino", "Sortino Ratio"),
        ("avg_max_drawdown", "Max Drawdown"),
        ("avg_win_rate", "Win Rate"),
        ("avg_transitions_per_year", "Transitions/Year"),
        ("avg_regime_duration_days", "Avg Duration (days)"),
        ("avg_confidence", "Avg Confidence"),
        ("elapsed_seconds", "Runtime (sec)"),
    ]:
        jump_val = results.get("jump", {}).get(metric_key, "N/A")
        hmm_val = results.get("hmm", {}).get(metric_key, "N/A")
        print(f"  {label:<33} {str(jump_val):>15} {str(hmm_val):>15}")

    print()

    # Warnings
    if "jump" in results and "hmm" in results:
        jump = results["jump"]
        hmm = results["hmm"]

        if jump["avg_sharpe"] < hmm["avg_sharpe"] - 0.05:
            logger.warning(
                "Jump Model Sharpe (%.3f) is significantly lower than HMM (%.3f)",
                jump["avg_sharpe"], hmm["avg_sharpe"],
            )
        else:
            logger.info(
                "Jump Model Sharpe (%.3f) vs HMM (%.3f) — within tolerance",
                jump["avg_sharpe"], hmm["avg_sharpe"],
            )

        if jump["avg_max_drawdown"] < hmm["avg_max_drawdown"] - 0.01:
            logger.warning(
                "Jump Model max DD (%.3f) is worse than HMM (%.3f)",
                jump["avg_max_drawdown"], hmm["avg_max_drawdown"],
            )

        if jump["avg_transitions_per_year"] > hmm["avg_transitions_per_year"]:
            logger.warning(
                "Jump Model has MORE transitions/year (%.1f) than HMM (%.1f) — "
                "expected fewer with explicit persistence",
                jump["avg_transitions_per_year"], hmm["avg_transitions_per_year"],
            )
        else:
            logger.info(
                "Jump Model has fewer transitions/year (%.1f vs %.1f) — as expected",
                jump["avg_transitions_per_year"], hmm["avg_transitions_per_year"],
            )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "regime_model_comparison.json"

    # Remove per_ticker DataFrames (not JSON serializable if they contain timestamps)
    save_results = {}
    for method_name, method_results in results.items():
        save_data = {k: v for k, v in method_results.items() if k != "per_ticker"}
        # Include per-ticker summaries without non-serializable objects
        save_data["per_ticker"] = []
        for m in method_results.get("per_ticker", []):
            ticker_data = {k: v for k, v in m.items()}
            save_data["per_ticker"].append(ticker_data)
        save_results[method_name] = save_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
