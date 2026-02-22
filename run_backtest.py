#!/usr/bin/env python3
"""
Backtest the trained model on historical data.

Usage:
    python3 run_backtest.py                       # Quick universe, 10d horizon
    python3 run_backtest.py --full                # Full universe
    python3 run_backtest.py --horizon 5           # 5d holding period
    python3 run_backtest.py --no-survivorship      # Opt out of survivorship-bias-free universe
    python3 run_backtest.py --no-validate          # Skip walk-forward validation
    python3 run_backtest.py --tickers AAPL NVDA   # Specific tickers
    python3 run_backtest.py --risk                # Enable risk management (dynamic sizing, stops, drawdown controls)
    python3 run_backtest.py --advanced            # Deflated Sharpe, Monte Carlo, PBO, capacity analysis
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from quant_engine.config import (
    UNIVERSE_FULL, UNIVERSE_QUICK, RESULTS_DIR, REGIME_NAMES, ENTRY_THRESHOLD,
    CPCV_PARTITIONS, CPCV_TEST_PARTITIONS, SPA_BOOTSTRAPS,
    SURVIVORSHIP_UNIVERSE_NAME,
    FEATURE_MODE_DEFAULT, WF_MAX_TRAIN_DATES,
)
from quant_engine.data.loader import load_universe, load_survivorship_universe
from quant_engine.data.survivorship import filter_panel_by_point_in_time_universe
from quant_engine.features.pipeline import FeaturePipeline
from quant_engine.regime.detector import RegimeDetector
from quant_engine.models.predictor import EnsemblePredictor
from quant_engine.backtest.engine import Backtester
from quant_engine.backtest.validation import (
    walk_forward_validate,
    run_statistical_tests,
    combinatorial_purged_cv,
    strategy_signal_returns,
    superior_predictive_ability,
)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest the trained ensemble model",
    )
    parser.add_argument("--full", action="store_true", help="Use full universe")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction/holding horizon (days)")
    parser.add_argument("--no-validate", action="store_true", help="Skip walk-forward validation")
    parser.add_argument("--years", type=int, default=15, help="Years of data for backtest")
    parser.add_argument(
        "--feature-mode",
        choices=["minimal", "core", "full"],
        default=FEATURE_MODE_DEFAULT,
        help="Feature profile: minimal (~20 indicators), core (reduced complexity), or full",
    )
    parser.add_argument("--risk", action="store_true", help="Enable risk management (dynamic sizing, stops, drawdown controls)")
    parser.add_argument("--advanced", action="store_true", help="Run advanced validation (Deflated Sharpe, Monte Carlo, PBO, capacity)")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of strategy variants tested (for Deflated Sharpe)")
    parser.add_argument("--version", type=str, default="latest", help="Model version to test (default: latest)")
    parser.add_argument("--no-survivorship", action="store_true", help="Opt out of survivorship-bias-free universe (use static universe instead)")
    parser.add_argument("--allow-in-sample", action="store_true",
                        help="Allow scoring dates that overlap model training history")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Minimum model confidence to enter (default: config CONFIDENCE_THRESHOLD)")
    parser.add_argument("--min-predicted", type=float, default=None,
                        help="Minimum predicted return to enter (default: config ENTRY_THRESHOLD)")
    parser.add_argument("--output", type=str, help="Save trade log to CSV")
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
        print(f"QUANT ENGINE — BACKTEST ({args.horizon}d)")
        print(f"{'='*60}")
        print(f"  Universe: {len(tickers)} tickers")
        print(f"  Horizon: {args.horizon}d")
        print(f"  Data: {args.years} years")

    # ── Load data ──
    if verbose:
        print(f"\n── Loading data ──")
    if args.no_survivorship:
        data = load_universe(tickers, years=args.years, verbose=verbose)
    else:
        data = load_survivorship_universe(years=args.years, verbose=verbose)

    if not data:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # ── Load model ──
    if verbose:
        print(f"\n── Loading {args.horizon}d model ──")
    try:
        predictor = EnsemblePredictor(horizon=args.horizon, version=args.version)
    except FileNotFoundError:
        print(f"ERROR: No trained model found for {args.horizon}d horizon.")
        print(f"  Run: python3 run_train.py --horizon {args.horizon}")
        sys.exit(1)

    # ── Compute features and predictions for each ticker ──
    if verbose:
        print(f"\n── Computing features & predictions ──")

    pipeline = FeaturePipeline(feature_mode=args.feature_mode, include_interactions=True)
    universe_features, universe_targets = pipeline.compute_universe(
        data=data,
        verbose=verbose,
        compute_targets_flag=True,
    )
    available_permnos = set(universe_features.index.get_level_values(0))
    detector = RegimeDetector()
    all_preds = []

    for i, permno in enumerate(data):
        if verbose:
            print(f"  [{i+1}/{len(data)}] {permno}...", end="", flush=True)
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
            preds = predictor.predict(
                features,
                regimes,
                confidence,
                regime_probabilities=regime_probs,
            )
            # Add PERMNO index level for backtester
            preds["permno"] = permno
            preds = preds.set_index("permno", append=True).swaplevel()
            all_preds.append(preds)
            if verbose:
                n_signals = (preds["predicted_return"] > ENTRY_THRESHOLD).sum()
                print(f" {n_signals} signals")
        except (ValueError, KeyError, TypeError) as e:
            if verbose:
                print(f" ERROR: {e}")

    if not all_preds:
        print("No predictions generated.")
        sys.exit(1)

    predictions = pd.concat(all_preds)

    # Enforce strict out-of-sample evaluation by default.
    if not args.allow_in_sample:
        train_end_str = predictor.meta.get("train_data_end")
        if train_end_str:
            train_end = pd.Timestamp(train_end_str)
            pred_dates = pd.to_datetime(predictions.index.get_level_values(1))
            oos_mask = pred_dates > train_end
            n_dropped = int((~oos_mask).sum())
            predictions = predictions[oos_mask]
            if verbose:
                print(f"  Strict OOS: dropped {n_dropped} in-sample rows (<= {train_end.date()})")
            if len(predictions) == 0:
                print(
                    "ERROR: No out-of-sample prediction rows remain. "
                    "Train on an earlier window or run with --allow-in-sample.",
                )
                sys.exit(1)

    # Enforce point-in-time universe membership when using survivorship mode (default).
    if not args.no_survivorship:
        predictions = filter_panel_by_point_in_time_universe(
            panel=predictions,
            universe_name=SURVIVORSHIP_UNIVERSE_NAME,
            verbose=verbose,
        )
        if len(predictions) == 0:
            print(
                "ERROR: No prediction rows remain after PIT universe filtering. "
                "Check WRDS survivorship history hydration.",
            )
            sys.exit(1)

    if verbose:
        total_signals = (predictions["predicted_return"] > ENTRY_THRESHOLD).sum()
        print(f"\n  Total predictions: {len(predictions)}")
        print(f"  Total signals (pred > {ENTRY_THRESHOLD:.2%}): {total_signals}")

    # ── Run backtest ──
    bt_kwargs = dict(holding_days=args.horizon, use_risk_management=args.risk)
    if args.min_confidence is not None:
        bt_kwargs["confidence_threshold"] = args.min_confidence
    if args.min_predicted is not None:
        bt_kwargs["entry_threshold"] = args.min_predicted
    backtester = Backtester(**bt_kwargs)
    result = backtester.run(predictions, data, verbose=verbose)

    # ── Walk-forward validation ──
    if not args.no_validate and result.total_trades > 0:
        if verbose:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD VALIDATION")
            print(f"{'='*60}")

        # Align predictions with actual forward returns, sorted by date
        target_col = f"target_{args.horizon}d"
        all_pred_vals = []
        all_actual_vals = []
        all_conf_vals = []

        for permno in data:
            if universe_targets is None or permno not in available_permnos:
                continue
            targets = universe_targets.loc[permno]
            if permno in predictions.index.get_level_values(0):
                ticker_preds = predictions.loc[permno]["predicted_return"]
                ticker_conf = predictions.loc[permno]["confidence"]
                ticker_actuals = targets[target_col]
                # Align indices
                common = ticker_preds.index.intersection(ticker_actuals.index)
                if len(common) > 0:
                    all_pred_vals.append(ticker_preds.loc[common])
                    all_conf_vals.append(ticker_conf.loc[common])
                    all_actual_vals.append(ticker_actuals.loc[common])

        if all_pred_vals:
            pred_series = pd.concat(all_pred_vals)
            conf_series = pd.concat(all_conf_vals)
            actual_series = pd.concat(all_actual_vals)

            # CRITICAL: Sort by date for proper temporal walk-forward
            # Use index-based sorting + align to prevent cross-ticker misalignment
            pred_series = pred_series.sort_index()
            conf_series = conf_series.sort_index()
            actual_series = actual_series.sort_index()
            n_before = len(pred_series)
            pred_series, actual_series = pred_series.align(actual_series, join="inner")
            conf_series = conf_series.reindex(pred_series.index)
            n_dropped = n_before - len(pred_series)
            if n_dropped > 0:
                pct_dropped = 100.0 * n_dropped / n_before
                logger.warning(
                    "Inner join dropped %d rows (%.1f%%) during prediction alignment",
                    n_dropped,
                    pct_dropped,
                )

            wf = walk_forward_validate(
                pred_series, actual_series,
                purge_gap=args.horizon,
                embargo=max(1, args.horizon // 2),
                max_train_samples=WF_MAX_TRAIN_DATES,
            )

            print(f"\n  Walk-Forward Results:")
            print(f"    Folds: {wf.n_folds}")
            print(f"    Avg OOS Spearman: {wf.avg_oos_corr:.4f}")
            print(f"    Avg IS Spearman: {wf.avg_is_corr:.4f}")
            print(f"    IS-OOS gap: {wf.is_oos_gap:.4f}")
            print(f"    Profitable folds: {wf.profitable_folds}/{wf.n_folds}")
            print(f"    Overfit detected: {wf.is_overfit}")
            if wf.warnings:
                for w in wf.warnings:
                    print(f"    WARNING: {w}")

            for fold in wf.folds:
                print(f"    Fold {fold.fold}: IS={fold.train_corr:.4f}, "
                      f"OOS={fold.test_corr:.4f}, "
                      f"mean_ret={fold.test_mean_return:+.4f}")

            # Statistical tests
            trade_returns = np.array([t.net_return for t in result.trades])
            stat_tests = run_statistical_tests(
                pred_series, actual_series, trade_returns,
                holding_days=args.horizon,
            )

            print(f"\n  Statistical Tests (FDR-corrected):")
            print(f"    Spearman correlation: {stat_tests.pred_actual_corr:.4f} (p={stat_tests.pred_actual_pval:.4f})")
            print(f"    Long signal mean return: {stat_tests.long_mean_return:.4f} "
                  f"(t={stat_tests.long_tstat:.2f}, p={stat_tests.long_pval:.4f})")
            print(f"    Sharpe significance: {stat_tests.sharpe:.2f} "
                  f"(SE={stat_tests.sharpe_se:.3f}, p={stat_tests.sharpe_pval:.4f})")
            print(f"    IC: mean={stat_tests.ic_mean:.4f}, IR={stat_tests.ic_ir:.2f} "
                  f"(t={stat_tests.ic_tstat:.2f}, p={stat_tests.ic_pval:.4f})")
            print(f"    FDR threshold: {stat_tests.fdr_threshold:.4f} (from {stat_tests.n_tests} tests)")
            print(f"    PASSES ALL TESTS: {stat_tests.passes}")

            # CPCV robustness
            effective_entry = float(args.min_predicted if args.min_predicted is not None else ENTRY_THRESHOLD)
            effective_conf = float(args.min_confidence if args.min_confidence is not None else 0.0)
            cpcv = combinatorial_purged_cv(
                predictions=pred_series,
                actuals=actual_series,
                entry_threshold=effective_entry,
                n_partitions=CPCV_PARTITIONS,
                n_test_partitions=CPCV_TEST_PARTITIONS,
                purge_gap=args.horizon,
                embargo=max(1, args.horizon // 2),
            )
            print(f"\n  CPCV Robustness:")
            print(f"    Partitions: {cpcv.n_partitions}, Test partitions: {cpcv.n_test_partitions}")
            print(f"    Combinations: {cpcv.n_combinations}")
            print(f"    Mean IS Spearman: {cpcv.mean_is_corr:.4f}")
            print(f"    Mean OOS Spearman: {cpcv.mean_oos_corr:.4f} (median={cpcv.median_oos_corr:.4f})")
            print(f"    OOS corr std: {cpcv.oos_corr_std:.4f}")
            print(f"    Positive OOS fraction: {cpcv.positive_oos_fraction:.1%}")
            print(f"    Mean OOS long return: {cpcv.mean_oos_return:.4f}")
            print(f"    PASSES CPCV: {cpcv.passes}")
            for w in cpcv.warnings:
                print(f"    WARNING: {w}")

            # SPA bootstrap
            signal_returns = strategy_signal_returns(
                predictions=pred_series,
                actuals=actual_series,
                entry_threshold=effective_entry,
                confidence=conf_series,
                min_confidence=effective_conf,
            )
            spa = superior_predictive_ability(
                strategy_returns=signal_returns,
                benchmark_returns=None,  # null model = no-signal baseline
                n_bootstraps=SPA_BOOTSTRAPS,
                block_size=max(5, args.horizon),
            )
            print(f"\n  SPA Bootstrap:")
            print(f"    Mean differential return: {spa.observed_mean:.6f}")
            print(f"    Test statistic: {spa.observed_stat:.4f}")
            print(f"    p-value: {spa.p_value:.4f} ({spa.n_bootstraps} bootstraps)")
            print(f"    PASSES SPA: {spa.passes}")

    # ── Advanced validation ──
    if args.advanced and result.total_trades > 10:
        from quant_engine.backtest.advanced_validation import run_advanced_validation
        trade_returns = np.array([t.net_return for t in result.trades])
        adv_report = run_advanced_validation(
            trade_returns=trade_returns,
            trades=result.trades,
            price_data=data,
            n_strategy_variants=args.n_trials,
            holding_days=args.horizon,
            verbose=verbose,
        )

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if result.total_trades > 0:
        # Save trade log
        trade_data = []
        for t in result.trades:
            trade_data.append({
                "permno": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "predicted_return": t.predicted_return,
                "actual_return": t.actual_return,
                "net_return": t.net_return,
                "regime": REGIME_NAMES.get(t.regime, f"regime_{t.regime}"),
                "confidence": t.confidence,
                "holding_days": t.holding_days,
                "position_size": t.position_size,
                "exit_reason": t.exit_reason,
                "fill_ratio": t.fill_ratio,
                "entry_impact_bps": t.entry_impact_bps,
                "exit_impact_bps": t.exit_impact_bps,
            })
        trade_df = pd.DataFrame(trade_data)

        if args.output:
            out_path = Path(args.output)
        else:
            out_path = RESULTS_DIR / f"backtest_{args.horizon}d_trades.csv"
        trade_df.to_csv(out_path, index=False)

        # Save summary
        summary = {
            "horizon": args.horizon,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "avg_return": result.avg_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "annualized_return": result.annualized_return,
            "trades_per_year": result.trades_per_year,
            "regime_breakdown": result.regime_breakdown,
        }
        summary_path = RESULTS_DIR / f"backtest_{args.horizon}d_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if verbose:
            print(f"\n  Trade log: {out_path}")
            print(f"  Summary: {summary_path}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
