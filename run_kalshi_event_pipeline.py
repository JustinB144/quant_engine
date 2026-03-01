#!/usr/bin/env python3
"""
Run the integrated Kalshi event-time pipeline inside quant_engine.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from quant_engine.config import (
    KALSHI_DB_PATH,
    KALSHI_DISTANCE_LAGS,
    KALSHI_DISTRIBUTION_FREQ,
    KALSHI_ENABLED,
    KALSHI_FAR_EVENT_MINUTES,
    KALSHI_FAR_EVENT_STALE_MINUTES,
    KALSHI_NEAR_EVENT_MINUTES,
    KALSHI_NEAR_EVENT_STALE_MINUTES,
    KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER,
    KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD,
    KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD,
    KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER,
    KALSHI_STALE_MARKET_TYPE_MULTIPLIERS,
    KALSHI_SNAPSHOT_HORIZONS,
    KALSHI_STALE_AFTER_MINUTES,
    KALSHI_TAIL_THRESHOLDS,
    RESULTS_DIR,
)
from quant_engine.kalshi.distribution import DistributionConfig
from quant_engine.kalshi.events import EventFeatureConfig, build_event_labels
from quant_engine.kalshi.pipeline import KalshiPipeline
from quant_engine.kalshi.promotion import EventPromotionConfig
from quant_engine.kalshi.walkforward import EventWalkForwardConfig


def _read_df(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame based on the file extension."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main():
    """Run Kalshi ingestion, distribution building, event features, walk-forward evaluation, and reporting tasks."""
    parser = argparse.ArgumentParser(description="Kalshi event-time pipeline")
    parser.add_argument("--db-path", type=str, default=str(KALSHI_DB_PATH))
    parser.add_argument("--backend", choices=["duckdb", "sqlite"], default="duckdb")
    parser.add_argument("--start-ts", type=str, default=None)
    parser.add_argument("--end-ts", type=str, default=None)

    parser.add_argument("--sync-reference", action="store_true")
    parser.add_argument("--sync-quotes", action="store_true")
    parser.add_argument("--build-distributions", action="store_true")
    parser.add_argument("--build-event-features", action="store_true")

    parser.add_argument("--event-map", type=str, help="CSV/Parquet with event_id/event_type + market_id")
    parser.add_argument("--options-reference", type=str, default=None, help="CSV/Parquet options panel for disagreement features")
    parser.add_argument("--options-ts-col", type=str, default="ts", help="Timestamp column in options reference panel")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "kalshi_event_features.parquet"))

    parser.add_argument("--labels-first-print", type=str, default=None)
    parser.add_argument("--labels-revised", type=str, default=None)
    parser.add_argument("--run-walkforward", action="store_true")
    parser.add_argument("--disable-promotion-gate", action="store_true")
    parser.add_argument("--strategy-id", type=str, default="kalshi_event_default")

    parser.add_argument("--build-health-report", action="store_true", help="Materialize daily health report aggregates")
    parser.add_argument("--print-health-report", action="store_true", help="Print latest daily health report table")
    parser.add_argument("--health-report-output", type=str, default=None, help="CSV/Parquet output path for health report")

    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not KALSHI_ENABLED:
        print("KALSHI_ENABLED is False in config.py. Enable it first.")
        return

    verbose = not args.quiet
    pipe = KalshiPipeline.from_store(db_path=args.db_path, backend=args.backend)

    report = {}
    if args.sync_reference:
        if verbose:
            print("Syncing Kalshi market catalog/contracts...")
        report["sync_reference"] = pipe.sync_reference()

    if args.sync_quotes:
        if verbose:
            print("Syncing Kalshi intraday quotes...")
        report["quotes_synced"] = pipe.sync_intraday_quotes(
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )

    if args.build_health_report or args.print_health_report or args.health_report_output:
        health = (
            pipe.provider.materialize_daily_health_report()
            if args.build_health_report
            else pipe.provider.get_daily_health_report()
        )
        report["health_report_rows"] = int(len(health))
        if args.health_report_output:
            hp = Path(args.health_report_output)
            hp.parent.mkdir(parents=True, exist_ok=True)
            if hp.suffix.lower() == ".csv":
                health.to_csv(hp, index=False)
            else:
                health.to_parquet(hp, index=False)
            report["health_report_output"] = str(hp)
        if args.print_health_report and verbose:
            print("\nKalshi Daily Health Report")
            if len(health) == 0:
                print("(empty)")
            else:
                print(health.to_string(index=False))

    if args.build_distributions:
        if verbose:
            print("Building intraday market distributions...")
        panel = pipe.build_distributions(
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            freq=KALSHI_DISTRIBUTION_FREQ,
            config=DistributionConfig(
                stale_after_minutes=int(KALSHI_STALE_AFTER_MINUTES),
                near_event_minutes=float(KALSHI_NEAR_EVENT_MINUTES),
                near_event_stale_minutes=float(KALSHI_NEAR_EVENT_STALE_MINUTES),
                far_event_minutes=float(KALSHI_FAR_EVENT_MINUTES),
                far_event_stale_minutes=float(KALSHI_FAR_EVENT_STALE_MINUTES),
                stale_market_type_multipliers=dict(KALSHI_STALE_MARKET_TYPE_MULTIPLIERS),
                stale_liquidity_low_threshold=float(KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD),
                stale_liquidity_high_threshold=float(KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD),
                stale_low_liquidity_multiplier=float(KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER),
                stale_high_liquidity_multiplier=float(KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER),
                tail_thresholds_by_event_type=dict(KALSHI_TAIL_THRESHOLDS),
                distance_lags=list(KALSHI_DISTANCE_LAGS),
            ),
        )
        report["distribution_rows"] = int(len(panel))

    features = None
    if args.build_event_features or args.event_map:
        mapping = _read_df(args.event_map) if args.event_map else None
        opt_ref = _read_df(args.options_reference) if args.options_reference else None
        cfg = EventFeatureConfig(snapshot_horizons=KALSHI_SNAPSHOT_HORIZONS)
        features = pipe.build_event_features(
            event_market_map=mapping,
            asof_ts=args.end_ts,
            config=cfg,
            options_reference=opt_ref,
            options_ts_col=args.options_ts_col,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            features.reset_index().to_csv(out_path, index=False)
        else:
            features.reset_index().to_parquet(out_path)
        report["event_feature_rows"] = int(len(features))
        report["event_feature_output"] = str(out_path)

    if args.run_walkforward:
        if features is None:
            raise ValueError("run-walkforward requires event features. Use --build-event-features.")
        if not args.labels_first_print:
            raise ValueError("run-walkforward requires --labels-first-print dataset.")

        first = _read_df(args.labels_first_print)
        revised = _read_df(args.labels_revised) if args.labels_revised else None
        macro_events = pipe.provider.get_macro_events()
        labels = build_event_labels(
            macro_events=macro_events,
            event_outcomes_first_print=first,
            event_outcomes_revised=revised,
            label_mode="latest" if revised is not None else "first_print",
        )
        wf = pipe.run_walkforward(
            event_features=features,
            labels=labels,
            config=EventWalkForwardConfig(),
            label_col="label_value",
        )
        contract = pipe.evaluate_walkforward_contract(
            walkforward_result=wf,
            n_bootstrap=500,
            max_events_per_day=6.0,
        )
        report["walkforward"] = {
            "n_folds": len(wf.folds),
            **wf.to_metrics(),
            **contract,
        }
        if not args.disable_promotion_gate:
            decision = pipe.evaluate_event_promotion(
                walkforward_result=wf,
                promotion_config=EventPromotionConfig(
                    strategy_id=str(args.strategy_id),
                    horizon=1,
                    entry_threshold=0.0,
                    confidence_threshold=0.0,
                    use_risk_management=False,
                    max_positions=1,
                    position_size_pct=1.0,
                ),
                extra_contract_metrics={
                    **contract,
                    "wf_oos_corr": float(wf.wf_oos_corr),
                    "wf_positive_fold_fraction": float(wf.wf_positive_fold_fraction),
                    "wf_is_oos_gap": float(wf.wf_is_oos_gap),
                },
            )
            report["promotion"] = {
                "strategy_id": decision.candidate.strategy_id,
                "passed": bool(decision.passed),
                "score": float(decision.score),
                "reasons": list(decision.reasons),
                "metrics": dict(decision.metrics),
            }

    # ── Write reproducibility manifest ──
    try:
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest
        manifest = build_run_manifest(
            run_type="kalshi_event_pipeline",
            config_snapshot=vars(args),
            script_name="run_kalshi_event_pipeline",
        )
        manifest["extra"] = report
        write_run_manifest(manifest, output_dir=RESULTS_DIR)
    except Exception as e:
        logger.warning("Could not write reproducibility manifest: %s", e)

    if verbose:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
