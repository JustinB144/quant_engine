"""
Test module for kalshi hardening behavior and regressions.
"""

import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_engine.kalshi.client import KalshiClient, KalshiSigner
from quant_engine.kalshi.distribution import DistributionConfig, build_distribution_snapshot
from quant_engine.kalshi.events import (
    EventFeatureConfig,
    build_event_feature_panel,
    build_event_labels,
)
from quant_engine.kalshi.mapping_store import EventMarketMappingRecord, EventMarketMappingStore
from quant_engine.kalshi.options import add_options_disagreement_features, build_options_reference_panel
from quant_engine.kalshi.promotion import EventPromotionConfig, evaluate_event_promotion
from quant_engine.kalshi.provider import KalshiProvider
from quant_engine.kalshi.quality import (
    StalePolicy,
    compute_quality_dimensions,
    dynamic_stale_cutoff_minutes,
)
from quant_engine.kalshi.storage import EventTimeStore
from quant_engine.kalshi.walkforward import (
    EventWalkForwardConfig,
    evaluate_event_contract_metrics,
    run_event_walkforward,
)


class KalshiHardeningTests(unittest.TestCase):
    """Test cases covering kalshi hardening behavior and regression protections."""
    def test_bin_distribution_mass_normalizes_to_one(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1", "M1", "M1"],
                "bin_low": [2.0, 3.0, 4.0],
                "bin_high": [3.0, 4.0, 5.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z"],
                "mid": [0.2, 0.3, 0.5],
                "bid": [0.19, 0.29, 0.49],
                "ask": [0.21, 0.31, 0.51],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
            event_type="CPI",
        )
        self.assertAlmostEqual(sum(stats["_mass"]), 1.0, places=8)

    def test_threshold_direction_semantics_change_tail_probabilities(self):
        base_contracts = pd.DataFrame(
            {
                "contract_id": ["T1", "T2", "T3"],
                "market_id": ["M2", "M2", "M2"],
                "threshold_value": [2.0, 3.0, 4.0],
                "payout_structure": ["threshold", "threshold", "threshold"],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["T1", "T2", "T3"],
                "ts": ["2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z"],
                "mid": [0.40, 0.65, 0.20],
                "bid": [0.39, 0.64, 0.19],
                "ask": [0.41, 0.66, 0.21],
            },
        )
        cfg = DistributionConfig(
            price_scale="prob",
            tail_thresholds_by_event_type={"TEST": [3.0, 3.5, 4.0], "_default": [3.0, 3.5, 4.0]},
        )

        ge_contracts = base_contracts.assign(direction="above")
        le_contracts = base_contracts.assign(direction="below")

        ge = build_distribution_snapshot(
            contracts=ge_contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=cfg,
            event_type="TEST",
        )
        le = build_distribution_snapshot(
            contracts=le_contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=cfg,
            event_type="TEST",
        )

        self.assertNotAlmostEqual(float(ge["tail_p_1"]), float(le["tail_p_1"]), places=6)
        self.assertGreaterEqual(float(ge["tail_p_1"]), float(ge["tail_p_2"]))
        self.assertGreaterEqual(float(ge["tail_p_2"]), float(ge["tail_p_3"]))
        self.assertGreater(int(ge["monotonic_violations_pre"]), 0)

    def test_unknown_threshold_direction_marked_quality_low(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["U1", "U2", "U3"],
                "market_id": ["M3", "M3", "M3"],
                "threshold_value": [1.0, 2.0, 3.0],
                "direction": ["", "", ""],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["U1", "U2", "U3"],
                "ts": ["2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z"],
                "mid": [0.2, 0.4, 0.5],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
            event_type="TEST",
        )
        self.assertEqual(int(stats["quality_low"]), 1)
        self.assertTrue(pd.isna(stats["tail_p_1"]))

    def test_dynamic_stale_cutoff_tightens_near_event(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1", "M1", "M1"],
                "bin_low": [1.0, 2.0, 3.0],
                "bin_high": [2.0, 3.0, 4.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T13:03:00Z", "2025-01-01T13:03:00Z", "2025-01-01T13:03:00Z"],
                "mid": [0.2, 0.5, 0.3],
            },
        )
        cfg = DistributionConfig(
            price_scale="prob",
            near_event_minutes=60,
            near_event_stale_minutes=1,
            far_event_minutes=24 * 60,
            far_event_stale_minutes=60,
        )
        asof = pd.Timestamp("2025-01-01T13:05:00Z")

        near = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=asof,
            config=cfg,
            event_ts=pd.Timestamp("2025-01-01T13:35:00Z"),
            event_type="CPI",
        )
        far = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=asof,
            config=cfg,
            event_ts=pd.Timestamp("2025-01-03T13:35:00Z"),
            event_type="CPI",
        )

        self.assertLess(float(near["coverage_ratio"]), float(far["coverage_ratio"]))

    def test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity(self):
        policy = StalePolicy(
            base_stale_minutes=30.0,
            near_event_minutes=60.0,
            near_event_stale_minutes=2.0,
            far_event_minutes=24.0 * 60.0,
            far_event_stale_minutes=60.0,
            market_type_multipliers={"CPI": 0.8, "_default": 1.0},
            liquidity_low_threshold=2.0,
            liquidity_high_threshold=6.0,
            low_liquidity_multiplier=1.4,
            high_liquidity_multiplier=0.8,
        )
        low_liq = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=180.0,
            policy=policy,
            market_type="CPI",
            liquidity_proxy=1.0,
        )
        high_liq = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=180.0,
            policy=policy,
            market_type="CPI",
            liquidity_proxy=8.0,
        )
        self.assertGreater(low_liq, high_liq)

    def test_quality_score_behaves_sensibly_on_synthetic_cases(self):
        hi = compute_quality_dimensions(
            expected_contracts=10,
            observed_contracts=10,
            spreads=[0.01] * 10,
            quote_ages_seconds=[5] * 10,
            volumes=[50_000] * 10,
            open_interests=[10_000] * 10,
            violation_magnitude=0.0,
        )
        lo = compute_quality_dimensions(
            expected_contracts=10,
            observed_contracts=3,
            spreads=[0.8] * 3,
            quote_ages_seconds=[3600] * 3,
            volumes=[0] * 3,
            open_interests=[0] * 3,
            violation_magnitude=2.0,
        )
        self.assertGreater(float(hi.quality_score), float(lo.quality_score))

    def test_event_panel_supports_event_id_mapping(self):
        macro_events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "event_type": ["CPI"],
                "release_ts": ["2025-01-01T13:30:00Z"],
            },
        )
        event_map = pd.DataFrame(
            {
                "event_id": ["E1"],
                "event_type": ["OTHER"],
                "market_id": ["M1"],
            },
        )
        distributions = pd.DataFrame(
            {
                "market_id": ["M1"],
                "ts": ["2025-01-01T13:00:00Z"],
                "mean": [3.1],
                "var": [0.2],
                "skew": [0.0],
                "entropy": [1.0],
                "quality_score": [1.0],
            },
        )
        panel = build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=event_map,
            kalshi_distributions=distributions,
            config=EventFeatureConfig(snapshot_horizons=["15m"]),
        )
        self.assertEqual(len(panel), 1)

    def test_event_labels_first_vs_latest(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "release_ts": ["2025-01-01T13:30:00Z"],
                "event_type": ["CPI"],
            },
        )
        first = pd.DataFrame(
            {
                "event_id": ["E1"],
                "realized_value": [3.2],
                "release_ts": ["2025-01-01T13:30:00Z"],
                "asof_ts": ["2025-01-01T13:31:00Z"],
            },
        )
        revised = pd.DataFrame(
            {
                "event_id": ["E1"],
                "realized_value": [3.3],
                "release_ts": ["2025-01-01T13:30:00Z"],
                "asof_ts": ["2025-01-10T13:30:00Z"],
            },
        )
        y0 = build_event_labels(events, first, revised, label_mode="first_print")
        y1 = build_event_labels(events, first, revised, label_mode="latest")
        self.assertAlmostEqual(float(y0.iloc[0]["label_value"]), 3.2, places=8)
        self.assertAlmostEqual(float(y1.iloc[0]["label_value"]), 3.3, places=8)

    def test_walkforward_runs_and_counts_trials(self):
        n = 90
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        event_ids = [f"E{i:03d}" for i in range(n)]
        x = np.sin(np.linspace(0, 6, n))
        y = x + 0.05 * np.cos(np.linspace(0, 12, n))

        panel = pd.DataFrame(
            {
                "event_id": event_ids,
                "asof_ts": dates - pd.Timedelta(hours=1),
                "release_ts": dates,
                "mean": x,
                "var": np.abs(x) + 0.1,
                "entropy": np.abs(x) + 0.2,
            },
        ).set_index(["event_id", "asof_ts"])
        labels = pd.DataFrame({"event_id": event_ids, "label_value": y}).set_index("event_id")

        wf = run_event_walkforward(
            panel=panel,
            labels=labels,
            config=EventWalkForwardConfig(
                train_min_events=40,
                test_events_per_fold=15,
                step_events=15,
                alphas=(0.1, 1.0, 10.0),
            ),
        )
        self.assertGreater(len(wf.folds), 0)
        self.assertGreaterEqual(wf.n_trials_total, len(wf.folds) * 3)

    def test_walkforward_contract_metrics_are_computed(self):
        n = 80
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        event_ids = [f"E{i:03d}" for i in range(n)]
        x = np.sin(np.linspace(0, 8, n))
        y = x + 0.03 * np.cos(np.linspace(0, 10, n))
        event_types = np.where(np.arange(n) % 2 == 0, "CPI", "UNEMPLOYMENT")

        panel = pd.DataFrame(
            {
                "event_id": event_ids,
                "asof_ts": dates - pd.Timedelta(hours=1),
                "release_ts": dates,
                "event_type": event_types,
                "mean": x,
                "var": np.abs(x) + 0.1,
                "entropy": np.abs(x) + 0.2,
            },
        ).set_index(["event_id", "asof_ts"])
        labels = pd.DataFrame({"event_id": event_ids, "label_value": y}).set_index("event_id")

        wf = run_event_walkforward(
            panel=panel,
            labels=labels,
            config=EventWalkForwardConfig(
                train_min_events=30,
                test_events_per_fold=10,
                step_events=10,
                alphas=(0.1, 1.0, 10.0),
            ),
        )
        metrics = evaluate_event_contract_metrics(wf, n_bootstrap=120)
        self.assertIn("dsr_p_value", metrics)
        self.assertIn("mc_p_value", metrics)
        self.assertIn("capacity_utilization", metrics)
        self.assertIn("event_regime_stability", metrics)

    def test_event_promotion_flow_uses_walkforward_contract_metrics(self):
        n = 90
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        event_ids = [f"E{i:03d}" for i in range(n)]
        x = np.sin(np.linspace(0, 8, n))
        y = x + 0.02 * np.cos(np.linspace(0, 10, n))
        event_types = np.where(np.arange(n) % 3 == 0, "CPI", "UNEMPLOYMENT")

        panel = pd.DataFrame(
            {
                "event_id": event_ids,
                "asof_ts": dates - pd.Timedelta(hours=1),
                "release_ts": dates,
                "event_type": event_types,
                "mean": x,
                "var": np.abs(x) + 0.1,
                "entropy": np.abs(x) + 0.2,
            },
        ).set_index(["event_id", "asof_ts"])
        labels = pd.DataFrame({"event_id": event_ids, "label_value": y}).set_index("event_id")

        wf = run_event_walkforward(
            panel=panel,
            labels=labels,
            config=EventWalkForwardConfig(
                train_min_events=40,
                test_events_per_fold=15,
                step_events=15,
                alphas=(0.1, 1.0, 10.0),
            ),
        )
        decision = evaluate_event_promotion(
            walkforward_result=wf,
            config=EventPromotionConfig(strategy_id="kalshi_evt_test", horizon=1),
        )
        self.assertEqual(decision.candidate.strategy_id, "kalshi_evt_test")
        self.assertIn("worst_event_loss", decision.metrics)
        self.assertIn("surprise_hit_rate", decision.metrics)

    def test_options_disagreement_features_are_joined_asof(self):
        event_panel = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "asof_ts": ["2025-01-01T13:00:00Z", "2025-01-01T14:00:00Z"],
                "release_ts": ["2025-01-01T13:30:00Z", "2025-01-01T14:30:00Z"],
                "entropy": [0.9, 1.1],
                "tail_p_1": [0.35, 0.40],
                "speed_mean_per_hour": [0.05, -0.02],
            },
        ).set_index(["event_id", "asof_ts"])

        options = pd.DataFrame(
            {
                "ts": ["2025-01-01T12:30:00Z", "2025-01-01T13:30:00Z"],
                "iv_atm_30": [0.18, 0.22],
                "iv_atm_60": [0.19, 0.23],
                "iv_atm_90": [0.21, 0.24],
                "iv_put_25d": [0.24, 0.27],
                "iv_call_25d": [0.16, 0.19],
                "Close": [100.0, 101.0],
            },
        )

        ref = build_options_reference_panel(options, ts_col="ts")
        joined = add_options_disagreement_features(event_panel, ref, options_ts_col="ts")

        cols = set(joined.columns)
        self.assertIn("entropy_gap", cols)
        self.assertIn("tail_gap", cols)
        self.assertIn("repricing_speed_gap", cols)

    def test_mapping_store_asof(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "kalshi.sqlite"
            store = EventTimeStore(db_path=db_path, backend="sqlite")
            mapping = EventMarketMappingStore(store)
            mapping.upsert(
                [
                    EventMarketMappingRecord(
                        event_id="E1",
                        event_type="CPI",
                        market_id="M1",
                        effective_start_ts="2025-01-01T00:00:00Z",
                        effective_end_ts="2025-01-15T00:00:00Z",
                        mapping_version="v1",
                        source="manual",
                    ),
                    EventMarketMappingRecord(
                        event_id="E1",
                        event_type="CPI",
                        market_id="M2",
                        effective_start_ts="2025-01-15T00:00:00Z",
                        effective_end_ts=None,
                        mapping_version="v2",
                        source="manual",
                    ),
                ],
            )
            m_old = mapping.asof("2025-01-10T00:00:00Z")
            m_new = mapping.asof("2025-01-20T00:00:00Z")
            self.assertIn("M1", set(m_old["market_id"].astype(str)))
            self.assertIn("M2", set(m_new["market_id"].astype(str)))

    def test_store_ingestion_and_health_tables(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "kalshi.sqlite"
            store = EventTimeStore(db_path=db_path, backend="sqlite")
            store.upsert_ingestion_logs(
                [
                    {
                        "endpoint": "/markets",
                        "asof_date": "2025-01-01",
                        "source_env": "demo",
                        "ingest_ts": "2025-01-01T00:00:00Z",
                        "records_pulled": 10,
                        "missing_markets": 1,
                        "missing_contracts": 0,
                        "missing_bins": 0,
                        "p95_quote_age_seconds": 3.0,
                        "error_count": 0,
                        "notes": "ok",
                    },
                ],
            )
            store.upsert_daily_health_reports(
                [
                    {
                        "asof_date": "2025-01-01",
                        "markets_synced": 10,
                        "contracts_synced": 100,
                        "quotes_synced": 1000,
                        "avg_missing_fraction": 0.02,
                        "avg_quality_score": 0.95,
                        "total_constraint_violations": 0.1,
                        "p95_quote_age_seconds": 3.0,
                        "ingestion_errors": 0,
                        "updated_at": "2025-01-01T01:00:00Z",
                    },
                ],
            )
            logs = store.query_df("SELECT * FROM kalshi_ingestion_logs")
            health = store.query_df("SELECT * FROM kalshi_daily_health_report")
            self.assertEqual(len(logs), 1)
            self.assertEqual(len(health), 1)

    def test_provider_materializes_daily_health_report(self):
        class _DummyClient:
            environment = "demo"

            @staticmethod
            def available() -> bool:
                return False

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "kalshi.sqlite"
            store = EventTimeStore(db_path=db_path, backend="sqlite")
            provider = KalshiProvider(client=_DummyClient(), store=store)

            store.upsert_coverage_diagnostics(
                [
                    {
                        "market_id": "M1",
                        "asof_date": "2025-01-01",
                        "expected_contracts": 10,
                        "observed_contracts": 9,
                        "missing_fraction": 0.1,
                        "average_spread": 0.02,
                        "median_quote_age_seconds": 4.0,
                        "constraint_violations": 0.2,
                        "quality_score": 0.9,
                    },
                ],
            )
            store.upsert_ingestion_logs(
                [
                    {
                        "endpoint": "/markets",
                        "asof_date": "2025-01-01",
                        "source_env": "demo",
                        "ingest_ts": "2025-01-01T00:00:00Z",
                        "records_pulled": 4,
                        "missing_markets": 0,
                        "missing_contracts": 0,
                        "missing_bins": 0,
                        "p95_quote_age_seconds": 1.0,
                        "error_count": 0,
                        "notes": "",
                    },
                    {
                        "endpoint": "/markets/{id}/contracts",
                        "asof_date": "2025-01-01",
                        "source_env": "demo",
                        "ingest_ts": "2025-01-01T00:01:00Z",
                        "records_pulled": 30,
                        "missing_markets": 0,
                        "missing_contracts": 1,
                        "missing_bins": 2,
                        "p95_quote_age_seconds": 2.0,
                        "error_count": 0,
                        "notes": "",
                    },
                    {
                        "endpoint": "/contracts/{id}/quotes",
                        "asof_date": "2025-01-01",
                        "source_env": "demo",
                        "ingest_ts": "2025-01-01T00:02:00Z",
                        "records_pulled": 500,
                        "missing_markets": 0,
                        "missing_contracts": 0,
                        "missing_bins": 0,
                        "p95_quote_age_seconds": 3.0,
                        "error_count": 1,
                        "notes": "",
                    },
                ],
            )

            report = provider.materialize_daily_health_report()
            self.assertEqual(len(report), 1)
            row = report.iloc[0]
            self.assertEqual(str(row["asof_date"]), "2025-01-01")
            self.assertGreater(float(row["quotes_synced"]), 0.0)

    def test_signer_canonical_payload_and_header_fields(self):
        captured = {}

        def sign_func(payload: bytes) -> bytes:
            captured["payload"] = payload.decode("utf-8")
            return b"abc"

        signer = KalshiSigner(access_key="demo_key", sign_func=sign_func)
        sig = signer.sign("123456", "GET", "/markets?cursor=xyz")
        self.assertEqual(sig, base64.b64encode(b"abc").decode("ascii"))
        self.assertEqual(captured["payload"], "123456GET/markets")

        client = KalshiClient(signer=signer, base_url="https://demo-api.kalshi.co/trade-api/v2")
        headers = client._auth_headers("GET", "/markets")
        self.assertIn("KALSHI-ACCESS-KEY", headers)
        self.assertIn("KALSHI-ACCESS-TIMESTAMP", headers)
        self.assertIn("KALSHI-ACCESS-SIGNATURE", headers)


if __name__ == "__main__":
    unittest.main()
