"""
Kalshi provider: ingestion + storage + feature-ready retrieval.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    KALSHI_API_BASE_URL,
    KALSHI_DISTANCE_LAGS,
    KALSHI_ENV,
    KALSHI_FAR_EVENT_MINUTES,
    KALSHI_FAR_EVENT_STALE_MINUTES,
    KALSHI_HISTORICAL_API_BASE_URL,
    KALSHI_HISTORICAL_CUTOFF_TS,
    KALSHI_NEAR_EVENT_MINUTES,
    KALSHI_NEAR_EVENT_STALE_MINUTES,
    KALSHI_RATE_LIMIT_BURST,
    KALSHI_RATE_LIMIT_RPS,
    KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER,
    KALSHI_STALE_AFTER_MINUTES,
    KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD,
    KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD,
    KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER,
    KALSHI_STALE_MARKET_TYPE_MULTIPLIERS,
    KALSHI_TAIL_THRESHOLDS,
)
from .client import KalshiClient, RateLimitPolicy
from .distribution import DistributionConfig, build_distribution_panel
from .mapping_store import EventMarketMappingStore
from .storage import EventTimeStore


def _to_iso_utc(value: object) -> Optional[str]:
    """Normalize a timestamp-like value to an ISO-8601 UTC string, returning None on failure."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    except (ValueError, KeyError, TypeError):
        return None


def _safe_hash_text(text: str) -> str:
    """Return a stable SHA-256 hash for text fields used in spec/provenance snapshots."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _asof_date(value: object) -> str:
    """Convert a timestamp-like value to a UTC calendar date string for daily rollups."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return str(ts.date())


class KalshiProvider:
    """
    Provider interface similar to WRDSProvider, but for event-market data.
    """

    def __init__(
        self,
        client: Optional[KalshiClient] = None,
        store: Optional[EventTimeStore] = None,
    ):
        """Initialize KalshiProvider."""
        self.client = client or KalshiClient(
            base_url=KALSHI_API_BASE_URL,
            environment=KALSHI_ENV,
            historical_base_url=KALSHI_HISTORICAL_API_BASE_URL,
            historical_cutoff_ts=KALSHI_HISTORICAL_CUTOFF_TS,
            rate_limit_policy=RateLimitPolicy(
                requests_per_second=KALSHI_RATE_LIMIT_RPS,
                burst=KALSHI_RATE_LIMIT_BURST,
            ),
        )
        self.store = store

    def available(self) -> bool:
        """Return whether the resource is available in the current runtime."""
        return bool(self.client and self.client.available())

    def sync_account_limits(self) -> Dict[str, object]:
        """Synchronize account limits with local storage."""
        if not self.available():
            return {}
        try:
            return self.client.get_account_limits()
        except (OSError, ValueError, RuntimeError):
            return {}

    def refresh_historical_cutoff(self) -> Optional[str]:
        """refresh historical cutoff."""
        if not self.available():
            return None
        try:
            return self.client.fetch_historical_cutoff()
        except (OSError, ValueError, RuntimeError):
            return None

    def sync_market_catalog(
        self,
        status: Optional[str] = None,
        mapping_version: str = "v1",
        mapping_source: str = "api",
    ) -> int:
        """Synchronize market catalog with local storage."""
        if not self.available():
            return 0

        error_count = 0
        try:
            markets = self.client.list_markets(status=status)
        except (OSError, ValueError, RuntimeError):
            markets = []
            error_count += 1
        spec_ts = datetime.now(timezone.utc).isoformat()
        rows: List[Dict[str, object]] = []
        spec_rows: List[Dict[str, object]] = []
        mapping_rows: List[Dict[str, object]] = []
        missing_markets = 0

        for m in markets:
            market_id = str(m.get("ticker") or m.get("market_id") or "")
            if not market_id:
                missing_markets += 1
                continue
            event_id = str(m.get("event_ticker") or m.get("event_id") or "")
            event_type = str(m.get("event_type", ""))
            rules_text = str(m.get("rules_primary", m.get("rules_text", "")))
            open_ts = _to_iso_utc(m.get("open_time") or m.get("open_ts"))
            close_ts = _to_iso_utc(m.get("close_time") or m.get("close_ts"))
            settle_ts = _to_iso_utc(m.get("settlement_time") or m.get("settle_ts"))

            rows.append(
                {
                    "market_id": market_id,
                    "event_id": event_id,
                    "event_type": event_type,
                    "title": str(m.get("title", "")),
                    "rules_text": rules_text,
                    "rules_hash": _safe_hash_text(rules_text),
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "settle_ts": settle_ts,
                    "status": str(m.get("status", "")),
                    "spec_version_ts": spec_ts,
                    "raw_market_json": json.dumps(m, sort_keys=True),
                },
            )

            spec_rows.append(
                {
                    "market_id": market_id,
                    "spec_version_ts": spec_ts,
                    "rules_text": rules_text,
                    "rules_hash": _safe_hash_text(rules_text),
                    "raw_market_json": json.dumps(m, sort_keys=True),
                    "source": "kalshi_api",
                },
            )

            if event_id:
                mapping_rows.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "market_id": market_id,
                        "effective_start_ts": open_ts or spec_ts,
                        "effective_end_ts": None,
                        "mapping_version": str(mapping_version),
                        "source": str(mapping_source),
                    },
                )

        if self.store is not None:
            self.store.upsert_markets(rows)
            self.store.append_market_specs(spec_rows)
            if mapping_rows:
                self.store.upsert_event_market_map_versions(mapping_rows)
            self.store.upsert_data_provenance(
                [
                    {
                        "market_id": "*",
                        "asof_date": _asof_date(spec_ts),
                        "source_env": self.client.environment,
                        "endpoint": "/markets",
                        "ingest_ts": spec_ts,
                        "records_pulled": len(rows),
                        "notes": "catalog sync",
                    },
                ],
            )
            self.store.upsert_ingestion_logs(
                [
                    {
                        "endpoint": "/markets",
                        "asof_date": _asof_date(spec_ts),
                        "source_env": self.client.environment,
                        "ingest_ts": spec_ts,
                        "records_pulled": len(rows),
                        "missing_markets": int(missing_markets),
                        "missing_contracts": 0,
                        "missing_bins": 0,
                        "p95_quote_age_seconds": np.nan,
                        "error_count": int(error_count),
                        "notes": "catalog sync",
                    },
                ],
            )
        return len(rows)

    def sync_contracts(self, market_ids: Iterable[str]) -> int:
        """Synchronize contracts with local storage."""
        if not self.available():
            return 0

        spec_ts = datetime.now(timezone.utc).isoformat()
        rows: List[Dict[str, object]] = []
        spec_rows: List[Dict[str, object]] = []
        error_count = 0
        missing_contracts = 0
        missing_bins = 0

        for market_id in market_ids:
            try:
                contracts = self.client.list_contracts(str(market_id))
            except (OSError, ValueError, RuntimeError):
                error_count += 1
                continue
            for c in contracts:
                contract_id = str(c.get("ticker") or c.get("contract_id") or "")
                if not contract_id:
                    missing_contracts += 1
                    continue

                row = {
                    "contract_id": contract_id,
                    "market_id": str(market_id),
                    "bin_low": c.get("floor_strike", c.get("bin_low")),
                    "bin_high": c.get("cap_strike", c.get("bin_high")),
                    "threshold_value": c.get("strike", c.get("threshold_value")),
                    "payout_structure": str(c.get("settlement_type", c.get("payout_structure", "binary"))),
                    "direction": str(c.get("direction", "")),
                    "tick_size": c.get("tick_size", None),
                    "fee_bps": c.get("fee_bps", None),
                    "status": str(c.get("status", "")),
                    "spec_version_ts": spec_ts,
                    "raw_contract_json": json.dumps(c, sort_keys=True),
                }
                has_bin_bounds = (row["bin_low"] is not None) or (row["bin_high"] is not None)
                has_threshold = row["threshold_value"] is not None
                if not has_bin_bounds and not has_threshold:
                    missing_bins += 1
                rows.append(row)
                spec_rows.append(
                    {
                        "contract_id": contract_id,
                        "spec_version_ts": spec_ts,
                        "market_id": str(market_id),
                        "bin_low": row["bin_low"],
                        "bin_high": row["bin_high"],
                        "threshold_value": row["threshold_value"],
                        "payout_structure": row["payout_structure"],
                        "direction": row["direction"],
                        "status": row["status"],
                        "raw_contract_json": row["raw_contract_json"],
                    },
                )

        if self.store is not None:
            self.store.upsert_contracts(rows)
            self.store.append_contract_specs(spec_rows)
            self.store.upsert_data_provenance(
                [
                    {
                        "market_id": "*",
                        "asof_date": _asof_date(spec_ts),
                        "source_env": self.client.environment,
                        "endpoint": "/markets/{id}/contracts",
                        "ingest_ts": spec_ts,
                        "records_pulled": len(rows),
                        "notes": "contract sync",
                    },
                ],
            )
            self.store.upsert_ingestion_logs(
                [
                    {
                        "endpoint": "/markets/{id}/contracts",
                        "asof_date": _asof_date(spec_ts),
                        "source_env": self.client.environment,
                        "ingest_ts": spec_ts,
                        "records_pulled": len(rows),
                        "missing_markets": 0,
                        "missing_contracts": int(missing_contracts),
                        "missing_bins": int(missing_bins),
                        "p95_quote_age_seconds": np.nan,
                        "error_count": int(error_count),
                        "notes": "contract sync",
                    },
                ],
            )
        return len(rows)

    def sync_quotes(
        self,
        contract_ids: Iterable[str],
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> int:
        """Synchronize quotes with local storage."""
        if not self.available():
            return 0

        rows: List[Dict[str, object]] = []
        error_count = 0
        missing_contracts = 0
        ingest_ts = datetime.now(timezone.utc).isoformat()
        for contract_id in contract_ids:
            cid = str(contract_id).strip()
            if not cid:
                missing_contracts += 1
                continue
            try:
                quotes = self.client.list_quotes(
                    contract_id=cid,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            except (OSError, ValueError, RuntimeError):
                error_count += 1
                continue
            for q in quotes:
                bid = q.get("bid", None)
                ask = q.get("ask", None)
                mid = q.get("mid", None)
                if mid is None and bid is not None and ask is not None:
                    try:
                        mid = (float(bid) + float(ask)) / 2.0
                    except (ValueError, TypeError):
                        mid = None
                rows.append(
                    {
                        "contract_id": cid,
                        "ts": _to_iso_utc(q.get("ts") or q.get("created_time")),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": q.get("last", q.get("price")),
                        "volume": q.get("volume", None),
                        "oi": q.get("open_interest", None),
                        "market_status": str(q.get("market_status", "")),
                    },
                )

        p95_age_seconds = np.nan
        if rows:
            qdf = pd.DataFrame(rows)
            ingest = pd.to_datetime(ingest_ts, utc=True, errors="coerce")
            qdf = qdf.assign(ts=pd.to_datetime(qdf["ts"], utc=True, errors="coerce"))
            valid = qdf["ts"].notna() & pd.notna(ingest)
            if valid.any():
                ages = (ingest - qdf.loc[valid, "ts"]).dt.total_seconds()
                if ages.notna().any():
                    p95_age_seconds = float(np.nanpercentile(ages.to_numpy(dtype=float), 95.0))

        if self.store is not None:
            self.store.append_quotes(rows)
            self.store.upsert_data_provenance(
                [
                    {
                        "market_id": "*",
                        "asof_date": _asof_date(ingest_ts),
                        "source_env": self.client.environment,
                        "endpoint": "/contracts/{id}/quotes",
                        "ingest_ts": ingest_ts,
                        "records_pulled": len(rows),
                        "notes": "quote sync",
                    },
                ],
            )
            self.store.upsert_ingestion_logs(
                [
                    {
                        "endpoint": "/contracts/{id}/quotes",
                        "asof_date": _asof_date(ingest_ts),
                        "source_env": self.client.environment,
                        "ingest_ts": ingest_ts,
                        "records_pulled": len(rows),
                        "missing_markets": 0,
                        "missing_contracts": int(missing_contracts),
                        "missing_bins": 0,
                        "p95_quote_age_seconds": p95_age_seconds,
                        "error_count": int(error_count),
                        "notes": "quote sync",
                    },
                ],
            )
        return len(rows)

    def get_markets(self) -> pd.DataFrame:
        """Return markets."""
        if self.store is None:
            return pd.DataFrame()
        return self.store.query_df("SELECT * FROM kalshi_markets")

    def get_contracts(self) -> pd.DataFrame:
        """Return contracts."""
        if self.store is None:
            return pd.DataFrame()
        return self.store.query_df("SELECT * FROM kalshi_contracts")

    def get_quotes(
        self,
        market_id: Optional[str] = None,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return quotes."""
        if self.store is None:
            return pd.DataFrame()
        sql = """
            SELECT q.*
            FROM kalshi_quotes q
            LEFT JOIN kalshi_contracts c ON q.contract_id = c.contract_id
            WHERE 1=1
        """
        params: List[object] = []
        if market_id is not None:
            sql += " AND c.market_id = ?"
            params.append(str(market_id))
        if start_ts is not None:
            sql += " AND q.ts >= ?"
            params.append(str(start_ts))
        if end_ts is not None:
            sql += " AND q.ts <= ?"
            params.append(str(end_ts))
        sql += " ORDER BY q.contract_id, q.ts"
        return self.store.query_df(sql, params=params)

    def get_event_market_map_asof(self, asof_ts: str) -> pd.DataFrame:
        """Return event market map asof."""
        if self.store is None:
            return pd.DataFrame()
        return EventMarketMappingStore(self.store).asof(asof_ts)

    def get_macro_events(self, versioned: bool = False) -> pd.DataFrame:
        """Return macro events."""
        if self.store is None:
            return pd.DataFrame()
        if versioned:
            return self.store.query_df("SELECT * FROM macro_events_versioned ORDER BY event_id, version_ts")
        return self.store.query_df("SELECT * FROM macro_events ORDER BY release_ts")

    def get_event_outcomes(self, table: str = "first_print") -> pd.DataFrame:
        """Return event outcomes."""
        if self.store is None:
            return pd.DataFrame()
        t = str(table).lower().strip()
        if t in {"first", "first_print", "event_outcomes_first_print"}:
            return self.store.query_df(
                "SELECT * FROM event_outcomes_first_print ORDER BY event_id, asof_ts",
            )
        if t in {"revised", "event_outcomes_revised"}:
            return self.store.query_df(
                "SELECT * FROM event_outcomes_revised ORDER BY event_id, asof_ts",
            )
        return self.store.query_df("SELECT * FROM event_outcomes ORDER BY event_id, asof_ts")

    def compute_and_store_distributions(
        self,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        freq: str = "5min",
        config: Optional[DistributionConfig] = None,
    ) -> pd.DataFrame:
        """Compute and store distributions."""
        if self.store is None:
            return pd.DataFrame()

        markets = self.get_markets()
        contracts = self.get_contracts()
        quotes = self.get_quotes(start_ts=start_ts, end_ts=end_ts)
        if markets.empty or contracts.empty or quotes.empty:
            return pd.DataFrame()

        quotes = quotes.assign(ts=pd.to_datetime(quotes["ts"], utc=True, errors="coerce"))
        q = quotes[quotes["ts"].notna()].copy()
        if q.empty:
            return pd.DataFrame()

        cfg = config or DistributionConfig(
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
        )

        t0 = q["ts"].min()
        t1 = q["ts"].max()
        snapshots = pd.date_range(start=t0, end=t1, freq=freq, tz="UTC")
        panel = build_distribution_panel(
            markets=markets,
            contracts=contracts,
            quotes=q,
            snapshot_times=snapshots,
            config=cfg,
        )
        if panel.empty:
            return panel

        self.store.upsert_distributions(panel.to_dict(orient="records"))

        # Diagnostics for reproducibility/health.
        contract_counts = (
            contracts.assign(market_id=contracts["market_id"].astype(str))
            .groupby("market_id")["contract_id"]
            .count()
            .to_dict()
        )
        diags = []
        for (market_id, ts_date), block in panel.groupby(["market_id", panel["ts"].dt.date]):
            expected = int(contract_counts.get(str(market_id), 0))
            coverage = pd.to_numeric(block.get("coverage_ratio"), errors="coerce")
            observed_est = int(round(float(np.nanmedian(coverage)) * expected)) if expected > 0 and coverage.notna().any() else 0
            diags.append(
                {
                    "market_id": str(market_id),
                    "asof_date": str(ts_date),
                    "expected_contracts": expected,
                    "observed_contracts": observed_est,
                    "missing_fraction": float(max(expected - observed_est, 0) / max(expected, 1)),
                    "average_spread": float(pd.to_numeric(block.get("median_spread"), errors="coerce").mean()),
                    "median_quote_age_seconds": float(pd.to_numeric(block.get("median_quote_age_seconds"), errors="coerce").median()),
                    "constraint_violations": float(pd.to_numeric(block.get("monotonic_violation_magnitude"), errors="coerce").mean()),
                    "quality_score": float(pd.to_numeric(block.get("quality_score"), errors="coerce").mean()),
                },
            )
        self.store.upsert_coverage_diagnostics(diags)
        self.materialize_daily_health_report()

        return panel

    def materialize_daily_health_report(self) -> pd.DataFrame:
        """
        Build and persist daily Kalshi ingestion/coverage health aggregates.
        """
        if self.store is None:
            return pd.DataFrame()

        cov = self.store.query_df("SELECT * FROM kalshi_coverage_diagnostics")
        logs = self.store.query_df("SELECT * FROM kalshi_ingestion_logs")

        cov_daily = pd.DataFrame()
        if len(cov) > 0:
            cov = cov.copy(deep=True)
            cov.loc[:, "asof_date"] = cov["asof_date"].astype(str)
            cov_daily = (
                cov.groupby("asof_date", as_index=False)
                .agg(
                    avg_missing_fraction=("missing_fraction", "mean"),
                    avg_quality_score=("quality_score", "mean"),
                    total_constraint_violations=("constraint_violations", "sum"),
                )
            )

        log_daily = pd.DataFrame()
        if len(logs) > 0:
            logs = logs.copy(deep=True)
            logs.loc[:, "asof_date"] = logs["asof_date"].astype(str)
            logs.loc[:, "endpoint"] = logs["endpoint"].astype(str)
            logs.loc[:, "records_pulled"] = pd.to_numeric(logs["records_pulled"], errors="coerce").fillna(0.0)
            logs.loc[:, "error_count"] = pd.to_numeric(logs["error_count"], errors="coerce").fillna(0.0)
            logs.loc[:, "p95_quote_age_seconds"] = pd.to_numeric(logs["p95_quote_age_seconds"], errors="coerce")

            by_day = logs.groupby("asof_date")
            log_daily = by_day.agg(
                quotes_synced=("records_pulled", lambda s: float(s[logs.loc[s.index, "endpoint"] == "/contracts/{id}/quotes"].sum())),
                contracts_synced=("records_pulled", lambda s: float(s[logs.loc[s.index, "endpoint"] == "/markets/{id}/contracts"].sum())),
                markets_synced=("records_pulled", lambda s: float(s[logs.loc[s.index, "endpoint"] == "/markets"].sum())),
                ingestion_errors=("error_count", "sum"),
                p95_quote_age_seconds=("p95_quote_age_seconds", "max"),
            ).reset_index()

        if len(cov_daily) == 0 and len(log_daily) == 0:
            return pd.DataFrame()

        if len(cov_daily) == 0:
            report = log_daily.copy()
            report["avg_missing_fraction"] = np.nan
            report["avg_quality_score"] = np.nan
            report["total_constraint_violations"] = np.nan
        elif len(log_daily) == 0:
            report = cov_daily.copy()
            report["quotes_synced"] = 0.0
            report["contracts_synced"] = 0.0
            report["markets_synced"] = 0.0
            report["ingestion_errors"] = 0.0
            report["p95_quote_age_seconds"] = np.nan
        else:
            report = cov_daily.merge(log_daily, on="asof_date", how="outer")

        report = report.copy(deep=True)
        report.loc[:, "updated_at"] = datetime.now(timezone.utc).isoformat()
        self.store.upsert_daily_health_reports(report.to_dict(orient="records"))
        return report.sort_values("asof_date").reset_index(drop=True)

    def get_daily_health_report(self) -> pd.DataFrame:
        """Return daily health report."""
        if self.store is None:
            return pd.DataFrame()
        return self.store.query_df(
            "SELECT * FROM kalshi_daily_health_report ORDER BY asof_date",
        )

    def store_clock_check(self) -> Dict[str, object]:
        """store clock check."""
        server_now = self.client.server_time_utc()
        local_now = datetime.now(timezone.utc)
        return {
            "server_time_utc": server_now.isoformat(),
            "local_time_utc": local_now.isoformat(),
            "clock_skew_seconds": float((local_now - server_now).total_seconds()),
        }
