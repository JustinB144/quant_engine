"""
Orchestration helpers for the Kalshi event-market vertical.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .distribution import DistributionConfig
from .events import EventFeatureConfig, build_event_feature_panel
from .options import add_options_disagreement_features, build_options_reference_panel
from .promotion import EventPromotionConfig, evaluate_event_promotion
from .provider import KalshiProvider
from .storage import EventTimeStore
from .walkforward import (
    EventWalkForwardConfig,
    EventWalkForwardResult,
    evaluate_event_contract_metrics,
    run_event_walkforward,
)
from ..autopilot.promotion_gate import PromotionDecision


@dataclass
class KalshiPipeline:
    """High-level orchestration wrapper for Kalshi sync, feature, walk-forward, and promotion workflows."""
    provider: KalshiProvider
    store: EventTimeStore

    @classmethod
    def from_store(
        cls,
        db_path: Path | str,
        backend: str = "duckdb",
        provider: Optional[KalshiProvider] = None,
    ) -> "KalshiPipeline":
        """from store."""
        store = EventTimeStore(db_path=db_path, backend=backend)
        prov = provider or KalshiProvider(store=store)
        if prov.store is None:
            prov.store = store
        return cls(provider=prov, store=store)

    def sync_reference(self, status: Optional[str] = None, mapping_version: str = "v1") -> dict:
        """Synchronize reference with local storage."""
        self.provider.sync_account_limits()
        self.provider.refresh_historical_cutoff()
        n_markets = self.provider.sync_market_catalog(
            status=status,
            mapping_version=mapping_version,
        )
        markets = self.provider.get_markets()
        market_ids = markets["market_id"].dropna().astype(str).tolist() if len(markets) > 0 else []
        n_contracts = self.provider.sync_contracts(market_ids)
        health = self.provider.materialize_daily_health_report()
        return {
            "markets_synced": int(n_markets),
            "contracts_synced": int(n_contracts),
            "health_days_materialized": int(len(health)),
        }

    def sync_intraday_quotes(
        self,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> int:
        """Synchronize intraday quotes with local storage."""
        contracts = self.provider.get_contracts()
        if len(contracts) == 0:
            return 0
        contract_ids = contracts["contract_id"].dropna().astype(str).tolist()
        return int(
            self.provider.sync_quotes(
                contract_ids=contract_ids,
                start_ts=start_ts,
                end_ts=end_ts,
            ),
        )

    def build_distributions(
        self,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        freq: str = "5min",
        config: Optional[DistributionConfig] = None,
    ) -> pd.DataFrame:
        """Build distributions."""
        cfg = config or DistributionConfig()
        return self.provider.compute_and_store_distributions(
            start_ts=start_ts,
            end_ts=end_ts,
            freq=freq,
            config=cfg,
        )

    def build_event_features(
        self,
        event_market_map: Optional[pd.DataFrame] = None,
        asof_ts: Optional[str] = None,
        config: Optional[EventFeatureConfig] = None,
        options_reference: Optional[pd.DataFrame] = None,
        options_ts_col: str = "ts",
    ) -> pd.DataFrame:
        """Build event features."""
        macro_events = self.provider.get_macro_events()
        dists = self.store.query_df("SELECT * FROM kalshi_distributions")

        mapping = event_market_map
        if mapping is None:
            ts = asof_ts or datetime.now(timezone.utc).isoformat()
            mapping = self.provider.get_event_market_map_asof(ts)

        panel = build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=mapping,
            kalshi_distributions=dists,
            config=config,
        )
        if options_reference is not None and len(options_reference) > 0 and len(panel) > 0:
            ref = build_options_reference_panel(options_reference, ts_col=options_ts_col)
            panel = add_options_disagreement_features(panel, ref, options_ts_col=options_ts_col)
        return panel

    def run_walkforward(
        self,
        event_features: pd.DataFrame,
        labels: pd.DataFrame,
        config: Optional[EventWalkForwardConfig] = None,
        label_col: str = "label_value",
    ) -> EventWalkForwardResult:
        """Run walkforward."""
        return run_event_walkforward(
            panel=event_features,
            labels=labels,
            config=config,
            label_col=label_col,
        )

    def evaluate_walkforward_contract(
        self,
        walkforward_result: EventWalkForwardResult,
        n_bootstrap: int = 500,
        max_events_per_day: float = 6.0,
    ) -> dict:
        """Evaluate walkforward contract."""
        return evaluate_event_contract_metrics(
            result=walkforward_result,
            n_bootstrap=n_bootstrap,
            max_events_per_day=max_events_per_day,
        )

    def evaluate_event_promotion(
        self,
        walkforward_result: EventWalkForwardResult,
        promotion_config: Optional[EventPromotionConfig] = None,
        extra_contract_metrics: Optional[dict] = None,
    ) -> PromotionDecision:
        """Evaluate event promotion."""
        return evaluate_event_promotion(
            walkforward_result=walkforward_result,
            config=promotion_config,
            extra_contract_metrics=extra_contract_metrics,
        )
