"""
End-to-end autopilot cycle:
1) ensure/train baseline model
2) discover candidate execution variants
3) evaluate candidates on strict-OOS history
4) promote passers to active registry
5) run automatic paper-trading cycle
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..backtest.engine import Backtester
from ..backtest.advanced_validation import (
    capacity_analysis,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from ..backtest.validation import walk_forward_validate
from ..models.walk_forward import _expanding_walk_forward_folds
from ..config import (
    AUTOPILOT_CYCLE_REPORT,
    AUTOPILOT_FEATURE_MODE,
    BACKTEST_ASSUMED_CAPITAL_USD,
    EXEC_MAX_PARTICIPATION,
    REQUIRE_PERMNO,
    SURVIVORSHIP_UNIVERSE_NAME,
    WF_MAX_TRAIN_DATES,
)
from ..data.loader import load_survivorship_universe, load_universe
from ..data.survivorship import filter_panel_by_point_in_time_universe
from ..features.pipeline import FeaturePipeline
from ..models.cross_sectional import cross_sectional_rank
from ..models.predictor import EnsemblePredictor
from ..models.trainer import ModelTrainer
from ..regime.detector import RegimeDetector
from ..risk.covariance import compute_regime_covariance, get_regime_covariance
from ..risk.portfolio_optimizer import optimize_portfolio
from .paper_trader import PaperTrader
from .promotion_gate import PromotionDecision, PromotionGate
from .registry import StrategyRegistry
from .strategy_discovery import StrategyCandidate, StrategyDiscovery

_PERMNO_RE = re.compile(r"^\d{1,10}$")


class HeuristicPredictor:
    """
    Lightweight fallback predictor used when sklearn-backed model artifacts
    are unavailable in the runtime.
    """

    def __init__(self, horizon: int):
        """Initialize HeuristicPredictor."""
        self.horizon = horizon
        self.meta: Dict[str, object] = {}

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
        """Internal helper for rolling zscore."""
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        return (series - mean) / (std + 1e-10)

    def predict(
        self,
        features: pd.DataFrame,
        regimes: pd.Series,
        regime_confidence: pd.Series,
        regime_probabilities: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate predictions from the provided inputs."""
        score = pd.Series(0.0, index=features.index, dtype=float)
        norm = 0.0

        factor_spec = [
            ("return_20d", 0.35),
            ("return_5d", 0.20),
            ("PriceVsSMA_50", 0.20),
            ("MACD_12_26", 0.15),
            ("ZScore_20", -0.10),
        ]
        for col, weight in factor_spec:
            if col in features.columns:
                z = self._rolling_zscore(pd.to_numeric(features[col], errors="coerce")).fillna(0.0)
                score += weight * z
                norm += abs(weight)

        if norm > 0:
            score = score / norm
        else:
            score = pd.Series(0.0, index=features.index, dtype=float)

        predicted_return = np.tanh(score.to_numpy(dtype=float)) * 0.03
        result = pd.DataFrame(index=features.index)
        result["global_prediction"] = predicted_return
        result["regime_prediction"] = np.nan
        result["blend_alpha"] = 0.0
        result["regime"] = regimes.reindex(features.index).fillna(0).astype(int).to_numpy()
        result["predicted_return"] = predicted_return

        base_conf = 0.45 + 0.35 * np.clip(
            regime_confidence.reindex(features.index).fillna(0.0).to_numpy(),
            0.0,
            1.0,
        )
        if regime_probabilities is not None:
            prob_cols = [c for c in regime_probabilities.columns if str(c).startswith("regime_prob_")]
            if prob_cols:
                certainty = (
                    regime_probabilities[prob_cols]
                    .reindex(features.index)
                    .fillna(0.0)
                    .max(axis=1)
                    .to_numpy()
                )
                base_conf = base_conf * (0.7 + 0.3 * np.clip(certainty, 0.0, 1.0))
        result["confidence"] = np.clip(base_conf, 0.0, 1.0)
        return result


class AutopilotEngine:
    """Coordinates discovery, promotion, and paper execution."""

    def __init__(
        self,
        tickers: List[str],
        horizon: int = 10,
        years: int = 15,
        feature_mode: str = AUTOPILOT_FEATURE_MODE,
        model_version: str = "champion",
        max_candidates: Optional[int] = None,
        strict_oos: bool = True,
        survivorship_mode: bool = True,
        walk_forward: bool = True,
        verbose: bool = True,
        report_path: Path = AUTOPILOT_CYCLE_REPORT,
    ):
        """Initialize AutopilotEngine."""
        self.tickers = tickers
        self.horizon = horizon
        self.years = years
        self.feature_mode = feature_mode
        self.model_version = model_version
        self.max_candidates = max_candidates
        self.strict_oos = strict_oos
        self.survivorship_mode = survivorship_mode
        self.walk_forward = walk_forward
        self.verbose = verbose
        self.report_path = Path(report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.discovery = StrategyDiscovery()
        self.gate = PromotionGate()
        self.registry = StrategyRegistry()
        self.paper_trader = PaperTrader()

    def _log(self, msg: str):
        """Emit a log message when verbose logging is enabled."""
        if self.verbose:
            logger.info(msg)

    @staticmethod
    def _is_permno_key(value: object) -> bool:
        """Return whether permno key satisfies the expected condition."""
        return _PERMNO_RE.match(str(value).strip()) is not None

    def _assert_permno_price_data(self, data: Dict[str, pd.DataFrame], context: str) -> None:
        """Validate permno price data and raise on contract violations."""
        if not REQUIRE_PERMNO:
            return
        bad = sorted({str(k) for k in data.keys() if not self._is_permno_key(k)})
        if bad:
            preview = ", ".join(bad[:5])
            raise RuntimeError(f"{context}: non-PERMNO price_data keys detected: {preview}")

    def _assert_permno_prediction_panel(self, panel: pd.DataFrame, context: str) -> None:
        """Validate permno prediction panel and raise on contract violations."""
        if not REQUIRE_PERMNO:
            return
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
            raise RuntimeError(f"{context}: prediction history must be MultiIndex (permno, date).")
        bad = sorted({str(x) for x in panel.index.get_level_values(0) if not self._is_permno_key(x)})
        if bad:
            preview = ", ".join(bad[:5])
            raise RuntimeError(f"{context}: non-PERMNO prediction keys detected: {preview}")

    def _assert_permno_latest_predictions(self, latest: pd.DataFrame, context: str) -> None:
        """Validate permno latest predictions and raise on contract violations."""
        if not REQUIRE_PERMNO:
            return
        if "permno" not in latest.columns:
            raise RuntimeError(f"{context}: latest predictions missing 'permno' column.")
        bad = sorted({str(x) for x in latest["permno"].tolist() if not self._is_permno_key(x)})
        if bad:
            preview = ", ".join(bad[:5])
            raise RuntimeError(f"{context}: non-PERMNO latest prediction ids detected: {preview}")

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Internal helper to load data."""
        if self.survivorship_mode:
            self._log("  Loading survivorship-bias-free PIT universe data...")
            data = load_survivorship_universe(years=self.years, verbose=self.verbose)
        else:
            self._log(f"  Loading universe data for {len(self.tickers)} tickers...")
            data = load_universe(self.tickers, years=self.years, verbose=self.verbose)
        if not data:
            raise RuntimeError("No data loaded for autopilot cycle")
        self._assert_permno_price_data(data, context="autopilot load")
        return data

    def _build_regimes(
        self,
        features: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Internal helper to build regimes."""
        detector = RegimeDetector()
        regime_parts = []
        regime_prob_parts = []
        for permno in data:
            permno_feats = features.loc[permno]
            regime_df = detector.regime_features(permno_feats)
            regime_df["permno"] = permno
            regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
            regime_parts.append(regime_df["regime"])
            prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
            if prob_cols:
                regime_prob_parts.append(regime_df[prob_cols])
        regimes = pd.concat(regime_parts)
        regime_probs = pd.concat(regime_prob_parts) if regime_prob_parts else None
        return regimes, regime_probs

    def _train_baseline(self, data: Dict[str, pd.DataFrame]):
        """
        Train on the earliest 80% of dates so the most recent 20% remains OOS.
        """
        self._log("  Training baseline model for strict-OOS autopilot evaluation...")
        pipeline = FeaturePipeline(
            include_interactions=self.feature_mode != "core",
            include_research_factors=self.feature_mode != "core",
            include_cross_asset_factors=self.feature_mode != "core",
            verbose=self.verbose,
        )
        features, targets = pipeline.compute_universe(data, verbose=self.verbose)
        regimes, regime_probs = self._build_regimes(features, data)
        target_col = f"target_{self.horizon}d"
        if target_col not in targets.columns:
            raise RuntimeError(f"Target column missing: {target_col}")

        dates = pd.to_datetime(features.index.get_level_values(1))
        unique_dates = sorted(pd.Index(dates.unique()))
        if len(unique_dates) < 10:
            raise RuntimeError("Insufficient history for strict-OOS baseline training")
        cutoff = unique_dates[int(len(unique_dates) * 0.8) - 1]
        train_mask = dates <= cutoff

        trainer = ModelTrainer()
        trainer.train_ensemble(
            features=features[train_mask],
            targets=targets[target_col][train_mask],
            regimes=regimes[train_mask],
            regime_probabilities=regime_probs[train_mask] if regime_probs is not None else None,
            horizon=self.horizon,
            verbose=self.verbose,
            versioned=True,
        )

    def _ensure_predictor(self, data: Dict[str, pd.DataFrame]):
        """Internal helper for ensure predictor."""
        try:
            return EnsemblePredictor(horizon=self.horizon, version=self.model_version)
        except FileNotFoundError:
            if self.model_version == "champion":
                try:
                    self._log("  Champion model not available; falling back to latest version.")
                    return EnsemblePredictor(horizon=self.horizon, version="latest")
                except (FileNotFoundError, ImportError, RuntimeError):
                    pass
            try:
                self._train_baseline(data)
                return EnsemblePredictor(horizon=self.horizon, version="latest")
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
                self._log(
                    f"  Baseline model unavailable ({type(e).__name__}: {e}); "
                    "using deterministic heuristic predictor.",
                )
                return HeuristicPredictor(horizon=self.horizon)
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            self._log(
                f"  Predictor load failed ({type(e).__name__}: {e}); "
                "using deterministic heuristic predictor.",
            )
            return HeuristicPredictor(horizon=self.horizon)

    def _predict_universe(
        self,
        data: Dict[str, pd.DataFrame],
        predictor,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Internal helper for predict universe."""
        pipeline = FeaturePipeline(
            include_interactions=self.feature_mode != "core",
            include_research_factors=self.feature_mode != "core",
            include_cross_asset_factors=self.feature_mode != "core",
            verbose=False,
        )
        universe_features, _ = pipeline.compute_universe(
            data=data,
            verbose=False,
            compute_targets_flag=False,
        )
        available_permnos = set(universe_features.index.get_level_values(0))
        detector = RegimeDetector()
        all_preds = []
        latest_rows = []

        for permno in data:
            if permno not in available_permnos:
                continue
            feats = universe_features.loc[permno]
            regime_df = detector.regime_features(feats)
            regime_probs = regime_df[[c for c in regime_df.columns if c.startswith("regime_prob_")]]
            preds = predictor.predict(
                feats,
                regime_df["regime"],
                regime_df["regime_confidence"],
                regime_probabilities=regime_probs,
            )
            preds["permno"] = permno
            preds_idx = preds.set_index("permno", append=True).reorder_levels([1, 0])
            all_preds.append(preds_idx)

            latest = preds.iloc[-1].copy()
            latest["permno"] = permno
            latest["date"] = str(feats.index[-1].date()) if hasattr(feats.index[-1], "date") else str(feats.index[-1])
            latest_rows.append(latest)

        if not all_preds:
            raise RuntimeError("No predictions generated in autopilot cycle")

        history_preds = pd.concat(all_preds).sort_index()
        latest_preds = pd.DataFrame(latest_rows)

        if self.strict_oos:
            train_end_str = predictor.meta.get("train_data_end")
            if train_end_str:
                cutoff = pd.Timestamp(train_end_str)
                pred_dates = pd.to_datetime(history_preds.index.get_level_values(1))
                oos_mask = pred_dates > cutoff
                filtered = history_preds[oos_mask]
                if len(filtered) > 0:
                    self._log(f"  Strict OOS enabled: using {len(filtered)} rows after {cutoff.date()}")
                    history_preds = filtered
                else:
                    self._log("  Strict OOS produced no rows; retraining baseline on earlier cutoff...")
                    self._train_baseline(data)
                    predictor = EnsemblePredictor(horizon=self.horizon, version="latest")
                    return self._predict_universe(data, predictor)

        if self.survivorship_mode:
            history_preds = filter_panel_by_point_in_time_universe(
                panel=history_preds,
                universe_name=SURVIVORSHIP_UNIVERSE_NAME,
                verbose=self.verbose,
            )

        # ── Cross-sectional ranking ──
        # Rank predictions relative to peers at each date so position
        # selection uses cross-sectional z-scores instead of raw returns.
        if isinstance(history_preds.index, pd.MultiIndex):
            # Reset MultiIndex to get date as a column for ranking
            reset_preds = history_preds.reset_index()
            date_level = history_preds.index.names[-1] or "level_1"
            if date_level in reset_preds.columns and "predicted_return" in reset_preds.columns:
                ranked = cross_sectional_rank(
                    reset_preds,
                    date_col=date_level,
                    prediction_col="predicted_return",
                )
                # Restore original index structure
                idx_names = [n or f"level_{i}" for i, n in enumerate(history_preds.index.names)]
                ranked = ranked.set_index(idx_names)
                history_preds = ranked

        # Also add cs_zscore to latest_preds if enough stocks
        if len(latest_preds) >= 2 and "predicted_return" in latest_preds.columns:
            preds_vals = latest_preds["predicted_return"]
            mean_val = preds_vals.mean()
            std_val = preds_vals.std()
            if std_val > 0:
                latest_preds["cs_zscore"] = (preds_vals - mean_val) / std_val
            else:
                latest_preds["cs_zscore"] = 0.0

        return history_preds, latest_preds

    def _walk_forward_predictions(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate OOS predictions via rolling walk-forward training.

        For each expanding-window fold:
        1. Train a fresh ensemble on the training portion
        2. Generate predictions on the OOS test portion
        3. Concatenate all OOS predictions — no in-sample contamination

        Falls back to the single-split _train_baseline + _predict_universe
        flow if walk-forward produces insufficient data.
        """
        self._log("  Walk-forward prediction pipeline...")

        pipeline = FeaturePipeline(
            include_interactions=self.feature_mode != "core",
            include_research_factors=self.feature_mode != "core",
            include_cross_asset_factors=self.feature_mode != "core",
            verbose=self.verbose,
        )
        features, targets = pipeline.compute_universe(
            data, verbose=self.verbose, compute_targets_flag=True,
        )
        regimes, regime_probs = self._build_regimes(features, data)

        target_col = f"target_{self.horizon}d"
        if target_col not in targets.columns:
            raise RuntimeError(f"Target column missing: {target_col}")

        # Build date series and expanding-window folds
        dates = pd.to_datetime(features.index.get_level_values(-1))
        date_series = pd.Series(dates, index=features.index)
        n_folds = 5
        folds = _expanding_walk_forward_folds(date_series, n_folds=n_folds, horizon=self.horizon)

        if not folds:
            self._log("  Walk-forward: insufficient history for folds, falling back to single-split.")
            return None, None

        all_oos_preds = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            self._log(f"    Fold {fold_idx + 1}/{len(folds)}: "
                      f"train={len(train_idx)} rows, test={len(test_idx)} rows")

            train_features = features.iloc[train_idx]
            train_targets = targets[target_col].iloc[train_idx]
            train_regimes = regimes.iloc[train_idx]
            train_regime_probs = regime_probs.iloc[train_idx] if regime_probs is not None else None

            test_features = features.iloc[test_idx]
            test_regimes = regimes.iloc[test_idx]

            # Train a fresh model on this fold's training window
            trainer = ModelTrainer()
            try:
                trainer.train_ensemble(
                    features=train_features,
                    targets=train_targets,
                    regimes=train_regimes,
                    regime_probabilities=train_regime_probs,
                    horizon=self.horizon,
                    verbose=False,
                    versioned=False,
                )
            except (ValueError, RuntimeError, ImportError) as e:
                self._log(f"    Fold {fold_idx + 1} training failed: {e}")
                continue

            # Load the just-trained model for OOS prediction
            try:
                fold_predictor = EnsemblePredictor(horizon=self.horizon, version="latest")
            except (FileNotFoundError, ImportError, RuntimeError) as e:
                self._log(f"    Fold {fold_idx + 1} predictor load failed: {e}")
                continue

            # Generate OOS predictions per permno in the test window
            detector = RegimeDetector()
            test_permnos = set(test_features.index.get_level_values(0))
            for permno in test_permnos:
                if permno not in test_features.index.get_level_values(0):
                    continue
                feats = test_features.loc[permno]
                regime_df = detector.regime_features(feats)
                r_probs = regime_df[[c for c in regime_df.columns if c.startswith("regime_prob_")]]
                try:
                    preds = fold_predictor.predict(
                        feats,
                        regime_df["regime"],
                        regime_df["regime_confidence"],
                        regime_probabilities=r_probs,
                    )
                    preds["permno"] = permno
                    preds_idx = preds.set_index("permno", append=True).reorder_levels([1, 0])
                    all_oos_preds.append(preds_idx)
                except (ValueError, KeyError, RuntimeError):
                    continue

        if not all_oos_preds:
            self._log("  Walk-forward produced no OOS predictions, falling back to single-split.")
            return None, None

        history_preds = pd.concat(all_oos_preds).sort_index()
        # Deduplicate: keep last prediction for each (permno, date) pair
        history_preds = history_preds[~history_preds.index.duplicated(keep="last")]

        self._log(f"  Walk-forward: {len(history_preds)} total OOS prediction rows")

        if self.survivorship_mode:
            history_preds = filter_panel_by_point_in_time_universe(
                panel=history_preds,
                universe_name=SURVIVORSHIP_UNIVERSE_NAME,
                verbose=self.verbose,
            )

        # Cross-sectional ranking
        if isinstance(history_preds.index, pd.MultiIndex):
            from ..models.cross_sectional import cross_sectional_rank
            reset_preds = history_preds.reset_index()
            date_level = history_preds.index.names[-1] or "level_1"
            if date_level in reset_preds.columns and "predicted_return" in reset_preds.columns:
                ranked = cross_sectional_rank(
                    reset_preds,
                    date_col=date_level,
                    prediction_col="predicted_return",
                )
                idx_names = [n or f"level_{i}" for i, n in enumerate(history_preds.index.names)]
                ranked = ranked.set_index(idx_names)
                history_preds = ranked

        # Build latest predictions from the most recent date per permno
        latest_rows = []
        for permno in data:
            if permno not in history_preds.index.get_level_values(0):
                continue
            permno_preds = history_preds.loc[permno]
            latest = permno_preds.iloc[-1].copy()
            latest["permno"] = permno
            latest["date"] = str(permno_preds.index[-1].date()) if hasattr(permno_preds.index[-1], "date") else str(permno_preds.index[-1])
            latest_rows.append(latest)

        if not latest_rows:
            self._log("  Walk-forward produced no latest predictions, falling back to single-split.")
            return None, None

        latest_preds = pd.DataFrame(latest_rows)

        # Add cs_zscore to latest_preds
        if len(latest_preds) >= 2 and "predicted_return" in latest_preds.columns:
            preds_vals = latest_preds["predicted_return"]
            mean_val = preds_vals.mean()
            std_val = preds_vals.std()
            if std_val > 0:
                latest_preds["cs_zscore"] = (preds_vals - mean_val) / std_val
            else:
                latest_preds["cs_zscore"] = 0.0

        return history_preds, latest_preds

    def _evaluate_candidates(
        self,
        candidates: List[StrategyCandidate],
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[PromotionDecision]:
        """Internal helper for evaluate candidates."""
        def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
            xv = np.asarray(x, dtype=float)
            yv = np.asarray(y, dtype=float)
            m = np.isfinite(xv) & np.isfinite(yv)
            if m.sum() < 3:
                return 0.0
            xr = pd.Series(xv[m]).rank(method="average").to_numpy(dtype=float)
            yr = pd.Series(yv[m]).rank(method="average").to_numpy(dtype=float)
            xr = xr - xr.mean()
            yr = yr - yr.mean()
            denom = np.sqrt(np.sum(xr**2) * np.sum(yr**2))
            if denom <= 1e-12:
                return 0.0
            return float(np.sum(xr * yr) / denom)

        # Step 1: run backtests for all candidates.
        bt_results: List[Tuple[StrategyCandidate, object]] = []
        for i, c in enumerate(candidates):
            self._log(f"    Evaluating candidate {i+1}/{len(candidates)}: {c.strategy_id}")
            backtester = Backtester(
                entry_threshold=c.entry_threshold,
                confidence_threshold=c.confidence_threshold,
                holding_days=c.horizon,
                max_positions=c.max_positions,
                position_size_pct=c.position_size_pct,
                use_risk_management=c.use_risk_management,
            )
            result = backtester.run(predictions=predictions, price_data=price_data, verbose=False)
            bt_results.append((c, result))

        # Step 2: compute global PBO over the full candidate search set.
        global_pbo = None
        daily_return_map: Dict[str, pd.Series] = {}
        for c, result in bt_results:
            if result.daily_equity is None or len(result.daily_equity) < 5:
                continue
            dr = result.daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if len(dr) > 0:
                daily_return_map[c.strategy_id] = dr
        if len(daily_return_map) >= 2:
            try:
                returns_matrix = pd.concat(daily_return_map, axis=1).fillna(0.0)
                pbo_result = probability_of_backtest_overfitting(returns_matrix=returns_matrix)
                global_pbo = float(pbo_result.pbo)
            except (ValueError, RuntimeError):
                global_pbo = None

        # Step 3: build realized forward-return panel for walk-forward checks.
        actual_parts: List[pd.Series] = []
        for permno, df in price_data.items():
            if len(df) == 0:
                continue
            ret_stream = None
            if "total_ret" in df.columns:
                ret_stream = pd.to_numeric(df["total_ret"], errors="coerce")
            elif "Return" in df.columns:
                ret_stream = pd.to_numeric(df["Return"], errors="coerce")

            if ret_stream is not None:
                gross = (1.0 + ret_stream).shift(-1)
                actual = (
                    gross.rolling(window=self.horizon, min_periods=self.horizon)
                    .apply(np.prod, raw=True)
                    .shift(-(self.horizon - 1))
                    - 1.0
                )
            else:
                if "Close" not in df.columns:
                    continue
                close = pd.to_numeric(df["Close"], errors="coerce")
                actual = close.shift(-self.horizon) / close - 1.0
            actual.name = "actual_forward"
            actual = actual.to_frame()
            actual["permno"] = str(permno)
            actual = actual.set_index("permno", append=True).reorder_levels([1, 0])
            actual_parts.append(actual["actual_forward"])
        actual_forward = pd.concat(actual_parts).sort_index() if actual_parts else pd.Series(dtype=float)

        # Step 4: strict promotion contract metrics per candidate.
        decisions: List[PromotionDecision] = []
        n_trials = max(1, len(candidates))
        for c, result in bt_results:
            pred_series = predictions["predicted_return"].copy()
            conf_series = predictions["confidence"].copy()
            regime_series = predictions["regime"].copy()
            pred_series = pred_series.where(conf_series >= c.confidence_threshold, 0.0)
            actual_series = actual_forward.reindex(pred_series.index)
            pred_series = pred_series.sort_index()
            actual_series = actual_series.sort_index()
            pred_series, actual_series = pred_series.align(actual_series, join="inner")

            wf_oos = 0.0
            wf_gap = np.inf
            wf_pos_frac = 0.0
            if len(pred_series.dropna()) >= 150:
                wf = walk_forward_validate(
                    predictions=pred_series,
                    actuals=actual_series,
                    entry_threshold=c.entry_threshold,
                    purge_gap=c.horizon,
                    embargo=max(1, c.horizon // 2),
                    max_train_samples=WF_MAX_TRAIN_DATES,
                )
                wf_oos = float(wf.avg_oos_corr)
                wf_gap = float(wf.is_oos_gap)
                if wf.n_folds > 0:
                    pos = sum(1 for f in wf.folds if f.test_corr > 0)
                    wf_pos_frac = float(pos / wf.n_folds)

            # Regime stability: fraction of regimes with positive rank correlation.
            regime_positive_fraction = 0.0
            reg = regime_series.reindex(pred_series.index)
            reg_corrs = []
            for code in sorted(pd.Series(reg).dropna().unique()):
                mask = reg == code
                if mask.sum() < 30:
                    continue
                reg_corrs.append(_safe_spearman(pred_series[mask].values, actual_series[mask].values))
            if reg_corrs:
                regime_positive_fraction = float(np.mean(np.array(reg_corrs) > 0.0))

            # Deflated Sharpe (multiple-testing aware).
            trade_returns = np.array([t.net_return for t in result.trades], dtype=float)
            dsr_sig = False
            dsr_p = 1.0
            if len(trade_returns) >= 10:
                skew = float(pd.Series(trade_returns).skew()) if len(trade_returns) > 3 else 0.0
                kurt = float(pd.Series(trade_returns).kurt()) + 3.0 if len(trade_returns) > 3 else 3.0
                ann_factor = np.sqrt(252.0 / max(1, c.horizon))
                dsr = deflated_sharpe_ratio(
                    observed_sharpe=float(result.sharpe_ratio),
                    n_trials=n_trials,
                    n_returns=len(trade_returns),
                    skewness=skew,
                    kurtosis=kurt,
                    annualization_factor=ann_factor,
                )
                dsr_sig = bool(dsr.is_significant)
                dsr_p = float(dsr.p_value)

            # Capacity realism.
            cap = capacity_analysis(
                trades=result.trades,
                price_data=price_data,
                capital_usd=BACKTEST_ASSUMED_CAPITAL_USD,
                max_participation_rate=EXEC_MAX_PARTICIPATION,
            )
            if cap.estimated_capacity_usd > 0:
                cap_util = float(BACKTEST_ASSUMED_CAPITAL_USD / cap.estimated_capacity_usd)
            else:
                cap_util = float(np.inf)

            contract_metrics = {
                "dsr_significant": dsr_sig,
                "dsr_p_value": dsr_p,
                "pbo": global_pbo,
                "capacity_constrained": bool(cap.capacity_constrained),
                "capacity_utilization": cap_util,
                "wf_oos_corr": wf_oos,
                "wf_positive_fold_fraction": wf_pos_frac,
                "wf_is_oos_gap": wf_gap,
                "regime_positive_fraction": regime_positive_fraction,
            }
            decisions.append(self.gate.evaluate(c, result, contract_metrics=contract_metrics))

        return decisions

    def _compute_optimizer_weights(
        self,
        latest_predictions: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> Optional[pd.Series]:
        """Compute confidence-weighted portfolio optimizer weights.

        Uses the predicted returns from latest predictions as expected returns
        and regime-conditional covariance for risk budgeting.  After
        optimisation, each asset's weight is scaled by its calibrated
        confidence score so that high-confidence predictions receive full
        exposure while low-confidence predictions are reduced.

        Returns None if optimisation is not feasible (too few assets, missing
        data, etc.).
        """
        if len(latest_predictions) < 2:
            return None

        # Build expected returns from cs_zscore (if available) or predicted_return
        ranking_col = "cs_zscore" if "cs_zscore" in latest_predictions.columns else "predicted_return"
        if "permno" not in latest_predictions.columns:
            return None

        expected_returns = pd.Series(
            latest_predictions[ranking_col].values,
            index=latest_predictions["permno"].values,
            dtype=float,
        )
        expected_returns = expected_returns.dropna()
        if len(expected_returns) < 2:
            return None

        # Build return history for covariance estimation
        return_parts = []
        for permno, df in data.items():
            if str(permno) not in expected_returns.index:
                continue
            if "Close" in df.columns:
                close = pd.to_numeric(df["Close"], errors="coerce")
                ret = close.pct_change().dropna()
                ret.name = str(permno)
                return_parts.append(ret)
            elif "total_ret" in df.columns:
                ret = pd.to_numeric(df["total_ret"], errors="coerce").dropna()
                ret.name = str(permno)
                return_parts.append(ret)

        if len(return_parts) < 2:
            return None

        returns_df = pd.concat(return_parts, axis=1).dropna(how="all")
        if returns_df.shape[0] < 30 or returns_df.shape[1] < 2:
            return None

        # Compute covariance (use full-sample for simplicity here;
        # regime-conditional covariance requires aligned regime labels)
        try:
            from ..risk.covariance import CovarianceEstimator
            estimator = CovarianceEstimator()
            cov_estimate = estimator.estimate(returns_df)
            cov_matrix = cov_estimate.covariance
        except (ImportError, ValueError, RuntimeError):
            cov_matrix = returns_df.cov()

        # Align assets between expected returns and covariance
        common = sorted(set(expected_returns.index) & set(cov_matrix.columns))
        if len(common) < 2:
            return None

        try:
            weights = optimize_portfolio(
                expected_returns=expected_returns[common],
                covariance=cov_matrix.loc[common, common],
            )

            # ── Confidence-weighted position sizing (NEW 3) ──
            # Scale each asset's optimizer weight by its calibrated
            # confidence score.  High-confidence predictions keep full
            # weight; low-confidence predictions get reduced exposure.
            # Confidence defaults to 1.0 when unavailable.
            confidence_map = {}
            if "confidence" in latest_predictions.columns and "permno" in latest_predictions.columns:
                for _, row in latest_predictions.iterrows():
                    pid = str(row["permno"])
                    conf = float(row.get("confidence", 1.0))
                    if not np.isfinite(conf) or conf < 0.0:
                        conf = 0.0
                    confidence_map[pid] = min(conf, 1.0)

            if confidence_map:
                conf_series = pd.Series(
                    {asset: confidence_map.get(asset, 1.0) for asset in weights.index},
                    dtype=float,
                )
                weights = weights * conf_series
                # Re-normalise so weights sum to 1 after confidence scaling
                w_sum = weights.sum()
                if abs(w_sum) > 1e-10:
                    weights = weights / w_sum
                # Zero out tiny weights for cleanliness
                weights[weights.abs() < 1e-6] = 0.0
                w_sum = weights.sum()
                if abs(w_sum) > 1e-10:
                    weights = weights / w_sum

            self._log(f"  Portfolio optimizer: {(weights != 0).sum()} non-zero weights "
                      f"out of {len(common)} assets (confidence-weighted)")
            return weights
        except (ValueError, RuntimeError) as e:
            self._log(f"  Portfolio optimizer failed ({type(e).__name__}: {e}); "
                      "using equal-weight sizing.")
            return None

    def run_cycle(self) -> Dict:
        """Run cycle."""
        self._log("\n=== AUTOPILOT CYCLE ===")
        data = self._load_data()

        # Walk-forward pipeline: rolling train windows with rolling OOS predictions.
        # Falls back to single-split when walk-forward produces insufficient data.
        predictions_hist = None
        latest_predictions = None
        if self.walk_forward:
            predictions_hist, latest_predictions = self._walk_forward_predictions(data)

        if predictions_hist is None or latest_predictions is None:
            predictor = self._ensure_predictor(data)
            predictions_hist, latest_predictions = self._predict_universe(data, predictor)
        self._assert_permno_prediction_panel(predictions_hist, context="autopilot predict history")
        self._assert_permno_latest_predictions(latest_predictions, context="autopilot latest predictions")

        # ── Portfolio optimisation on latest predictions ──
        # Weights are already confidence-scaled inside _compute_optimizer_weights.
        # We also store a per-row confidence_weight column so downstream
        # consumers (e.g. paper trader) can use it for independent sizing.
        optimizer_weights = self._compute_optimizer_weights(latest_predictions, data)
        if "confidence" in latest_predictions.columns:
            latest_predictions["confidence_weight"] = (
                latest_predictions["confidence"]
                .clip(lower=0.0, upper=1.0)
                .fillna(1.0)
            )
        else:
            latest_predictions["confidence_weight"] = 1.0

        if optimizer_weights is not None:
            latest_predictions["optimizer_weight"] = (
                latest_predictions["permno"]
                .map(optimizer_weights)
                .fillna(0.0)
            )
        else:
            latest_predictions["optimizer_weight"] = 0.0

        candidates = self.discovery.generate(horizon=self.horizon)
        if self.max_candidates is not None:
            candidates = candidates[: self.max_candidates]
        self._log(f"  Generated {len(candidates)} strategy candidates")

        decisions = self._evaluate_candidates(candidates, predictions_hist, data)
        ranked = PromotionGate.rank(decisions)
        promoted = self.registry.apply_promotions(ranked)
        active = self.registry.get_active()

        paper_report = self.paper_trader.run_cycle(
            active_strategies=active,
            latest_predictions=latest_predictions,
            price_data=data,
        )

        report = {
            "horizon": self.horizon,
            "years": self.years,
            "feature_mode": self.feature_mode,
            "strict_oos": self.strict_oos,
            "survivorship_mode": self.survivorship_mode,
            "n_candidates": len(candidates),
            "n_passed": sum(1 for d in decisions if d.passed),
            "n_promoted": len(promoted),
            "n_active": len(active),
            "top_decisions": [d.to_dict() for d in ranked[:10]],
            "paper_report": paper_report,
        }

        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self._log(f"  Cycle report: {self.report_path}")
        self._log(f"  Active strategies: {len(active)}")
        self._log(f"  Paper equity: {paper_report['equity']:.2f}")
        return report
