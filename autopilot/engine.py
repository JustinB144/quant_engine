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
from ..backtest.validation import (
    walk_forward_validate,
    run_statistical_tests,
    combinatorial_purged_cv,
    superior_predictive_ability,
    strategy_signal_returns,
)
from ..models.walk_forward import _expanding_walk_forward_folds
from ..config import (
    AUTOPILOT_CYCLE_REPORT,
    AUTOPILOT_FEATURE_MODE,
    BACKTEST_ASSUMED_CAPITAL_USD,
    CPCV_PARTITIONS,
    CPCV_TEST_PARTITIONS,
    EXEC_MAX_PARTICIPATION,
    PROMOTION_STRESS_REGIMES,
    PORTFOLIO_TURNOVER_PENALTY,
    PORTFOLIO_TURNOVER_DYNAMIC,
    PORTFOLIO_TURNOVER_COST_MULTIPLIER,
    REQUIRE_PERMNO,
    SURVIVORSHIP_UNIVERSE_NAME,
    WF_MAX_TRAIN_DATES,
    SIGNAL_TOPK_QUANTILE,
    META_LABELING_ENABLED,
    META_LABELING_CONFIDENCE_THRESHOLD,
    META_LABELING_RETRAIN_FREQ_DAYS,
    META_LABELING_MIN_SAMPLES,
)
from ..data.loader import load_survivorship_universe, load_universe
from ..data.survivorship import filter_panel_by_point_in_time_universe
from ..features.pipeline import FeaturePipeline
from ..models.cross_sectional import cross_sectional_rank
from ..models.predictor import EnsemblePredictor
from ..models.trainer import ModelTrainer
from ..regime.detector import RegimeDetector
from ..regime.uncertainty_gate import UncertaintyGate
from ..risk.portfolio_optimizer import optimize_portfolio
from .meta_labeler import MetaLabelingModel
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

        # Meta-labeling model (Spec 04) — load from disk if available
        self.meta_labeler = MetaLabelingModel()
        if META_LABELING_ENABLED:
            self.meta_labeler.load()

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
        # Collect per-asset disagreement for health tracking (SPEC-H02)
        disagreement_values = []
        last_member_names: list = []
        last_n_members = 0

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

            # Capture per-asset ensemble disagreement (SPEC-H02)
            if "ensemble_disagreement" in preds.columns:
                d_vals = preds["ensemble_disagreement"].dropna()
                if len(d_vals) > 0:
                    disagreement_values.append(float(d_vals.iloc[-1]))
            member_preds = preds.attrs.get("member_predictions", {})
            if member_preds:
                last_member_names = sorted(member_preds.keys())
                last_n_members = len(member_preds)

        if not all_preds:
            raise RuntimeError("No predictions generated in autopilot cycle")

        # ── Save ensemble disagreement to health tracking (SPEC-H02) ──
        self._save_disagreement_to_health_tracking(
            disagreement_values, last_n_members, last_member_names,
        )

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

        # ── Cross-sectional top-K quantile filtering (Spec 04 T1) ────
        # Replace the fixed z-score threshold with adaptive quantile
        # selection.  At each date, only signals in the top quantile
        # (by cs_zscore) are kept; the rest are zeroed out so the
        # backtester skips them.
        n_before_topk = 0
        n_after_topk = 0
        topk_threshold = float("nan")
        if (
            "cs_zscore" in predictions.columns
            and isinstance(predictions.index, pd.MultiIndex)
        ):
            predictions = predictions.copy()
            quantile = SIGNAL_TOPK_QUANTILE
            date_level = predictions.index.get_level_values(-1)

            # Compute per-date threshold: keep top (1-quantile) fraction
            grouped_threshold = (
                predictions.groupby(date_level)["cs_zscore"]
                .transform(
                    lambda x: np.nanquantile(x.dropna(), 1 - quantile)
                    if len(x.dropna()) >= 2
                    else float("-inf")
                )
            )
            below_mask = predictions["cs_zscore"] < grouped_threshold
            n_before_topk = int((predictions["predicted_return"].abs() > 1e-8).sum())
            predictions.loc[below_mask, "predicted_return"] = 0.0
            n_after_topk = int((predictions["predicted_return"].abs() > 1e-8).sum())

            # Representative threshold for logging (median across dates)
            topk_threshold = float(grouped_threshold.median())
            self._log(
                f"    Top-K filter: quantile={quantile:.2f}, "
                f"threshold={topk_threshold:.4f}, "
                f"signals {n_before_topk} -> {n_after_topk}"
            )

        # ── Meta-labeling confidence filter (Spec 04 T3) ─────────────
        # If a trained meta-labeling model is available, predict
        # P(signal_correct) for each (permno, date) and zero-out
        # low-confidence signals.
        n_before_ml = 0
        n_after_ml = 0
        ml_confidence_stats: Optional[Dict[str, float]] = None
        if (
            META_LABELING_ENABLED
            and self.meta_labeler.is_trained
            and "predicted_return" in predictions.columns
        ):
            try:
                ml_preds, ml_stats = self._apply_meta_labeling(
                    predictions, price_data
                )
                n_before_ml = int((predictions["predicted_return"].abs() > 1e-8).sum())
                predictions = ml_preds
                n_after_ml = int((predictions["predicted_return"].abs() > 1e-8).sum())
                ml_confidence_stats = ml_stats
                self._log(
                    f"    Meta-labeling filter: "
                    f"signals {n_before_ml} -> {n_after_ml}, "
                    f"median_conf={ml_stats.get('p50', 0):.3f}"
                )
            except Exception as exc:
                # Fail-open: if meta-labeling errors, skip filtering
                logger.warning(
                    "Meta-labeling filtering failed (fail-open): %s", exc
                )

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

        # Step 1b: update regime-conditioned position sizing stats (SPEC-P01).
        # Aggregate trades from all candidate backtests and feed them to the
        # position sizer so learned win-rates replace hardcoded priors.
        self._update_regime_stats_from_backtests(bt_results)

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
            wf_fold_metrics: List[Dict] = []
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

                # Fold-level metrics for promotion gate (Spec 04 T4)
                for fold in wf.folds:
                    wf_fold_metrics.append({
                        "fold_id": fold.fold,
                        "ic": fold.test_corr,
                        "sharpe_estimate": fold.sharpe_estimate,
                        "win_rate": fold.win_rate,
                        "profit_factor": fold.profit_factor,
                        "sample_count": fold.sample_count,
                    })

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
                stress_regimes=list(PROMOTION_STRESS_REGIMES),
            )
            if cap.estimated_capacity_usd > 0:
                cap_util = float(BACKTEST_ASSUMED_CAPITAL_USD / cap.estimated_capacity_usd)
            else:
                cap_util = float(np.inf)

            # ── Statistical tests (IC, FDR, t-test) ──
            stat_tests_pass = False
            stat_tests_result = None
            if len(pred_series.dropna()) >= 50 and len(actual_series.dropna()) >= 50:
                try:
                    stat_tests_result = run_statistical_tests(
                        predictions=pred_series,
                        actuals=actual_series,
                        entry_threshold=c.entry_threshold,
                    )
                    stat_tests_pass = bool(stat_tests_result.overall_pass)
                except (ValueError, RuntimeError):
                    pass

            # ── Combinatorial Purged Cross-Validation (CPCV) ──
            cpcv_passes = False
            cpcv_result = None
            if len(pred_series.dropna()) >= 150:
                try:
                    cpcv_result = combinatorial_purged_cv(
                        predictions=pred_series,
                        actuals=actual_series,
                        n_partitions=CPCV_PARTITIONS,
                        n_test_partitions=CPCV_TEST_PARTITIONS,
                        purge_gap=c.horizon,
                    )
                    cpcv_passes = bool(cpcv_result.is_significant)
                except (ValueError, RuntimeError):
                    pass

            # ── Superior Predictive Ability (SPA) bootstrap ──
            spa_passes = False
            spa_pvalue = 1.0
            if result.daily_equity is not None and len(result.daily_equity) > 20:
                try:
                    strat_returns = result.daily_equity.pct_change().replace(
                        [np.inf, -np.inf], np.nan
                    ).fillna(0.0)
                    spa_result = superior_predictive_ability(strat_returns)
                    spa_passes = bool(spa_result.rejects_null)
                    spa_pvalue = float(spa_result.p_value)
                except (ValueError, RuntimeError):
                    pass

            contract_metrics = {
                "dsr_significant": dsr_sig,
                "dsr_p_value": dsr_p,
                "pbo": global_pbo,
                "capacity_constrained": bool(cap.capacity_constrained),
                "capacity_utilization": cap_util,
                "estimated_capacity_usd": float(cap.estimated_capacity_usd),
                "wf_oos_corr": wf_oos,
                "wf_positive_fold_fraction": wf_pos_frac,
                "wf_is_oos_gap": wf_gap,
                "regime_positive_fraction": regime_positive_fraction,
                "stat_tests_pass": stat_tests_pass,
                "cpcv_passes": cpcv_passes,
                "spa_passes": spa_passes,
                "spa_pvalue": spa_pvalue,
            }
            # Stress-regime capacity (SPEC-V03)
            if cap.stress_capacity_usd is not None:
                contract_metrics["stress_capacity_usd"] = float(cap.stress_capacity_usd)
                contract_metrics["stress_market_impact_bps"] = float(cap.stress_market_impact_bps)
            # Fold-level metrics for fold consistency scoring (Spec 04 T4/T5)
            if wf_fold_metrics:
                contract_metrics["fold_metrics"] = wf_fold_metrics
            if stat_tests_result is not None:
                contract_metrics["ic_mean"] = float(stat_tests_result.ic_mean)
                contract_metrics["ic_ir"] = float(stat_tests_result.ic_ir)
            if cpcv_result is not None:
                contract_metrics["cpcv_mean_corr"] = float(cpcv_result.mean_test_corr)
            decisions.append(self.gate.evaluate(c, result, contract_metrics=contract_metrics))

        return decisions

    # ── Regime stats update (SPEC-P01) ────────────────────────────────

    def _update_regime_stats_from_backtests(
        self,
        bt_results: List[Tuple],
    ) -> None:
        """Aggregate trades from backtest results and update regime statistics.

        Collects trades from all candidate backtests, deduplicates by
        (ticker, entry_date, exit_date) to avoid inflating counts from
        overlapping candidates, then feeds them to a PositionSizer to
        compute learned regime-conditioned win rates.  The updated stats
        are persisted to disk for use by subsequent cycles.
        """
        from ..risk.position_sizer import PositionSizer

        all_trades = []
        for _candidate, result in bt_results:
            if not hasattr(result, "trades") or not result.trades:
                continue
            for t in result.trades:
                all_trades.append({
                    "ticker": getattr(t, "ticker", ""),
                    "entry_date": getattr(t, "entry_date", ""),
                    "exit_date": getattr(t, "exit_date", ""),
                    "net_return": getattr(t, "net_return", 0.0),
                    "regime": getattr(t, "regime", 0),
                })

        if not all_trades:
            return

        trades_df = pd.DataFrame(all_trades)

        # Deduplicate: same (ticker, entry_date, exit_date) across candidates
        trades_df = trades_df.drop_duplicates(
            subset=["ticker", "entry_date", "exit_date"],
            keep="first",
        )

        if trades_df.empty:
            return

        sizer = PositionSizer()
        sizer.update_regime_stats(trades_df, regime_col="regime", persist=True)
        self._log(
            f"    SPEC-P01: Updated regime stats from {len(trades_df)} "
            f"deduplicated backtest trades"
        )

    # ── Meta-labeling helpers (Spec 04 T2/T3/T6) ────────────────────

    def _apply_meta_labeling(
        self,
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply meta-labeling confidence filter to predictions.

        For each permno, builds meta-features from signals and market
        returns, predicts P(signal_correct), and zeros out low-confidence
        signals.

        Args:
            predictions: Prediction panel with MultiIndex (permno, date).
            price_data: OHLCV data keyed by permno.

        Returns:
            Tuple of (filtered predictions DataFrame, confidence stats dict).
        """
        preds = predictions.copy()
        all_confidences: List[float] = []
        threshold = META_LABELING_CONFIDENCE_THRESHOLD

        permnos = preds.index.get_level_values(0).unique()
        for permno in permnos:
            if permno not in preds.index.get_level_values(0):
                continue

            permno_preds = preds.loc[permno]
            signals = permno_preds["predicted_return"]

            # Build returns from price data
            pdf = price_data.get(permno) if price_data else None
            if pdf is not None and "Close" in pdf.columns:
                close = pd.to_numeric(pdf["Close"], errors="coerce")
                returns = close.pct_change().fillna(0.0)
            elif pdf is not None and "total_ret" in pdf.columns:
                returns = pd.to_numeric(
                    pdf["total_ret"], errors="coerce"
                ).fillna(0.0)
            else:
                # No price data — skip meta-labeling for this permno
                continue

            # Regime states
            regimes = permno_preds.get("regime", pd.Series(0, index=permno_preds.index))

            # Align returns to signals index
            returns_aligned = returns.reindex(signals.index).fillna(0.0)

            try:
                meta_feats = MetaLabelingModel.build_meta_features(
                    signals=signals,
                    returns=returns_aligned,
                    regime_states=regimes,
                )
                confidence = self.meta_labeler.predict_confidence(meta_feats)
                all_confidences.extend(confidence.tolist())

                # Zero out low-confidence signals
                low_conf_mask = confidence < threshold
                if low_conf_mask.any():
                    # Map back to MultiIndex
                    low_conf_dates = permno_preds.index[low_conf_mask.values]
                    for dt in low_conf_dates:
                        preds.loc[(permno, dt), "predicted_return"] = 0.0
            except (ValueError, RuntimeError, KeyError):
                continue

        # Confidence distribution stats
        stats: Dict[str, float] = {}
        if all_confidences:
            conf_arr = np.array(all_confidences, dtype=float)
            stats = {
                "min": float(np.min(conf_arr)),
                "p10": float(np.percentile(conf_arr, 10)),
                "p50": float(np.percentile(conf_arr, 50)),
                "p90": float(np.percentile(conf_arr, 90)),
                "max": float(np.max(conf_arr)),
            }

        return preds, stats

    def _retrain_meta_labeler(
        self,
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """Retrain the meta-labeling model on recent walk-forward data.

        Collects signals, actuals, and regime states from the predictions
        panel, builds training data, and retrains the meta-labeling model.
        Saves the updated model to disk.

        Args:
            predictions: Prediction panel with MultiIndex (permno, date).
            price_data: OHLCV data keyed by permno.

        Returns:
            True if retraining succeeded, False otherwise.
        """
        self._log("  Retraining meta-labeling model...")

        all_signals = []
        all_returns = []
        all_regimes = []
        all_actuals = []

        permnos = predictions.index.get_level_values(0).unique()
        for permno in permnos:
            if permno not in predictions.index.get_level_values(0):
                continue

            permno_preds = predictions.loc[permno]
            signals = permno_preds["predicted_return"]

            # Build returns
            pdf = price_data.get(permno) if price_data else None
            if pdf is not None and "Close" in pdf.columns:
                close = pd.to_numeric(pdf["Close"], errors="coerce")
                returns = close.pct_change().fillna(0.0)
                fwd = close.shift(-self.horizon) / close - 1.0
            elif pdf is not None and "total_ret" in pdf.columns:
                returns = pd.to_numeric(
                    pdf["total_ret"], errors="coerce"
                ).fillna(0.0)
                gross = (1 + returns).shift(-1)
                fwd = (
                    gross.rolling(window=self.horizon, min_periods=self.horizon)
                    .apply(np.prod, raw=True)
                    .shift(-(self.horizon - 1))
                    - 1.0
                )
            else:
                continue

            regimes = permno_preds.get(
                "regime", pd.Series(0, index=permno_preds.index)
            )

            # Align everything to the signals index
            returns_aligned = returns.reindex(signals.index).fillna(0.0)
            fwd_aligned = fwd.reindex(signals.index)

            # Prefix permno to avoid index collisions
            prefix = f"{permno}_"
            all_signals.append(signals.rename(lambda x: f"{prefix}{x}"))
            all_returns.append(returns_aligned.rename(lambda x: f"{prefix}{x}"))
            all_regimes.append(regimes.rename(lambda x: f"{prefix}{x}"))
            all_actuals.append(fwd_aligned.rename(lambda x: f"{prefix}{x}"))

        if not all_signals:
            self._log("  Meta-labeler retrain: no data available.")
            return False

        combined_signals = pd.concat(all_signals)
        combined_returns = pd.concat(all_returns)
        combined_regimes = pd.concat(all_regimes)
        combined_actuals = pd.concat(all_actuals)

        # Drop NaN actuals
        valid = combined_actuals.notna()
        combined_signals = combined_signals[valid]
        combined_returns = combined_returns[valid]
        combined_regimes = combined_regimes[valid]
        combined_actuals = combined_actuals[valid]

        if len(combined_signals) < META_LABELING_MIN_SAMPLES:
            self._log(
                f"  Meta-labeler retrain: insufficient samples "
                f"({len(combined_signals)} < {META_LABELING_MIN_SAMPLES})."
            )
            return False

        try:
            meta_feats = MetaLabelingModel.build_meta_features(
                signals=combined_signals,
                returns=combined_returns,
                regime_states=combined_regimes,
            )
            labels = MetaLabelingModel.build_labels(
                signals=combined_signals,
                actuals=combined_actuals,
            )
            metrics = self.meta_labeler.train(meta_feats, labels)
            self.meta_labeler.save()
            self._log(
                f"  Meta-labeler retrained: "
                f"n={metrics.get('n_samples', 0):.0f}, "
                f"train_acc={metrics.get('train_accuracy', 0):.3f}, "
                f"val_acc={metrics.get('val_accuracy', 0):.3f}"
            )
            return True
        except (RuntimeError, ValueError, ImportError) as exc:
            logger.warning("Meta-labeler retraining failed: %s", exc)
            return False

    def _compute_uncertainty_inputs(
        self,
        latest_predictions: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute signal_uncertainty, regime_entropy, and drift_score.

        These inputs feed the risk governor's uncertainty-aware sizing
        (Spec 05 T2/T8).  Each metric is in [0, 1]:

        - signal_uncertainty: rolling std of predicted returns, normalized.
          High uncertainty = model predictions are volatile.
        - regime_entropy: normalized Shannon entropy of regime probabilities.
          High entropy = regime is ambiguous.
        - drift_score: absolute price momentum (20-bar), normalized to [0, 1].
          High drift = strong trend = high confidence in direction.
        """
        df = latest_predictions.copy()

        # --- signal_uncertainty: from prediction volatility ---
        if "predicted_return" in df.columns:
            preds = df["predicted_return"].astype(float)
            pred_std = preds.std()
            if pred_std > 0 and np.isfinite(pred_std):
                # Normalize deviation from mean as uncertainty
                deviation = (preds - preds.mean()).abs() / pred_std
                # Map to [0, 1]: low deviation = low uncertainty
                df["signal_uncertainty"] = (1.0 - deviation.clip(0, 3) / 3.0).clip(0, 1)
                # Invert: high deviation from mean = high uncertainty about signal direction
                df["signal_uncertainty"] = 1.0 - df["signal_uncertainty"]
            else:
                df["signal_uncertainty"] = 0.5
        else:
            df["signal_uncertainty"] = 0.5

        # --- regime_entropy: from regime probability columns ---
        regime_prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
        if len(regime_prob_cols) >= 2:
            probs = df[regime_prob_cols].astype(float).clip(lower=1e-10)
            # Normalize rows to sum to 1
            row_sums = probs.sum(axis=1)
            probs = probs.div(row_sums, axis=0)
            # Shannon entropy normalized by log(n_regimes)
            entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(len(regime_prob_cols))
            df["regime_entropy"] = entropy.clip(0, 1).fillna(0.5)
        elif "regime_confidence" in df.columns:
            # If we have regime_confidence, map to entropy (inverse)
            conf = df["regime_confidence"].astype(float).clip(0, 1).fillna(0.5)
            df["regime_entropy"] = 1.0 - conf
        else:
            df["regime_entropy"] = 0.5

        # --- drift_score: price momentum from market data ---
        drift_scores = []
        id_col = "permno" if "permno" in df.columns else "ticker"
        for _, row in df.iterrows():
            permno = str(row.get(id_col, ""))
            price_df = data.get(permno)
            if price_df is not None and "Close" in price_df.columns and len(price_df) >= 20:
                close = price_df["Close"].astype(float)
                ret_20d = close.pct_change(20).iloc[-1]
                if np.isfinite(ret_20d):
                    # Normalize absolute momentum to [0, 1] using a sigmoid-like mapping
                    # abs(20-day return) of 10% -> drift_score ~ 0.73
                    drift = float(1.0 - np.exp(-10.0 * abs(ret_20d)))
                    drift_scores.append(max(0.0, min(1.0, drift)))
                else:
                    drift_scores.append(0.5)
            else:
                drift_scores.append(0.5)

        df["drift_score"] = drift_scores

        logger.debug(
            "Uncertainty inputs: signal_unc=[%.2f, %.2f], regime_ent=[%.2f, %.2f], "
            "drift=[%.2f, %.2f]",
            df["signal_uncertainty"].min(), df["signal_uncertainty"].max(),
            df["regime_entropy"].min(), df["regime_entropy"].max(),
            df["drift_score"].min(), df["drift_score"].max(),
        )

        return df

    # ── SPEC-P04 helpers ─────────────────────────────────────────────

    def _get_current_portfolio_weights(self) -> Optional[pd.Series]:
        """Extract current portfolio weights from the paper trader.

        Loads the paper trader's persisted state and converts active
        positions into a weight series (notional value / total equity)
        so the optimizer can minimise turnover relative to the existing
        allocation.  Returns ``None`` when there are no open positions
        (i.e. building a portfolio from scratch).
        """
        try:
            state = self.paper_trader._load_state()
            positions = state.get("positions", [])
            if not positions:
                return None

            # Total equity = cash + sum(position values)
            cash = float(state.get("cash", 0.0))
            position_values: Dict[str, float] = {}
            total_position_value = 0.0
            for pos in positions:
                permno = str(pos.get("permno", pos.get("ticker", "")))
                shares = float(pos.get("shares", 0))
                price = float(pos.get("last_price", pos.get("entry_price", 0)))
                if shares != 0 and price > 0:
                    val = shares * price
                    position_values[permno] = val
                    total_position_value += abs(val)

            equity = cash + total_position_value
            if equity <= 0:
                return None

            weights = {k: v / equity for k, v in position_values.items()}
            if not weights:
                return None
            return pd.Series(weights, dtype=float)
        except (AttributeError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _dynamic_turnover_penalty(
        self,
        returns_df: pd.DataFrame,
        assets: list,
    ) -> float:
        """Compute cost-aware turnover penalty.

        When ``PORTFOLIO_TURNOVER_DYNAMIC`` is enabled, this estimates the
        average round-trip execution cost across the universe and sets the
        penalty floor to ``PORTFOLIO_TURNOVER_COST_MULTIPLIER × avg_cost``
        (in decimal, not basis points).  This ensures the optimizer never
        proposes a rebalance whose penalty is below real trading friction.

        Falls back to the static ``PORTFOLIO_TURNOVER_PENALTY`` when cost
        estimation is unavailable.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Historical return panel (columns are asset identifiers).
        assets : list
            Asset identifiers in the optimization universe.

        Returns
        -------
        float
            The effective turnover penalty (decimal per unit turnover).
        """
        base_penalty = PORTFOLIO_TURNOVER_PENALTY
        try:
            from ..backtest.execution import ExecutionModel

            exec_model = ExecutionModel()

            cost_estimates_bps: list = []
            for asset in assets:
                if asset not in returns_df.columns:
                    continue
                ret_series = returns_df[asset].dropna()
                if len(ret_series) < 20:
                    continue

                # Annualised volatility from recent returns
                realized_vol = float(ret_series.tail(60).std() * np.sqrt(252))
                if not np.isfinite(realized_vol) or realized_vol <= 0:
                    realized_vol = 0.20  # conservative fallback

                # Use representative notional and volume for cost estimation.
                # These are order-of-magnitude estimates; the execution model's
                # relative cost scaling is what matters, not absolute levels.
                est_cost = exec_model.estimate_cost(
                    daily_volume=500_000.0,
                    desired_notional=100_000.0,
                    realized_vol=realized_vol,
                )
                if np.isfinite(est_cost) and est_cost > 0:
                    cost_estimates_bps.append(est_cost)

            if cost_estimates_bps:
                avg_cost_bps = float(np.median(cost_estimates_bps))
                # Convert from bps to decimal (10000 bps = 1.0)
                avg_cost_decimal = avg_cost_bps / 10_000.0
                dynamic_floor = avg_cost_decimal * PORTFOLIO_TURNOVER_COST_MULTIPLIER
                return max(base_penalty, dynamic_floor)

        except (ImportError, TypeError, ValueError) as e:
            logger.debug("Dynamic turnover penalty estimation failed (%s); using static", e)

        return base_penalty

    def _compute_optimizer_weights(
        self,
        latest_predictions: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
        predictions_hist: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """Compute confidence-weighted portfolio optimizer weights.

        Uses the predicted returns from latest predictions as expected returns
        and regime-conditional covariance for risk budgeting.  When regime
        data is available (via ``predictions_hist``), the optimizer uses
        the covariance matrix estimated for the current market regime rather
        than a full-sample estimate.  This causes the optimizer to reflect
        the higher correlations typical of stress regimes, producing more
        diversified portfolios when it matters most.

        After optimisation, each asset's weight is scaled by its calibrated
        confidence score so that high-confidence predictions receive full
        exposure while low-confidence predictions are reduced.

        Parameters
        ----------
        latest_predictions : pd.DataFrame
            Cross-sectional predictions for the most recent date with columns
            including ``permno``, ``predicted_return`` (or ``cs_zscore``),
            ``regime``, and optionally ``confidence``.
        data : dict[str, pd.DataFrame]
            Historical OHLCV data keyed by permno.
        predictions_hist : pd.DataFrame, optional
            Full panel of historical predictions (MultiIndex: permno × date)
            with a ``regime`` column.  Used to construct a date-level regime
            time-series for regime-conditional covariance estimation.

        Returns
        -------
        pd.Series or None
            Optimal portfolio weights indexed by permno, or ``None`` if
            optimisation is not feasible (too few assets, missing data, etc.).
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

        # ── SPEC-W04: Regime-conditional covariance ──
        # When regime data is available, use regime-specific covariance so the
        # optimizer reflects the higher correlations typical of stress regimes.
        # Falls back to generic Ledoit-Wolf when regime data is missing or
        # insufficient.
        from ..risk.covariance import (
            CovarianceEstimator,
            compute_regime_covariance,
            get_regime_covariance,
        )

        # Determine the current regime from latest cross-sectional predictions.
        current_regime: Optional[int] = None
        if "regime" in latest_predictions.columns:
            regime_vals = latest_predictions["regime"].dropna()
            if len(regime_vals) > 0:
                current_regime = int(regime_vals.mode().iloc[0])

        # Build a date-level regime time-series from prediction history.
        regime_series: Optional[pd.Series] = None
        if predictions_hist is not None and "regime" in predictions_hist.columns:
            try:
                if predictions_hist.index.nlevels > 1:
                    # MultiIndex (permno, date) — aggregate per date
                    regime_by_date = (
                        predictions_hist
                        .groupby(level=1)["regime"]
                        .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0)
                    )
                else:
                    regime_by_date = predictions_hist["regime"]
                regime_series = regime_by_date.astype(int)
            except (KeyError, IndexError, ValueError):
                regime_series = None

        cov_matrix: Optional[pd.DataFrame] = None
        if current_regime is not None and regime_series is not None and len(regime_series) >= 30:
            try:
                regime_covs = compute_regime_covariance(returns_df, regime_series)
                cov_matrix = get_regime_covariance(regime_covs, current_regime)
                logger.debug(
                    "Regime-conditional covariance: regime=%d, %d regime obs",
                    current_regime,
                    len(regime_series),
                )
            except (ValueError, KeyError) as e:
                logger.debug(
                    "Regime covariance failed (%s), falling back to generic estimator",
                    e,
                )
                cov_matrix = None

        if cov_matrix is None:
            try:
                estimator = CovarianceEstimator()
                cov_estimate = estimator.estimate(returns_df)
                cov_matrix = cov_estimate.covariance
            except (ValueError, RuntimeError):
                cov_matrix = returns_df.cov()

        # Align assets between expected returns and covariance
        common = sorted(set(expected_returns.index) & set(cov_matrix.columns))
        if len(common) < 2:
            return None

        # ── SPEC-P04: Configurable, cost-aware turnover penalty ──
        # Start with the static config penalty, then optionally raise it to
        # a floor of (PORTFOLIO_TURNOVER_COST_MULTIPLIER × estimated cost)
        # so the optimizer never proposes trades whose penalty is below the
        # real round-trip friction.
        effective_turnover_penalty = PORTFOLIO_TURNOVER_PENALTY
        if PORTFOLIO_TURNOVER_DYNAMIC:
            effective_turnover_penalty = self._dynamic_turnover_penalty(
                returns_df, common,
            )
            self._log(
                f"  Turnover penalty: {effective_turnover_penalty:.6f} "
                f"(base={PORTFOLIO_TURNOVER_PENALTY:.6f}, dynamic={PORTFOLIO_TURNOVER_DYNAMIC})"
            )

        # Retrieve current portfolio weights from the paper trader so the
        # optimizer can penalise deviations from the existing allocation.
        current_weights = self._get_current_portfolio_weights()

        try:
            weights = optimize_portfolio(
                expected_returns=expected_returns[common],
                covariance=cov_matrix.loc[common, common],
                turnover_penalty=effective_turnover_penalty,
                current_weights=current_weights,
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

            # ── SPEC-W03: Uncertainty gate — reduce portfolio weights when
            #    regime entropy is high (uncertain regime transitions). ──
            # Compute the current aggregate regime uncertainty from the
            # latest predictions.  Use regime_prob_* columns (Shannon entropy)
            # or fall back to regime_confidence (inverse).
            regime_prob_cols = [
                c for c in latest_predictions.columns
                if c.startswith("regime_prob_")
            ]
            if len(regime_prob_cols) >= 2:
                probs = latest_predictions[regime_prob_cols].astype(float).clip(lower=1e-10)
                row_sums = probs.sum(axis=1)
                probs = probs.div(row_sums, axis=0)
                entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(len(regime_prob_cols))
                current_uncertainty = float(entropy.clip(0, 1).mean())
            elif "regime_confidence" in latest_predictions.columns:
                conf = latest_predictions["regime_confidence"].astype(float).clip(0, 1).mean()
                current_uncertainty = float(1.0 - conf)
            else:
                current_uncertainty = 0.0

            if current_uncertainty > 0.0:
                gate = UncertaintyGate()
                weights = pd.Series(
                    gate.apply_uncertainty_gate(weights.values, current_uncertainty),
                    index=weights.index,
                )
                # Re-normalise so weights sum to 1 after gate scaling
                w_sum = weights.sum()
                if abs(w_sum) > 1e-10:
                    weights = weights / w_sum
                weights[weights.abs() < 1e-6] = 0.0
                w_sum = weights.sum()
                if abs(w_sum) > 1e-10:
                    weights = weights / w_sum
                self._log(f"  Uncertainty gate: multiplier={gate.compute_size_multiplier(current_uncertainty):.3f} "
                          f"(entropy={current_uncertainty:.3f})")

            self._log(f"  Portfolio optimizer: {(weights != 0).sum()} non-zero weights "
                      f"out of {len(common)} assets (confidence-weighted, uncertainty-gated)")
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
        optimizer_weights = self._compute_optimizer_weights(
            latest_predictions, data, predictions_hist=predictions_hist,
        )
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

        # ── Uncertainty inputs for risk governor sizing (Spec 05 T8) ──
        latest_predictions = self._compute_uncertainty_inputs(
            latest_predictions, data,
        )

        candidates = self.discovery.generate(horizon=self.horizon)
        if self.max_candidates is not None:
            candidates = candidates[: self.max_candidates]
        self._log(f"  Generated {len(candidates)} strategy candidates")

        decisions = self._evaluate_candidates(candidates, predictions_hist, data)
        ranked = PromotionGate.rank(decisions)
        promoted = self.registry.apply_promotions(ranked)
        active = self.registry.get_active()

        # ── Meta-labeling retraining schedule (Spec 04 T6) ───────────
        # Retrain the meta-labeling model on recent walk-forward data
        # when the configured interval has elapsed.
        meta_retrained = False
        if META_LABELING_ENABLED:
            from datetime import datetime, timedelta

            needs_retrain = False
            if self.meta_labeler.last_train_time is None:
                needs_retrain = True
            else:
                elapsed = datetime.now() - self.meta_labeler.last_train_time
                if elapsed >= timedelta(days=META_LABELING_RETRAIN_FREQ_DAYS):
                    needs_retrain = True

            if needs_retrain:
                meta_retrained = self._retrain_meta_labeler(
                    predictions_hist, data
                )

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
            "meta_labeling_enabled": META_LABELING_ENABLED,
            "meta_labeling_retrained": meta_retrained,
            "signal_topk_quantile": SIGNAL_TOPK_QUANTILE,
        }

        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # ── Save IC metrics to health tracking database (SPEC-H01) ──
        self._save_ic_to_health_tracking(decisions, candidates)

        self._log(f"  Cycle report: {self.report_path}")
        self._log(f"  Active strategies: {len(active)}")
        self._log(f"  Paper equity: {paper_report['equity']:.2f}")
        return report

    def _save_ic_to_health_tracking(
        self,
        decisions: list,
        candidates: list,
    ) -> None:
        """Persist IC metrics from this cycle to the health tracking database.

        Extracts the best (highest) ic_mean across all evaluated candidates
        and saves it via HealthService.save_ic_snapshot().  This enables
        the IC tracking health check (SPEC-H01) to monitor IC trends
        across autopilot cycles.
        """
        try:
            ic_values = []
            ic_ir_values = []
            best_strategy_id = ""

            for d in decisions:
                metrics = d.metrics if hasattr(d, "metrics") else {}
                ic = metrics.get("ic_mean")
                if ic is not None:
                    ic_values.append(float(ic))
                    ir = metrics.get("ic_ir")
                    if ir is not None:
                        ic_ir_values.append(float(ir))
                    # Track which strategy produced the best IC
                    if float(ic) == max(ic_values):
                        cand = d.candidate if hasattr(d, "candidate") else None
                        if cand and hasattr(cand, "strategy_id"):
                            best_strategy_id = cand.strategy_id

            if not ic_values:
                self._log("  IC tracking: no IC values from this cycle (skipping)")
                return

            best_ic = max(ic_values)
            best_ir = max(ic_ir_values) if ic_ir_values else None
            n_passed = sum(1 for d in decisions if getattr(d, "passed", False))

            from ..api.services.health_service import HealthService
            svc = HealthService()
            svc.save_ic_snapshot(
                ic_mean=best_ic,
                ic_ir=best_ir,
                n_candidates=len(candidates),
                n_passed=n_passed,
                best_strategy_id=best_strategy_id,
            )
            self._log(f"  IC tracking: saved ic_mean={best_ic:.4f} to health DB")
        except Exception as e:
            logger.warning("Failed to save IC to health tracking: %s", e)

    def _save_disagreement_to_health_tracking(
        self,
        disagreement_values: list,
        n_members: int,
        member_names: list,
    ) -> None:
        """Persist ensemble disagreement metrics to the health tracking database.

        Aggregates per-asset latest-row disagreement values and saves a
        summary snapshot via HealthService.save_disagreement_snapshot().
        This enables the ensemble disagreement health check (SPEC-H02)
        to monitor prediction consistency across autopilot cycles.
        """
        try:
            if not disagreement_values:
                self._log("  Disagreement tracking: no values from this cycle (skipping)")
                return

            import numpy as _np
            d_arr = _np.array(disagreement_values, dtype=float)
            mean_d = float(_np.mean(d_arr))
            max_d = float(_np.max(d_arr))

            # Compute fraction of assets with high disagreement
            try:
                from ..config import ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD
            except ImportError:
                ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD = 0.015
            pct_high = float((d_arr > ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD).mean())

            from ..api.services.health_service import HealthService
            svc = HealthService()
            svc.save_disagreement_snapshot(
                mean_disagreement=mean_d,
                max_disagreement=max_d,
                n_members=n_members,
                n_assets=len(disagreement_values),
                pct_high_disagreement=pct_high,
                member_names=member_names,
            )
            self._log(
                f"  Disagreement tracking: saved mean={mean_d:.4f} "
                f"(max={max_d:.4f}, {n_members} members, "
                f"{len(disagreement_values)} assets) to health DB"
            )
        except Exception as e:
            logger.warning("Failed to save disagreement to health tracking: %s", e)
