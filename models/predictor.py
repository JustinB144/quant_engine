"""
Model Predictor — loads trained ensemble and generates predictions.

Blends regime-specific model with global model based on regime confidence.

Anti-leakage measures:
    - Median imputation (matching trainer, not fillna(0))
    - Prediction clipping to ±3*target_std (prevents unrealistic outliers)
    - Uses joblib for safe deserialization (matching trainer)
    - Feature causality enforcement (blocks RESEARCH_ONLY features at runtime)
"""
import hashlib
import logging
import json
from pathlib import Path
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd

from ..config import MODEL_DIR, REGIME_NAMES, REGIME_SUPPRESS_ID, TRUTH_LAYER_ENFORCE_CAUSALITY, PREDICTION_MODE, VERIFY_MODEL_CHECKSUMS
from ..features.pipeline import get_feature_type
from .conformal import ConformalPredictor
from .governance import ModelGovernance
from .versioning import ModelRegistry

logger = logging.getLogger(__name__)


def _compute_checksum(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _prepare_features(
    raw: pd.DataFrame,
    expected: list,
    medians: dict,
) -> pd.DataFrame:
    """Align, impute, and return features matching expected column order."""
    available = [c for c in expected if c in raw.columns]
    if not available:
        raise ValueError("No expected features found in input DataFrame")
    X = raw[available].copy()
    for col in expected:
        if col not in X.columns:
            X[col] = medians.get(col, 0.0)
    X = X[expected]
    for col in expected:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
    X = X.fillna(X.median())
    return X


class EnsemblePredictor:
    """
    Loads a trained regime-conditional ensemble and generates predictions.

    Supports versioned model loading:
        - version="latest" (default): loads from registry's latest version
        - version="20260220_143022": loads specific version
        - Falls back to flat MODEL_DIR if registry doesn't exist (backward compat)

    Prediction formula:
        pred = alpha * regime_model(X) + (1 - alpha) * global_model(X)

    where alpha = regime_confidence * regime_model_reliability
    """

    def __init__(
        self,
        horizon: int = 10,
        model_dir: Optional[Path] = None,
        version: str = "latest",
    ):
        """Initialize EnsemblePredictor."""
        self.horizon = horizon
        self.model_dir = model_dir or MODEL_DIR
        self.version = version
        self._load()

    def _resolve_model_dir(self) -> Path:
        """Resolve the actual directory containing model files."""
        if self.version == "champion":
            governance = ModelGovernance()
            champion_id = governance.get_champion_version(self.horizon)
            if champion_id is None:
                raise FileNotFoundError(
                    f"No champion model configured for horizon={self.horizon}",
                )
            champion_dir = self.model_dir / champion_id
            if champion_dir.exists():
                return champion_dir
            raise FileNotFoundError(
                f"Champion model directory missing for version '{champion_id}'",
            )

        registry = ModelRegistry(model_dir=self.model_dir)

        if registry.has_versions():
            if self.version == "latest":
                version_dir = registry.get_latest_dir()
                if version_dir and version_dir.exists():
                    return version_dir
                raise FileNotFoundError(
                    f"Latest model version is registered but missing on disk: {registry.latest_version_id}",
                )
            else:
                version_dir = registry.get_version_dir(self.version)
                if version_dir.exists():
                    return version_dir
                raise FileNotFoundError(
                    f"Requested model version '{self.version}' not found in {self.model_dir}",
                )

        if self.version != "latest":
            # Explicit version requests must never silently fall back.
            raise FileNotFoundError(
                f"No version registry is available in {self.model_dir}; cannot resolve explicit version '{self.version}'",
            )

        # Fallback: flat MODEL_DIR (backward compat when version='latest')
        return self.model_dir

    def _load(self):
        """Load all model artifacts from disk.

        SECURITY NOTE: joblib.load() is equivalent to pickle.load() and can
        execute arbitrary code during deserialization. Model artifacts MUST
        only be sourced from trusted, operator-controlled directories.
        See: https://joblib.readthedocs.io/en/latest/persistence.html
        """
        resolved_dir = self._resolve_model_dir()
        prefix = resolved_dir / f"ensemble_{self.horizon}d"

        # Metadata (load first so checksums are available for verification)
        with open(f"{prefix}_meta.json", "r") as f:
            self.meta = json.load(f)

        # Checksum verification (defense-in-depth against artifact tampering)
        artifact_checksums = self.meta.get("artifact_checksums", {})

        def _verify_and_load(path: Path):
            """Verify checksum (if available) then load via joblib."""
            if VERIFY_MODEL_CHECKSUMS and artifact_checksums:
                expected = artifact_checksums.get(path.name)
                if expected:
                    actual = _compute_checksum(path)
                    if actual != expected:
                        raise ValueError(
                            f"Checksum mismatch for {path.name}: "
                            f"expected {expected}, got {actual}. "
                            f"Model artifact may be corrupted or tampered with."
                        )
            return joblib.load(str(path))

        # Global model
        self.global_model = _verify_and_load(Path(f"{prefix}_global_model.pkl"))
        self.global_scaler = _verify_and_load(Path(f"{prefix}_global_scaler.pkl"))
        self.global_features = self.meta["global_features"]

        # Median imputation values (from training data)
        self.global_medians = self.meta.get("global_feature_medians", {})
        # Target std for prediction clipping
        self.global_target_std = self.meta.get("global_target_std", 0.10)

        # Confidence calibrator (optional — trained on holdout predictions vs outcomes)
        self.calibrator = None
        calibrator_path = resolved_dir / f"ensemble_{self.horizon}d_calibrator.pkl"
        if calibrator_path.exists():
            try:
                self.calibrator = _verify_and_load(calibrator_path)
            except (OSError, ValueError):
                self.calibrator = None

        # Conformal predictor (optional — provides prediction intervals)
        self.conformal: Optional[ConformalPredictor] = None
        conformal_path = resolved_dir / f"ensemble_{self.horizon}d_conformal.json"
        if conformal_path.exists():
            try:
                with open(conformal_path, "r") as f:
                    conformal_data = json.load(f)
                self.conformal = ConformalPredictor.from_dict(conformal_data)
            except (OSError, ValueError, json.JSONDecodeError):
                self.conformal = None

        # Regime models
        self.regime_models = {}
        self.regime_scalers = {}
        self.regime_features = {}
        self.regime_reliability = {}
        self.regime_medians = {}
        self.regime_target_stds = {}

        for code_str, info in self.meta.get("regime_models", {}).items():
            code = int(code_str)
            try:
                self.regime_models[code] = _verify_and_load(
                    Path(f"{prefix}_regime{code}_model.pkl")
                )
                self.regime_scalers[code] = _verify_and_load(
                    Path(f"{prefix}_regime{code}_scaler.pkl")
                )
                self.regime_features[code] = info["features"]
                # Reliability = holdout Spearman correlation (clamped to 0-1)
                self.regime_reliability[code] = max(0, min(1, info.get("holdout_corr", 0)))
                # Median imputation values for this regime's features
                self.regime_medians[code] = info.get("feature_medians", {})
                self.regime_target_stds[code] = info.get("target_std", 0.10)
            except FileNotFoundError:
                continue

    def predict(
        self,
        features: pd.DataFrame,
        regimes: pd.Series,
        regime_confidence: pd.Series,
        regime_probabilities: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for all rows.

        Args:
            features: full feature matrix
            regimes: regime labels (0-3)
            regime_confidence: confidence in regime classification (0-1)
            regime_probabilities: optional posterior probabilities with columns
                                 regime_prob_0..regime_prob_3

        Returns:
            DataFrame with columns:
                predicted_return: blended prediction (clipped)
                global_prediction: global model prediction
                regime_prediction: regime-specific prediction (NaN if no regime model)
                confidence: overall prediction confidence
                regime: regime label
                blend_alpha: weight given to regime model
        """
        # ── Feature causality enforcement ──
        # Block RESEARCH_ONLY features from reaching live predictions.
        # RESEARCH_ONLY features may use future or cross-sectional data.
        if TRUTH_LAYER_ENFORCE_CAUSALITY:
            research_only = {
                col for col in features.columns
                if get_feature_type(col) == "RESEARCH_ONLY"
            }
            if research_only:
                raise ValueError(
                    f"RESEARCH_ONLY features in prediction: {sorted(research_only)}. "
                    f"Set TRUTH_LAYER_ENFORCE_CAUSALITY=False to override."
                )

            # Warn about END_OF_DAY features (safe for daily, unsafe for intraday)
            end_of_day = {
                col for col in features.columns
                if get_feature_type(col) == "END_OF_DAY"
            }
            if end_of_day:
                logger.info(
                    "Prediction includes %d END_OF_DAY features: %s. "
                    "These require same-day close data and are only valid for "
                    "end-of-day predictions.",
                    len(end_of_day), sorted(end_of_day)[:5],
                )

            if PREDICTION_MODE == "intraday" and end_of_day:
                raise ValueError(
                    f"END_OF_DAY features in intraday prediction: {sorted(end_of_day)}. "
                    f"Remove these features or switch to daily mode."
                )

        n = len(features)
        result = pd.DataFrame(index=features.index)

        # ── Global prediction with median imputation ──
        X_global = _prepare_features(features, self.global_features, self.global_medians)

        X_global_scaled = self.global_scaler.transform(X_global)
        global_pred = self.global_model.predict(X_global_scaled)

        # Clip global predictions to ±3 * target_std
        clip_bound = 3 * self.global_target_std
        global_pred = np.clip(global_pred, -clip_bound, clip_bound)
        result["global_prediction"] = global_pred

        # ── Regime-specific predictions ──
        regime_pred = np.full(n, np.nan)
        blend_alpha = np.zeros(n)
        # Collect individual member predictions for disagreement monitoring (SPEC-H02)
        member_preds: dict[str, np.ndarray] = {"global": global_pred.copy()}

        if regime_probabilities is not None:
            # Mixture-of-experts blend across all regime models.
            rp = regime_probabilities.reindex(features.index).fillna(0.0)
            weighted_sum = np.zeros(n)
            weight_total = np.zeros(n)

            for code, model in self.regime_models.items():
                feats = self.regime_features[code]
                try:
                    X_regime = _prepare_features(features, feats, self.regime_medians.get(code, {}))
                except ValueError:
                    continue

                X_regime_scaled = self.regime_scalers[code].transform(X_regime)
                pred = model.predict(X_regime_scaled)
                r_clip = 3 * self.regime_target_stds.get(code, self.global_target_std)
                pred = np.clip(pred, -r_clip, r_clip)

                # Store per-regime prediction for disagreement tracking (SPEC-H02)
                member_preds[f"regime_{code}"] = pred.copy()

                p_col = f"regime_prob_{code}"
                if p_col in rp.columns:
                    posterior = rp[p_col].values
                else:
                    posterior = (regimes == code).astype(float).values

                reliability = self.regime_reliability.get(code, 0.0)
                w = posterior * reliability
                weighted_sum += w * pred
                weight_total += w

            has_mix = weight_total > 1e-8
            regime_pred[has_mix] = weighted_sum[has_mix] / weight_total[has_mix]
            blend_alpha = np.clip(weight_total * regime_confidence.values, 0.0, 1.0)
        else:
            # Backward-compatible hard dispatch.
            for code, model in self.regime_models.items():
                mask = (regimes == code).values
                if not mask.any():
                    continue

                feats = self.regime_features[code]
                try:
                    X_regime = _prepare_features(features.loc[mask], feats, self.regime_medians.get(code, {}))
                except ValueError:
                    continue

                X_regime_scaled = self.regime_scalers[code].transform(X_regime)
                pred = model.predict(X_regime_scaled)

                r_clip = 3 * self.regime_target_stds.get(code, self.global_target_std)
                pred = np.clip(pred, -r_clip, r_clip)

                # Store per-regime prediction for disagreement tracking (SPEC-H02)
                full_pred = np.full(n, np.nan)
                full_pred[mask] = pred
                member_preds[f"regime_{code}"] = full_pred

                regime_pred[mask] = pred
                reliability = self.regime_reliability.get(code, 0)
                blend_alpha[mask] = regime_confidence[mask].values * reliability

        result["regime_prediction"] = regime_pred
        result["blend_alpha"] = blend_alpha
        result["regime"] = regimes.values
        # Pass through regime detection confidence so downstream consumers
        # (e.g. compute_shock_vectors) can use it directly.
        result["regime_confidence"] = regime_confidence.reindex(features.index).fillna(0.5).values

        # ── Ensemble disagreement (SPEC-H02) ──
        # Compute per-row std across all member predictions that are available.
        # High disagreement indicates ensemble uncertainty.
        if len(member_preds) >= 2:
            pred_matrix = np.column_stack(list(member_preds.values()))
            # Count finite values per row
            finite_mask = np.isfinite(pred_matrix)
            finite_count = finite_mask.sum(axis=1)
            # Compute std only where >= 2 members have predictions
            row_std = np.full(n, np.nan)
            valid_rows = finite_count >= 2
            if valid_rows.any():
                row_std[valid_rows] = np.nanstd(pred_matrix[valid_rows], axis=1)
            result["ensemble_disagreement"] = row_std
        else:
            result["ensemble_disagreement"] = np.full(n, 0.0)

        # ── Blended prediction ──
        # Where regime model exists: blend; otherwise: global only
        has_regime = ~np.isnan(regime_pred)
        blended = global_pred.copy()
        blended[has_regime] = (
            blend_alpha[has_regime] * regime_pred[has_regime]
            + (1 - blend_alpha[has_regime]) * global_pred[has_regime]
        )
        result["predicted_return"] = blended

        # ── Confidence score ──
        # Based on actual model performance metrics, not ad-hoc
        # Global holdout Spearman correlation as base
        global_corr = max(0, self.meta.get("global_holdout_corr", 0))
        # CV consistency: mean OOS correlation
        cv_corr = max(0, self.meta.get("global_cv_corr", 0))
        # Base confidence = geometric mean of holdout and CV correlations
        base_conf = np.sqrt(max(0, global_corr) * max(0, cv_corr)) if cv_corr > 0 else global_corr

        conf = np.full(n, base_conf)
        regime_certainty = np.ones(n)
        if regime_probabilities is not None:
            rp = regime_probabilities.reindex(features.index).fillna(0.0)
            prob_cols = [c for c in rp.columns if c.startswith("regime_prob_")]
            if prob_cols:
                regime_certainty = rp[prob_cols].max(axis=1).values

        # Boost where regime model is available.
        if regime_probabilities is not None and self.regime_models:
            rp = regime_probabilities.reindex(features.index).fillna(0.0)
            quality_num = np.zeros(n)
            quality_den = np.zeros(n)
            for code in self.regime_models:
                regime_cv = self.meta.get("regime_models", {}).get(str(code), {}).get("cv_corr", 0)
                regime_hold = self.regime_reliability.get(code, 0)
                regime_quality = np.sqrt(max(0, regime_cv) * max(0, regime_hold)) if regime_cv > 0 else regime_hold
                p_col = f"regime_prob_{code}"
                posterior = rp[p_col].values if p_col in rp.columns else (regimes == code).astype(float).values
                quality_num += posterior * regime_quality
                quality_den += posterior
            weighted_quality = np.divide(
                quality_num,
                np.maximum(quality_den, 1e-10),
                out=np.zeros_like(quality_num),
                where=quality_den > 0,
            )
            conf += blend_alpha * weighted_quality
        else:
            for code in self.regime_models:
                mask = (regimes == code).values & has_regime
                if mask.any():
                    regime_cv = self.meta.get("regime_models", {}).get(str(code), {}).get("cv_corr", 0)
                    regime_hold = self.regime_reliability.get(code, 0)
                    regime_quality = np.sqrt(max(0, regime_cv) * max(0, regime_hold)) if regime_cv > 0 else regime_hold
                    conf[mask] += blend_alpha[mask] * regime_quality

        conf = conf * (0.6 + 0.4 * regime_certainty)
        result["confidence"] = np.clip(conf, 0, 1)

        # ── Calibrate confidence if calibrator available ──
        # Calibrator was trained on composite confidence in [0, 1]
        # (see trainer.py _fit_calibrator); ensure input domain matches.
        if self.calibrator is not None:
            conf_vals = result["confidence"].values
            if not np.all((conf_vals >= 0.0) & (conf_vals <= 1.0)):
                logger.warning(
                    "Calibrator input outside [0, 1] (min=%.4f, max=%.4f); clipping",
                    float(np.min(conf_vals)), float(np.max(conf_vals)),
                )
                conf_vals = np.clip(conf_vals, 0.0, 1.0)
            result["confidence"] = self._calibrate_confidence(conf_vals)

        # ── Prediction intervals (conformal prediction) ──
        if self.conformal is not None and self.conformal.is_calibrated:
            intervals = self.conformal.predict_intervals_batch(blended)
            result["prediction_lower"] = intervals[:, 0]
            result["prediction_upper"] = intervals[:, 1]
            result["prediction_interval_width"] = intervals[:, 1] - intervals[:, 0]
            result["uncertainty_scalar"] = self.conformal.uncertainty_scalars(
                intervals[:, 1] - intervals[:, 0]
            )
        else:
            result["prediction_lower"] = np.nan
            result["prediction_upper"] = np.nan
            result["prediction_interval_width"] = np.nan
            result["uncertainty_scalar"] = 1.0

        # ── Regime suppression flag ──
        # Mark rows in the suppressed regime but preserve original confidence
        # so downstream consumers (backtest engine, live trader) can apply
        # REGIME_TRADE_POLICY with the high-confidence override intact.
        # See config.py REGIME_NAMES: {0: trending_bull, 1: trending_bear,
        #   2: mean_reverting, 3: high_volatility}
        regime_vals = regimes.reindex(features.index).fillna(-1).astype(int).values
        regime_suppress_mask = regime_vals == REGIME_SUPPRESS_ID  # high_volatility (canonical regime 3)
        result["regime_suppressed"] = regime_suppress_mask
        # DO NOT zero confidence here — let the backtest engine / live trader
        # apply REGIME_TRADE_POLICY gating with its min_confidence override.

        # Attach member predictions as DataFrame metadata for disagreement
        # tracking (SPEC-H02).  Callers can access via result.attrs.
        result.attrs["member_predictions"] = member_preds
        result.attrs["n_ensemble_members"] = len(member_preds)

        return result

    def _calibrate_confidence(self, raw_confidence: np.ndarray) -> np.ndarray:
        """Apply isotonic regression calibrator to raw confidence scores.

        The calibrator is fit during training on holdout predictions vs actual
        binary outcomes (direction correct or not). It maps raw model confidence
        to empirical probability of being correct.
        """
        if self.calibrator is None:
            return raw_confidence
        try:
            calibrated = self.calibrator.transform(raw_confidence)
            return np.clip(calibrated, 0.0, 1.0)
        except (ValueError, AttributeError):
            return raw_confidence

    @staticmethod
    def blend_multi_horizon(
        predictions: Dict[int, np.ndarray],
        regime: int,
        custom_weights: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> np.ndarray:
        """Blend predictions from multiple horizon models.

        Regime-adaptive blending weights give more weight to shorter horizons
        in high-volatility regimes (faster mean-reversion) and longer horizons
        in trending regimes (momentum persists).

        Args:
            predictions: {horizon_days: prediction_array} for each horizon.
            regime: Current regime code (0-3).
            custom_weights: Optional override {regime: {horizon: weight}}.

        Returns:
            Blended prediction array.
        """
        # Default regime-aware blending weights
        # Keys: regime code, Values: {horizon_days: weight}
        default_weights: Dict[int, Dict[int, float]] = {
            0: {5: 0.10, 10: 0.30, 20: 0.60},  # trending_bull: momentum persists
            1: {5: 0.30, 10: 0.40, 20: 0.30},  # trending_bear: balanced
            2: {5: 0.50, 10: 0.30, 20: 0.20},  # mean_reverting: quick reversals
            3: {5: 0.60, 10: 0.30, 20: 0.10},  # high_volatility: short-term focus
        }

        weights = (custom_weights or default_weights).get(
            regime, {5: 0.33, 10: 0.34, 20: 0.33}
        )

        available_horizons = sorted(set(predictions.keys()) & set(weights.keys()))
        if not available_horizons:
            # Fall back to equal weighting of whatever horizons are available
            n = len(predictions)
            if n == 0:
                return np.array([])
            return np.mean(list(predictions.values()), axis=0)

        # Normalize weights to sum to 1.0 for available horizons
        total_weight = sum(weights[h] for h in available_horizons)
        if total_weight <= 0:
            total_weight = 1.0

        blended = np.zeros_like(list(predictions.values())[0], dtype=float)
        for h in available_horizons:
            w = weights[h] / total_weight
            blended += w * predictions[h]

        return blended

    def predict_single(
        self,
        features: pd.Series,
        regime: int,
        regime_confidence: float,
        regime_probabilities: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """
        Predict for a single observation (e.g., latest bar).

        Returns dict with prediction, confidence, contributing features.
        """
        features_df = features.to_frame().T
        regimes_s = pd.Series([regime], index=features_df.index)
        conf_s = pd.Series([regime_confidence], index=features_df.index)
        probs_df = None
        if regime_probabilities:
            probs_df = pd.DataFrame(
                {
                    f"regime_prob_{k}": [float(v)]
                    for k, v in regime_probabilities.items()
                },
                index=features_df.index,
            )

        pred_df = self.predict(features_df, regimes_s, conf_s, regime_probabilities=probs_df)
        row = pred_df.iloc[0]

        # Top contributing features for explainability
        top_features = sorted(
            self.meta.get("global_feature_importance", {}).items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        return {
            "predicted_return": float(row["predicted_return"]),
            "confidence": float(row["confidence"]),
            "regime": REGIME_NAMES.get(regime, "unknown"),
            "blend_alpha": float(row["blend_alpha"]),
            "global_prediction": float(row["global_prediction"]),
            "regime_prediction": float(row["regime_prediction"]) if not np.isnan(row["regime_prediction"]) else None,
            "top_features": [(f, round(imp, 4)) for f, imp in top_features],
        }
