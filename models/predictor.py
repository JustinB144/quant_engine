"""
Model Predictor — loads trained ensemble and generates predictions.

Blends regime-specific model with global model based on regime confidence.

Anti-leakage measures:
    - Median imputation (matching trainer, not fillna(0))
    - Prediction clipping to ±3*target_std (prevents unrealistic outliers)
    - Uses joblib for safe deserialization (matching trainer)
"""
import json
from pathlib import Path
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd

from ..config import MODEL_DIR, REGIME_NAMES
from .governance import ModelGovernance
from .versioning import ModelRegistry


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
        """Load all model artifacts from disk."""
        resolved_dir = self._resolve_model_dir()
        prefix = resolved_dir / f"ensemble_{self.horizon}d"

        # Global model (joblib for safe deserialization)
        self.global_model = joblib.load(f"{prefix}_global_model.pkl")
        self.global_scaler = joblib.load(f"{prefix}_global_scaler.pkl")

        # Metadata
        with open(f"{prefix}_meta.json", "r") as f:
            self.meta = json.load(f)
        self.global_features = self.meta["global_features"]

        # Median imputation values (from training data)
        self.global_medians = self.meta.get("global_feature_medians", {})
        # Target std for prediction clipping
        self.global_target_std = self.meta.get("global_target_std", 0.10)

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
                self.regime_models[code] = joblib.load(
                    f"{prefix}_regime{code}_model.pkl"
                )
                self.regime_scalers[code] = joblib.load(
                    f"{prefix}_regime{code}_scaler.pkl"
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

                regime_pred[mask] = pred
                reliability = self.regime_reliability.get(code, 0)
                blend_alpha[mask] = regime_confidence[mask].values * reliability

        result["regime_prediction"] = regime_pred
        result["blend_alpha"] = blend_alpha
        result["regime"] = regimes.values

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

        # ── Regime 2 suppression ──
        # When regime == 2 (high_vol), suppress predictions by zeroing
        # confidence and flagging the row.  This centralizes the trade gate
        # so both backtester AND live trading honor it.
        regime_vals = regimes.reindex(features.index).fillna(-1).astype(int).values
        regime_2_mask = regime_vals == 2
        result["regime_suppressed"] = regime_2_mask
        result.loc[regime_2_mask, "confidence"] = 0.0

        return result

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
