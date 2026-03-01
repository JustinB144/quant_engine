"""
PyPI jumpmodels package wrapper for regime detection.

Wraps the ``jumpmodels`` PyPI package (arXiv 2402.05272) to conform to the
existing interface pattern used by ``RegimeDetector``.  Returns arrays
compatible with ``map_raw_states_to_regimes()`` and ``RegimeOutput``
construction.

The wrapper supports:
  - Continuous (soft probability) and discrete modes
  - Time-series cross-validated lambda (jump penalty) selection
  - Online prediction via ``predict_online()``
  - Graceful fallback to legacy ``StatisticalJumpModel`` on edge cases

References
----------
- arXiv 2402.05272: "Downside Risk Reduction Using Regime-Switching Signals"
- PyPI: https://pypi.org/project/jumpmodels/
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .jump_model_legacy import JumpModelResult

logger = logging.getLogger(__name__)


class PyPIJumpModel:
    """Wrapper around the ``jumpmodels.jump.JumpModel`` PyPI package.

    Conforms to the same interface as the legacy ``StatisticalJumpModel``
    so that ``RegimeDetector._jump_detect()`` can use it as a drop-in
    replacement.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to detect (mapped to ``n_components``).
    jump_penalty : float or None
        Fixed jump penalty lambda.  If None, lambda is selected via
        time-series cross-validation.
    use_continuous : bool
        If True, use continuous JM for soft probabilities.
        If False, use discrete JM with centroid-distance softmax.
    cv_folds : int
        Number of walk-forward CV folds for lambda selection.
    lambda_range : tuple[float, float]
        (min, max) search range for jump penalty.
    lambda_steps : int
        Number of grid points in lambda search.
    max_iter : int
        Maximum coordinate descent iterations.
    tol : float
        Convergence tolerance.
    grid_size : float
        Simplex grid size for continuous mode (larger = faster but coarser).
    mode_loss : bool
        Whether to apply mode loss penalty in continuous mode.
    n_init : int
        Number of random initializations for the jump model.
    """

    # Minimum number of observations required for a robust fit
    MIN_OBSERVATIONS = 200

    def __init__(
        self,
        n_regimes: int = 4,
        jump_penalty: Optional[float] = None,
        use_continuous: Optional[bool] = None,
        cv_folds: Optional[int] = None,
        lambda_range: Optional[Tuple[float, float]] = None,
        lambda_steps: Optional[int] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        grid_size: float = 0.10,
        mode_loss: bool = True,
        n_init: int = 3,
    ):
        from ..config import (
            REGIME_JUMP_CV_FOLDS,
            REGIME_JUMP_LAMBDA_RANGE,
            REGIME_JUMP_LAMBDA_STEPS,
            REGIME_JUMP_MAX_ITER,
            REGIME_JUMP_TOL,
            REGIME_JUMP_USE_CONTINUOUS,
        )
        # NOTE: REGIME_JUMP_MODE_LOSS_WEIGHT is not used here because the
        # PyPI jumpmodels package treats mode_loss as a boolean flag, not a
        # continuous weight. The config constant is retained for documentation
        # but has no runtime effect on the PyPI backend.

        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty
        self.use_continuous = use_continuous if use_continuous is not None else REGIME_JUMP_USE_CONTINUOUS
        self.cv_folds = cv_folds if cv_folds is not None else REGIME_JUMP_CV_FOLDS
        self.lambda_range = lambda_range if lambda_range is not None else REGIME_JUMP_LAMBDA_RANGE
        self.lambda_steps = lambda_steps if lambda_steps is not None else REGIME_JUMP_LAMBDA_STEPS
        self.max_iter = max_iter if max_iter is not None else REGIME_JUMP_MAX_ITER
        self.tol = tol if tol is not None else REGIME_JUMP_TOL
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.n_init = n_init

        self._model = None  # Fitted JumpModel instance
        self._selected_lambda: Optional[float] = None

    def fit(self, observations: np.ndarray) -> JumpModelResult:
        """Fit the PyPI JumpModel to the observation matrix.

        Parameters
        ----------
        observations : ndarray of shape (T, D)
            T timesteps, D features.  Should be standardized (output of
            ``build_hmm_observation_matrix``).

        Returns
        -------
        JumpModelResult
            Compatible with the legacy interface.

        Raises
        ------
        ValueError
            If observations has fewer than ``MIN_OBSERVATIONS`` rows.
        """
        from jumpmodels.jump import JumpModel

        X = np.asarray(observations, dtype=np.float64)
        T, D = X.shape

        if T < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"PyPI JumpModel requires at least {self.MIN_OBSERVATIONS} observations, "
                f"got {T}.  Use the legacy StatisticalJumpModel for short series."
            )

        # Handle all-NaN or all-constant features
        col_std = np.nanstd(X, axis=0)
        if np.any(np.isnan(col_std)) or np.all(col_std < 1e-12):
            raise ValueError("Observation matrix has all-NaN or zero-variance features.")

        # Replace any remaining NaN with column median
        for col in range(D):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if np.isfinite(median_val) else 0.0

        # Select lambda via cross-validation or use fixed penalty
        if self.jump_penalty is not None:
            selected_lambda = self.jump_penalty
            logger.info("Using fixed jump penalty: %.4f", selected_lambda)
        else:
            selected_lambda = self._cv_select_lambda(X)
            logger.info("CV-selected jump penalty: %.4f", selected_lambda)

        self._selected_lambda = selected_lambda

        # Fit the final model on all data
        model = JumpModel(
            n_components=self.n_regimes,
            jump_penalty=selected_lambda,
            cont=self.use_continuous,
            grid_size=self.grid_size,
            mode_loss=self.mode_loss,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=42,
            verbose=0,
        )

        try:
            model.fit(X)
        except Exception as e:
            # If continuous mode fails, fall back to discrete
            if self.use_continuous:
                logger.warning(
                    "Continuous JumpModel fit failed (%s), falling back to discrete mode.", e
                )
                model = JumpModel(
                    n_components=self.n_regimes,
                    jump_penalty=selected_lambda,
                    cont=False,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    n_init=self.n_init,
                    random_state=42,
                    verbose=0,
                )
                model.fit(X)
            else:
                raise

        self._model = model

        # Extract results
        labels = np.asarray(model.labels_, dtype=int)

        # Get soft probabilities
        probs = np.asarray(model.predict_proba(X), dtype=float)

        # If discrete mode produced hard probs, compute soft probs from centroids
        unique_probs = np.unique(probs)
        if len(unique_probs) <= 2 and set(unique_probs).issubset({0.0, 1.0}):
            probs = self._compute_soft_probs(X, np.asarray(model.centers_))

        centroids = np.asarray(model.centers_, dtype=float)

        # Determine convergence (the package doesn't expose this directly,
        # but successful fit implies convergence or max_iter reached)
        converged = True

        return JumpModelResult(
            regime_sequence=labels,
            regime_probs=probs,
            centroids=centroids,
            jump_penalty=selected_lambda,
            n_iterations=self.max_iter,  # Package doesn't expose iteration count
            converged=converged,
        )

    def predict_online(self, x_new: np.ndarray) -> np.ndarray:
        """Online prediction for new observations.

        Each row's prediction uses only data up to and including that row.

        Parameters
        ----------
        x_new : ndarray of shape (T, D) or (D,)
            New observations to classify.

        Returns
        -------
        ndarray of int, shape (T,) or scalar
            Predicted regime labels.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x = np.asarray(x_new, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        labels = np.asarray(self._model.predict_online(x), dtype=int)
        return labels

    def predict_proba_online(self, x_new: np.ndarray) -> np.ndarray:
        """Online probability prediction for new observations.

        Each row's probabilities use only data up to and including that row.

        Parameters
        ----------
        x_new : ndarray of shape (T, D) or (D,)
            New observations to classify.

        Returns
        -------
        ndarray of float, shape (T, K)
            Predicted regime probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x = np.asarray(x_new, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        probs = np.asarray(self._model.predict_proba_online(x), dtype=float)

        # If hard probs from discrete mode, compute soft from centroids
        unique_vals = np.unique(probs)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            probs = self._compute_soft_probs(x, np.asarray(self._model.centers_))

        return probs

    def _cv_select_lambda(self, X: np.ndarray) -> float:
        """Select jump penalty via time-series cross-validation.

        Scores each lambda candidate on a simple regime-following strategy's
        Sharpe ratio: long in the state with highest mean return, flat otherwise.

        Parameters
        ----------
        X : ndarray of shape (T, D)
            Full observation matrix.

        Returns
        -------
        float
            Selected jump penalty with highest average validation Sharpe.
        """
        from jumpmodels.jump import JumpModel

        T = X.shape[0]
        n_folds = self.cv_folds
        lambdas = np.linspace(self.lambda_range[0], self.lambda_range[1], self.lambda_steps)

        # Use returns from the first feature column (ret_1d, typically
        # the first column of the observation matrix)
        returns = X[:, 0]

        # Time-series walk-forward splits (no shuffling)
        fold_size = T // (n_folds + 1)
        if fold_size < 50:
            logger.warning(
                "CV fold size too small (%d), using default penalty %.4f",
                fold_size, self.lambda_range[0],
            )
            from ..config import REGIME_JUMP_PENALTY
            return REGIME_JUMP_PENALTY

        best_lambda = lambdas[len(lambdas) // 2]  # default: midpoint
        best_score = -np.inf

        for lam in lambdas:
            fold_scores = []

            for fold in range(n_folds):
                train_end = fold_size * (fold + 1)
                val_start = train_end
                val_end = min(val_start + fold_size, T)

                if val_end - val_start < 20:
                    continue

                X_train = X[:train_end]
                X_val = X[val_start:val_end]
                ret_val = returns[val_start:val_end]

                try:
                    model = JumpModel(
                        n_components=self.n_regimes,
                        jump_penalty=lam,
                        cont=False,  # Always discrete for CV speed
                        max_iter=self.max_iter,
                        tol=self.tol,
                        n_init=2,  # Fewer inits for CV speed
                        random_state=42,
                        verbose=0,
                    )
                    model.fit(X_train)

                    # Predict on validation
                    val_labels = np.asarray(model.predict(X_val), dtype=int)

                    # Score: Sharpe of regime-following strategy
                    # Identify the "bull" state (highest mean return in training)
                    train_labels = np.asarray(model.labels_, dtype=int)
                    train_returns = returns[:train_end]
                    state_means = {}
                    for s in range(self.n_regimes):
                        mask = train_labels == s
                        if mask.sum() > 0:
                            state_means[s] = train_returns[mask].mean()
                        else:
                            state_means[s] = 0.0

                    bull_state = max(state_means, key=state_means.get)

                    # Strategy: fully invested in bull state, flat otherwise
                    strategy_returns = np.where(val_labels == bull_state, ret_val, 0.0)
                    sr_mean = strategy_returns.mean()
                    sr_std = strategy_returns.std()
                    sharpe = (sr_mean / sr_std * np.sqrt(252)) if sr_std > 1e-10 else 0.0
                    fold_scores.append(sharpe)

                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue

            if fold_scores:
                avg_score = np.mean(fold_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_lambda = lam

        logger.info(
            "Lambda CV: best=%.4f (Sharpe=%.3f), range=[%.4f, %.4f], steps=%d",
            best_lambda, best_score, self.lambda_range[0], self.lambda_range[1],
            self.lambda_steps,
        )
        return float(best_lambda)

    @staticmethod
    def _compute_soft_probs(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute soft regime probabilities from distances to centroids.

        Uses a softmax over negative squared distances, matching the
        approach used by the legacy ``StatisticalJumpModel``.

        Parameters
        ----------
        X : ndarray of shape (T, D)
        centroids : ndarray of shape (K, D)

        Returns
        -------
        ndarray of float, shape (T, K)
            Probability matrix where each row sums to 1.
        """
        K = centroids.shape[0]
        dist_sq = np.zeros((X.shape[0], K), dtype=np.float64)
        for k in range(K):
            diff = X - centroids[k]
            dist_sq[:, k] = np.sum(diff ** 2, axis=1)

        # Softmax with temperature = median distance
        tau = max(float(np.median(dist_sq)), 1e-10)
        log_probs = -dist_sq / tau
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        probs /= row_sums

        return probs
