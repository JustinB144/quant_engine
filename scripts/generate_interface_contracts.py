#!/usr/bin/env python3
"""
JOB 4: Interface Boundary Analysis — YAML Generator
Generates docs/audit/INTERFACE_CONTRACTS.yaml from verified source analysis.
"""

import yaml
from datetime import date
from pathlib import Path


def build_contracts():
    """Build the complete interface contracts data structure."""

    boundaries = []

    # =========================================================================
    # CRITICAL BOUNDARY 1: backtest/engine.py → regime/shock_vector.py
    # =========================================================================
    boundaries.append({
        "id": "backtest_to_regime_shock_1",
        "provider_module": "regime",
        "consumer_module": "backtest",
        "symbols": [
            {
                "name": "compute_shock_vectors",
                "type": "function",
                "defined_in": "regime/shock_vector.py",
                "imported_by": "backtest/engine.py",
                "import_line": 77,
                "import_type": "top_level",
                "signature": (
                    "def compute_shock_vectors("
                    "ohlcv: pd.DataFrame, "
                    "regime_series: Optional[pd.Series] = None, "
                    "confidence_series: Optional[pd.Series] = None, "
                    "ticker: str = '', "
                    "bocpd_hazard_lambda: float = 1.0/60, "
                    "bocpd_hazard_func: str = 'constant', "
                    "bocpd_max_runlength: int = 200, "
                    "jump_sigma_threshold: float = 2.5, "
                    "vol_lookback: int = 20"
                    ") -> Dict"
                ),
                "parameters": [
                    {"name": "ohlcv", "type": "pd.DataFrame", "required": True},
                    {"name": "regime_series", "type": "Optional[pd.Series]", "required": False},
                    {"name": "confidence_series", "type": "Optional[pd.Series]", "required": False},
                    {"name": "ticker", "type": "str", "required": False},
                    {"name": "bocpd_hazard_lambda", "type": "float", "required": False},
                    {"name": "bocpd_hazard_func", "type": "str", "required": False},
                    {"name": "bocpd_max_runlength", "type": "int", "required": False},
                    {"name": "jump_sigma_threshold", "type": "float", "required": False},
                    {"name": "vol_lookback", "type": "int", "required": False},
                ],
                "return_type": "Dict[timestamp, ShockVector]",
                "stability": "stable",
                "breaking_change_impact": (
                    "Changing parameter defaults or return dict structure breaks "
                    "backtest shock-mode execution. All downstream backtests and "
                    "autopilot cycles that use shock vectors would produce different results."
                ),
            },
            {
                "name": "ShockVector",
                "type": "dataclass",
                "defined_in": "regime/shock_vector.py",
                "imported_by": "backtest/engine.py",
                "import_line": 77,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "schema_version", "type": "str"},
                    {"name": "timestamp", "type": "datetime"},
                    {"name": "ticker", "type": "str"},
                    {"name": "hmm_regime", "type": "int"},
                    {"name": "hmm_confidence", "type": "float"},
                    {"name": "hmm_uncertainty", "type": "float"},
                    {"name": "bocpd_changepoint_prob", "type": "float"},
                    {"name": "bocpd_runlength", "type": "int"},
                    {"name": "jump_detected", "type": "bool"},
                    {"name": "jump_magnitude", "type": "float"},
                    {"name": "structural_features", "type": "Dict[str, float]"},
                    {"name": "transition_matrix", "type": "Optional[np.ndarray]"},
                    {"name": "ensemble_model_type", "type": "str"},
                ],
                "key_methods": [
                    {"name": "to_dict", "signature": "def to_dict(self) -> Dict"},
                    {"name": "from_dict", "signature": "def from_dict(cls, data: Dict) -> ShockVector"},
                    {"name": "is_shock_event", "signature": "def is_shock_event(self, changepoint_threshold: float = 0.50) -> bool"},
                    {"name": "regime_name", "signature": "def regime_name(self) -> str"},
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "ShockVector is version-locked (schema_version='1.0'). Adding/removing "
                    "fields breaks serialization in backtest/execution.py and any cached "
                    "shock vector data. The to_dict/from_dict contract must remain stable."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "ShockVector schema is version-locked. Also imported by backtest/execution.py:29 "
            "(conditional import). Schema changes require version bump and migration. "
            "The structural_features dict is extensible but consumers must handle missing keys."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 2: backtest/engine.py → regime/uncertainty_gate.py
    # =========================================================================
    boundaries.append({
        "id": "backtest_to_regime_uncertainty_2",
        "provider_module": "regime",
        "consumer_module": "backtest",
        "symbols": [
            {
                "name": "UncertaintyGate",
                "type": "class",
                "defined_in": "regime/uncertainty_gate.py",
                "imported_by": "backtest/engine.py",
                "import_line": 78,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "entropy_threshold", "type": "Optional[float]"},
                    {"name": "stress_threshold", "type": "Optional[float]"},
                    {"name": "sizing_map", "type": "Optional[Dict[float, float]]"},
                    {"name": "min_multiplier", "type": "Optional[float]"},
                ],
                "key_methods": [
                    {"name": "compute_size_multiplier", "signature": "def compute_size_multiplier(self, uncertainty: float) -> float"},
                    {"name": "apply_uncertainty_gate", "signature": "def apply_uncertainty_gate(self, weights: np.ndarray, uncertainty: float) -> np.ndarray"},
                    {"name": "should_assume_stress", "signature": "def should_assume_stress(self, uncertainty: float) -> bool"},
                    {"name": "is_uncertain", "signature": "def is_uncertain(self, uncertainty: float) -> bool"},
                    {"name": "gate_series", "signature": "def gate_series(self, uncertainty_series: pd.Series) -> pd.DataFrame"},
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "UncertaintyGate is imported by 3 modules: backtest/engine.py:78, "
                    "autopilot/engine.py:61, risk/position_sizer.py:27. Threshold changes "
                    "silently alter position sizing across all three consumers. Method "
                    "signature changes break the entire risk pipeline."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "UncertaintyGate threshold defaults come from config via lazy import. Verify "
            "that config values REGIME_UNCERTAINTY_ENTROPY_THRESHOLD, "
            "REGIME_UNCERTAINTY_STRESS_THRESHOLD, REGIME_UNCERTAINTY_SIZING_MAP, and "
            "REGIME_UNCERTAINTY_MIN_MULTIPLIER are consistent across all 3 consumer modules."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 3: backtest/engine.py → risk/ (5 files)
    # =========================================================================
    boundaries.append({
        "id": "backtest_to_risk_3",
        "provider_module": "risk",
        "consumer_module": "backtest",
        "symbols": [
            {
                "name": "PositionSizer",
                "type": "class",
                "defined_in": "risk/position_sizer.py",
                "imported_by": "backtest/engine.py",
                "import_line": 316,
                "import_type": "lazy",
                "constructor_params": [
                    {"name": "target_portfolio_vol", "type": "float"},
                    {"name": "max_position_pct", "type": "float"},
                    {"name": "min_position_pct", "type": "float"},
                    {"name": "max_risk_per_trade", "type": "float"},
                    {"name": "kelly_fraction", "type": "float"},
                    {"name": "atr_multiplier", "type": "float"},
                    {"name": "blend_weights", "type": "Optional[dict]"},
                    {"name": "max_portfolio_dd", "type": "float"},
                    {"name": "bayesian_alpha", "type": "float"},
                    {"name": "bayesian_beta", "type": "float"},
                ],
                "key_methods": [
                    {
                        "name": "size_position",
                        "signature": (
                            "def size_position(self, ticker: str, win_rate: float, "
                            "avg_win: float, avg_loss: float, realized_vol: float, "
                            "atr: float, price: float, holding_days: int = 10, "
                            "confidence: float = 0.5, n_current_positions: int = 0, "
                            "max_positions: int = 20, regime: Optional[str] = None, "
                            "current_drawdown: float = 0.0, n_trades: int = 100, "
                            "signal_uncertainty: Optional[float] = None, "
                            "regime_entropy: Optional[float] = None, "
                            "drift_score: Optional[float] = None, "
                            "portfolio_equity: Optional[float] = None, "
                            "current_positions: Optional[Dict[str, float]] = None, "
                            "regime_uncertainty: float = 0.0) -> PositionSize"
                        ),
                    },
                ],
                "stability": "evolving",
                "breaking_change_impact": (
                    "PositionSizer.size_position has 21 parameters including recent "
                    "additions for uncertainty scaling. Changes to return type PositionSize "
                    "or parameter semantics affect both backtest accuracy and live paper trading."
                ),
            },
            {
                "name": "DrawdownController",
                "type": "class",
                "defined_in": "risk/drawdown.py",
                "imported_by": "backtest/engine.py",
                "import_line": 317,
                "import_type": "lazy",
                "constructor_params": [
                    {"name": "warning_threshold", "type": "float"},
                    {"name": "caution_threshold", "type": "float"},
                    {"name": "critical_threshold", "type": "float"},
                    {"name": "daily_loss_limit", "type": "float"},
                    {"name": "weekly_loss_limit", "type": "float"},
                    {"name": "recovery_days", "type": "int"},
                    {"name": "initial_equity", "type": "float"},
                ],
                "key_methods": [
                    {"name": "update", "signature": "def update(self, daily_pnl: float) -> DrawdownStatus"},
                    {"name": "reset", "signature": "def reset(self, equity: float = 1.0) -> None"},
                    {"name": "get_summary", "signature": "def get_summary(self) -> Dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "DrawdownController.update() return type DrawdownStatus drives "
                    "position sizing and trade blocking in backtester. Threshold changes "
                    "alter backtest results silently."
                ),
            },
            {
                "name": "StopLossManager",
                "type": "class",
                "defined_in": "risk/stop_loss.py",
                "imported_by": "backtest/engine.py",
                "import_line": 318,
                "import_type": "lazy",
                "constructor_params": [
                    {"name": "hard_stop_pct", "type": "float"},
                    {"name": "atr_stop_multiplier", "type": "float"},
                    {"name": "trailing_atr_multiplier", "type": "float"},
                    {"name": "trailing_activation_pct", "type": "float"},
                    {"name": "max_holding_days", "type": "int"},
                    {"name": "regime_change_exit", "type": "bool"},
                    {"name": "profit_target_pct", "type": "Optional[float]"},
                    {"name": "spread_buffer_bps", "type": "float"},
                ],
                "key_methods": [
                    {
                        "name": "evaluate",
                        "signature": (
                            "def evaluate(self, entry_price: float, current_price: float, "
                            "highest_price: float, atr: float, bars_held: int, "
                            "entry_regime: int, current_regime: int) -> StopResult"
                        ),
                    },
                    {"name": "compute_initial_stop", "signature": "def compute_initial_stop(self, entry_price: float, atr: float, regime: int = 0) -> float"},
                    {"name": "compute_risk_per_share", "signature": "def compute_risk_per_share(self, entry_price: float, atr: float, regime: int = 0) -> float"},
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "StopLossManager.evaluate() return type StopResult determines "
                    "trade exits. Changes to stop logic or regime multiplier mapping "
                    "alter all backtest trade durations and P&L."
                ),
            },
            {
                "name": "PortfolioRiskManager",
                "type": "class",
                "defined_in": "risk/portfolio_risk.py",
                "imported_by": "backtest/engine.py",
                "import_line": 319,
                "import_type": "lazy",
                "constructor_params": [
                    {"name": "max_sector_pct", "type": "float"},
                    {"name": "max_corr_between", "type": "float"},
                    {"name": "max_gross_exposure", "type": "float"},
                    {"name": "max_single_name_pct", "type": "float"},
                    {"name": "max_beta_exposure", "type": "float"},
                    {"name": "max_portfolio_vol", "type": "float"},
                    {"name": "correlation_lookback", "type": "int"},
                    {"name": "covariance_method", "type": "str"},
                    {"name": "sector_map", "type": "Optional[Dict[str, str]]"},
                    {"name": "universe_config", "type": "Optional[UniverseConfig]"},
                ],
                "key_methods": [
                    {
                        "name": "check_new_position",
                        "signature": (
                            "def check_new_position(self, ticker: str, position_size: float, "
                            "current_positions: Dict[str, float], "
                            "price_data: Dict[str, pd.DataFrame], "
                            "benchmark_data: Optional[pd.DataFrame] = None, "
                            "regime: Optional[int] = None, "
                            "regime_labels: Optional[pd.Series] = None) -> RiskCheck"
                        ),
                    },
                    {
                        "name": "compute_constraint_utilization",
                        "signature": (
                            "def compute_constraint_utilization(self, positions: Dict[str, float], "
                            "price_data: Dict[str, pd.DataFrame], "
                            "regime: Optional[int] = None) -> Dict[str, float]"
                        ),
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "PortfolioRiskManager.check_new_position() gates every trade entry. "
                    "Changing constraint thresholds or RiskCheck structure alters which "
                    "trades are allowed in backtest and paper trading."
                ),
            },
            {
                "name": "RiskMetrics",
                "type": "class",
                "defined_in": "risk/metrics.py",
                "imported_by": "backtest/engine.py",
                "import_line": 320,
                "import_type": "lazy",
                "constructor_params": [
                    {"name": "annual_trading_days", "type": "int"},
                ],
                "key_methods": [
                    {
                        "name": "compute_full_report",
                        "signature": (
                            "def compute_full_report(self, trade_returns: np.ndarray, "
                            "equity_curve: Optional[pd.Series] = None, "
                            "trade_details: Optional[List[Dict]] = None, "
                            "holding_days: int = 10) -> RiskReport"
                        ),
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "RiskMetrics.compute_full_report() generates the RiskReport attached "
                    "to every BacktestResult. Changes to RiskReport fields break "
                    "downstream evaluation and API serialization."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "All 5 risk classes are lazy-imported at lines 316-320 inside "
            "Backtester._init_risk_managers(). This means import failures only surface "
            "when use_risk_management=True. Verify all 5 classes can be imported "
            "independently. PositionSizer is evolving (21 params, recent uncertainty additions)."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 4: autopilot/engine.py → (8 modules)
    # =========================================================================
    boundaries.append({
        "id": "autopilot_to_multi_4",
        "provider_module": "backtest, models, data, features, regime, risk, api, config",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "Backtester",
                "type": "class",
                "defined_in": "backtest/engine.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 20,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Backtester is the core backtest engine. Constructor or run() changes break autopilot strategy evaluation.",
            },
            {
                "name": "capacity_analysis",
                "type": "function",
                "defined_in": "backtest/advanced_validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 21,
                "import_type": "top_level",
                "signature": (
                    "def capacity_analysis(trades: list, price_data: Dict[str, pd.DataFrame], "
                    "capital_usd: float = 1_000_000, max_participation_rate: float = 0.01, "
                    "impact_coefficient_bps: float = 30.0, stress_regimes: Optional[List[int]] = None, "
                    "min_stress_trades: int = 10) -> CapacityResult"
                ),
                "return_type": "CapacityResult",
                "stability": "stable",
                "breaking_change_impact": "CapacityResult fields feed promotion gate capacity checks.",
            },
            {
                "name": "deflated_sharpe_ratio",
                "type": "function",
                "defined_in": "backtest/advanced_validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 21,
                "import_type": "top_level",
                "signature": (
                    "def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, "
                    "n_returns: int, skewness: float = 0.0, kurtosis: float = 3.0, "
                    "annualization_factor: float = 1.0) -> DeflatedSharpeResult"
                ),
                "return_type": "DeflatedSharpeResult",
                "stability": "stable",
                "breaking_change_impact": "DSR result drives promotion gate p-value checks.",
            },
            {
                "name": "probability_of_backtest_overfitting",
                "type": "function",
                "defined_in": "backtest/advanced_validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 21,
                "import_type": "top_level",
                "signature": "def probability_of_backtest_overfitting(returns_matrix: pd.DataFrame, n_partitions: int = 8) -> PBOResult",
                "return_type": "PBOResult",
                "stability": "stable",
                "breaking_change_impact": "PBO result drives promotion gate overfitting checks.",
            },
            {
                "name": "walk_forward_validate",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "signature": (
                    "def walk_forward_validate(features: pd.DataFrame, targets: pd.Series, "
                    "regimes: pd.Series, n_folds: int = 5, horizon: int = 10, "
                    "verbose: bool = False) -> WalkForwardResult"
                ),
                "return_type": "WalkForwardResult",
                "stability": "stable",
                "breaking_change_impact": "WalkForwardResult feeds promotion gate OOS correlation checks.",
            },
            {
                "name": "run_statistical_tests",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "signature": (
                    "def run_statistical_tests(predictions: np.ndarray, actuals: np.ndarray, "
                    "long_signals: Optional[np.ndarray] = None) -> StatisticalTests"
                ),
                "return_type": "StatisticalTests",
                "stability": "stable",
                "breaking_change_impact": "StatisticalTests result drives promotion gate statistical significance checks.",
            },
            {
                "name": "combinatorial_purged_cv",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "signature": (
                    "def combinatorial_purged_cv(features: pd.DataFrame, targets: pd.Series, "
                    "n_splits: int = 5, embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]"
                ),
                "return_type": "List[Tuple[np.ndarray, np.ndarray]]",
                "stability": "stable",
                "breaking_change_impact": "CPCV fold structure feeds PBO computation.",
            },
            {
                "name": "superior_predictive_ability",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "signature": "def superior_predictive_ability(baseline_returns: np.ndarray, strategy_returns: np.ndarray) -> Dict[str, float]",
                "return_type": "Dict[str, float]",
                "stability": "stable",
                "breaking_change_impact": "SPA p-value feeds promotion gate.",
            },
            {
                "name": "strategy_signal_returns",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "signature": "def strategy_signal_returns(predictions: pd.Series, returns: pd.Series, threshold: float = 0.0) -> Dict[str, float]",
                "return_type": "Dict[str, float]",
                "stability": "stable",
                "breaking_change_impact": "Signal return computation affects strategy evaluation.",
            },
            {
                "name": "_expanding_walk_forward_folds",
                "type": "function",
                "defined_in": "models/walk_forward.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 33,
                "import_type": "top_level",
                "signature": "def _expanding_walk_forward_folds(dates: pd.Series, n_folds: int, horizon: int) -> List[tuple]",
                "return_type": "List[tuple]",
                "stability": "stable",
                "breaking_change_impact": "Fold generation change alters all autopilot training/validation splits.",
            },
            {
                "name": "load_survivorship_universe",
                "type": "function",
                "defined_in": "data/loader.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 54,
                "import_type": "top_level",
                "signature": "def load_survivorship_universe(...) -> Tuple[Dict[str, pd.DataFrame], Dict]",
                "return_type": "Tuple[Dict[str, pd.DataFrame], Dict]",
                "stability": "stable",
                "breaking_change_impact": "Changes to universe loading break autopilot data pipeline.",
            },
            {
                "name": "load_universe",
                "type": "function",
                "defined_in": "data/loader.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 54,
                "import_type": "top_level",
                "signature": "def load_universe(tickers: List[str], years: int = 15, ...) -> Dict[str, pd.DataFrame]",
                "return_type": "Dict[str, pd.DataFrame]",
                "stability": "stable",
                "breaking_change_impact": "Return dict structure change breaks all downstream feature/model code.",
            },
            {
                "name": "filter_panel_by_point_in_time_universe",
                "type": "function",
                "defined_in": "data/survivorship.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 55,
                "import_type": "top_level",
                "signature": (
                    "def filter_panel_by_point_in_time_universe(panel: pd.DataFrame, "
                    "pit_universe: Dict[str, pd.DataFrame], date_col: str = 'date', "
                    "ticker_col: str = 'ticker') -> pd.DataFrame"
                ),
                "return_type": "pd.DataFrame",
                "stability": "stable",
                "breaking_change_impact": "Survivorship filtering change alters universe composition in backtests.",
            },
            {
                "name": "FeaturePipeline",
                "type": "class",
                "defined_in": "features/pipeline.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 56,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Feature column changes break model training/prediction alignment.",
            },
            {
                "name": "cross_sectional_rank",
                "type": "function",
                "defined_in": "models/cross_sectional.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 57,
                "import_type": "top_level",
                "signature": (
                    "def cross_sectional_rank(predictions: pd.DataFrame, "
                    "date_col: str = 'date', prediction_col: str = 'predicted_return', "
                    "asset_col: Optional[str] = None, long_quantile: float = 0.80, "
                    "short_quantile: float = 0.20) -> pd.DataFrame"
                ),
                "return_type": "pd.DataFrame",
                "stability": "stable",
                "breaking_change_impact": "Quantile threshold changes alter signal selection.",
            },
            {
                "name": "EnsemblePredictor",
                "type": "class",
                "defined_in": "models/predictor.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 58,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Predictor API change breaks all prediction consumers.",
            },
            {
                "name": "ModelTrainer",
                "type": "class",
                "defined_in": "models/trainer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 59,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Trainer interface change breaks autopilot training loop.",
            },
            {
                "name": "RegimeDetector",
                "type": "class",
                "defined_in": "regime/detector.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 60,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Regime label changes propagate through entire pipeline.",
            },
            {
                "name": "UncertaintyGate",
                "type": "class",
                "defined_in": "regime/uncertainty_gate.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 61,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Threshold changes silently alter position sizing.",
            },
            {
                "name": "optimize_portfolio",
                "type": "function",
                "defined_in": "risk/portfolio_optimizer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 62,
                "import_type": "top_level",
                "signature": (
                    "def optimize_portfolio(expected_returns: pd.Series, "
                    "covariance: pd.DataFrame, current_weights: Optional[pd.Series] = None, "
                    "max_position: float = 0.10, max_portfolio_vol: float = 0.30, "
                    "turnover_penalty: float = 0.001, risk_aversion: float = 1.0, "
                    "sector_map: Optional[Dict[str, str]] = None, "
                    "max_sector_exposure: Optional[float] = None) -> pd.Series"
                ),
                "return_type": "pd.Series",
                "stability": "stable",
                "breaking_change_impact": "Portfolio weight computation change alters all allocation decisions.",
            },
            {
                "name": "HealthService",
                "type": "class",
                "defined_in": "api/services/health_service.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1868,
                "import_type": "lazy",
                "stability": "evolving",
                "breaking_change_impact": "Circular dependency (autopilot→api). Method changes break IC/disagreement tracking.",
            },
            {
                "name": "PositionSizer",
                "type": "class",
                "defined_in": "risk/position_sizer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 971,
                "import_type": "lazy",
                "stability": "evolving",
                "breaking_change_impact": "Position sizing changes affect all autopilot trade decisions.",
            },
            {
                "name": "ExecutionModel",
                "type": "class",
                "defined_in": "backtest/execution.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1405,
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "Execution model changes alter fill simulation in autopilot backtests.",
            },
            {
                "name": "CovarianceEstimator",
                "type": "class",
                "defined_in": "risk/covariance.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1528,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "Covariance estimate changes alter portfolio optimization inputs.",
            },
            {
                "name": "compute_regime_covariance",
                "type": "function",
                "defined_in": "risk/covariance.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1528,
                "import_type": "lazy",
                "signature": (
                    "def compute_regime_covariance(returns: pd.DataFrame, regimes: pd.Series, "
                    "min_obs: int = 30, shrinkage: float = 0.1) -> Dict[int, pd.DataFrame]"
                ),
                "return_type": "Dict[int, pd.DataFrame]",
                "stability": "stable",
                "breaking_change_impact": "Regime covariance computation affects portfolio optimization.",
            },
            {
                "name": "get_regime_covariance",
                "type": "function",
                "defined_in": "risk/covariance.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1528,
                "import_type": "lazy",
                "signature": "def get_regime_covariance(regime_covs: Dict[int, pd.DataFrame], current_regime: int) -> pd.DataFrame",
                "return_type": "pd.DataFrame",
                "stability": "stable",
                "breaking_change_impact": "Regime covariance lookup affects portfolio optimization.",
            },
        ],
        "shared_artifacts": [
            {
                "path": "trained_models/ensemble_*d_*.pkl",
                "format": "joblib",
                "schema_owner": "models",
                "writer": "models/trainer.py",
                "readers": ["models/predictor.py", "api/services/model_service.py"],
                "schema_fields": ["global_model", "global_scaler", "regime_models", "calibrator"],
                "breaking_change_impact": "Model binary format change breaks all prediction paths.",
            },
            {
                "path": "trained_models/ensemble_*d_meta.json",
                "format": "json",
                "schema_owner": "models",
                "writer": "models/trainer.py",
                "readers": ["models/predictor.py", "api/services/model_service.py"],
                "schema_fields": [
                    "horizon", "train_data_start", "train_data_end",
                    "global_features", "global_cv_corr", "global_holdout_r2",
                    "global_holdout_corr", "global_target_std", "global_feature_medians",
                    "global_warnings", "global_feature_importance",
                    "regime_models", "train_time_seconds", "total_samples",
                    "structural_weights_applied",
                ],
                "breaking_change_impact": (
                    "Meta JSON schema change breaks EnsemblePredictor._load(). "
                    "Required fields: global_features, global_feature_medians, "
                    "global_target_std, regime_models."
                ),
            },
            {
                "path": "results/autopilot/strategy_registry.json",
                "format": "json",
                "schema_owner": "autopilot",
                "writer": "autopilot/registry.py",
                "readers": ["autopilot/registry.py", "api/services/autopilot_service.py"],
                "schema_fields": ["active", "history"],
                "breaking_change_impact": (
                    "Registry schema change breaks strategy loading. "
                    "Active list and history structure must remain stable."
                ),
            },
        ],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "CRITICAL — autopilot/engine.py imports from 8 distinct modules with 23+ symbols. "
            "It is the primary transitive amplifier for 12/15 hotspot blast radii. "
            "Any change in any of the 8 provider modules can break autopilot. "
            "Lines 1868 and 1911 create circular autopilot→api dependency. "
            "Config imports at line 34 pull 18+ constants. "
            "Verify all lazy imports (lines 565, 971, 1405, 1528, 1868, 1911) can resolve at runtime."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 5: autopilot/paper_trader.py → api/services/
    # =========================================================================
    boundaries.append({
        "id": "autopilot_to_api_circular_5",
        "provider_module": "api",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "create_health_risk_gate",
                "type": "function",
                "defined_in": "api/services/health_risk_feedback.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 173,
                "import_type": "conditional",
                "signature": "def create_health_risk_gate() -> HealthRiskGate",
                "return_type": "HealthRiskGate",
                "stability": "stable",
                "breaking_change_impact": "Factory function change breaks paper trader health-to-risk feedback loop.",
            },
            {
                "name": "HealthService",
                "type": "class",
                "defined_in": "api/services/health_service.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 189,
                "import_type": "conditional",
                "key_methods": [
                    {"name": "get_health_history", "signature": "def get_health_history(self, limit: int = 30) -> List[Dict[str, Any]]"},
                    {
                        "name": "save_execution_quality_fill",
                        "signature": (
                            "def save_execution_quality_fill(self, symbol: str, side: str, "
                            "predicted_cost_bps: float, actual_cost_bps: float, "
                            "fill_ratio: Optional[float] = None, "
                            "participation_rate: Optional[float] = None, "
                            "regime: Optional[int] = None) -> None"
                        ),
                    },
                ],
                "stability": "evolving",
                "breaking_change_impact": (
                    "HealthService is imported at 3 locations in paper_trader.py (lines 189, 532) "
                    "and 2 in engine.py (lines 1868, 1911). This is a circular dependency: "
                    "api serves autopilot, but autopilot imports from api. Method signature "
                    "changes break IC tracking, disagreement monitoring, and execution quality logging."
                ),
            },
            {
                "name": "ABTestRegistry",
                "type": "class",
                "defined_in": "api/ab_testing.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 211,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "storage_path", "type": "Optional[Path]"},
                ],
                "key_methods": [
                    {"name": "get_active_test", "signature": "def get_active_test(self) -> Optional[ABTest]"},
                    {
                        "name": "create_test",
                        "signature": (
                            "def create_test(self, name: str, description: str, "
                            "control_name: str, treatment_name: str, "
                            "control_overrides: Optional[Dict[str, Any]] = None, "
                            "treatment_overrides: Optional[Dict[str, Any]] = None, "
                            "allocation: float = 0.5, min_trades: int = 50) -> ABTest"
                        ),
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": "ABTestRegistry API change breaks paper trader A/B test integration.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "CIRCULAR DEPENDENCY: autopilot→api at 4 import sites (paper_trader.py:173,189,211,532). "
            "All imports are conditional/lazy to avoid import cycles. "
            "Verify these imports work at runtime by testing paper_trader initialization. "
            "HealthService is the most fragile — it's a large evolving class (2700+ lines)."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 6: models/predictor.py → features/pipeline.py
    # =========================================================================
    boundaries.append({
        "id": "models_to_features_6",
        "provider_module": "features",
        "consumer_module": "models",
        "symbols": [
            {
                "name": "get_feature_type",
                "type": "function",
                "defined_in": "features/pipeline.py",
                "imported_by": "models/predictor.py",
                "import_line": 22,
                "import_type": "top_level",
                "signature": "def get_feature_type(feature_name: str) -> str",
                "parameters": [
                    {"name": "feature_name", "type": "str", "required": True},
                ],
                "return_type": "str (one of: 'CAUSAL', 'END_OF_DAY', 'RESEARCH_ONLY')",
                "stability": "stable",
                "breaking_change_impact": (
                    "get_feature_type controls causality filtering at prediction time. "
                    "If a feature name is not in FEATURE_METADATA, it defaults to 'CAUSAL'. "
                    "Feature name mismatches between training and prediction cause silent "
                    "data leakage (RESEARCH_ONLY features being used in live prediction). "
                    "This is the single most critical correctness boundary in the system."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "This is the causality enforcement boundary. The predictor filters out "
            "RESEARCH_ONLY features using get_feature_type. If FEATURE_METADATA in "
            "pipeline.py does not include all feature names, they default to CAUSAL "
            "and may leak forward-looking data into predictions. Audit FEATURE_METADATA "
            "completeness against actual computed features."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 7: validation/preconditions.py → config + config_structured
    # =========================================================================
    boundaries.append({
        "id": "validation_to_config_7",
        "provider_module": "config",
        "consumer_module": "validation",
        "symbols": [
            {
                "name": "RET_TYPE",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 16,
                "import_type": "top_level",
                "value_type": "str",
                "current_value": "log",
                "stability": "stable",
                "breaking_change_impact": "Changes return type assumption across entire pipeline.",
            },
            {
                "name": "LABEL_H",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 16,
                "import_type": "top_level",
                "value_type": "int",
                "current_value": "5",
                "stability": "stable",
                "breaking_change_impact": "Changes label horizon assumption across pipeline.",
            },
            {
                "name": "PX_TYPE",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 16,
                "import_type": "top_level",
                "value_type": "str",
                "current_value": "close",
                "stability": "stable",
                "breaking_change_impact": "Changes price column used for features and targets.",
            },
            {
                "name": "ENTRY_PRICE_TYPE",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 16,
                "import_type": "top_level",
                "value_type": "str",
                "current_value": "next_bar_open",
                "stability": "stable",
                "breaking_change_impact": "Changes assumed entry price for trade execution.",
            },
            {
                "name": "TRUTH_LAYER_STRICT_PRECONDITIONS",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 16,
                "import_type": "top_level",
                "value_type": "bool",
                "current_value": "True",
                "stability": "stable",
                "breaking_change_impact": "Disabling strict preconditions removes execution contract enforcement.",
            },
            {
                "name": "PreconditionsConfig",
                "type": "dataclass",
                "defined_in": "config_structured.py",
                "imported_by": "validation/preconditions.py",
                "import_line": 23,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "ret_type", "type": "ReturnType"},
                    {"name": "label_h", "type": "int"},
                    {"name": "px_type", "type": "PriceType"},
                    {"name": "entry_price_type", "type": "EntryType"},
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "PreconditionsConfig is the typed execution contract. Only file "
                    "importing config_structured.PreconditionsConfig. Enum changes "
                    "break validation."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "This is the execution contract enforcement boundary. enforce_preconditions() "
            "is called by Backtester.__init__() and ModelTrainer.__init__(). It validates "
            "that RET_TYPE, LABEL_H, PX_TYPE, and ENTRY_PRICE_TYPE form a consistent "
            "execution contract. The only file importing config_structured.PreconditionsConfig "
            "directly. Verify that flat config values and structured config values agree."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 8: features/pipeline.py → indicators/ (21+ symbols)
    # =========================================================================
    boundaries.append({
        "id": "features_to_indicators_8",
        "provider_module": "indicators",
        "consumer_module": "features",
        "symbols": [
            {
                "name": "90+ indicator classes (top-level import)",
                "type": "class",
                "defined_in": "indicators/__init__.py",
                "imported_by": "features/pipeline.py",
                "import_line": 21,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": (
                    "90+ indicator classes imported at line 21. Each has a compute() method "
                    "that returns a float or np.ndarray. Renaming any indicator class or "
                    "changing compute() signature breaks feature computation. Transitive "
                    "amplification: indicators has 1 direct dependent (features/pipeline.py) "
                    "but 9 transitive dependents through the feature→model→backtest chain."
                ),
            },
            {
                "name": "SpectralAnalyzer",
                "type": "class",
                "defined_in": "indicators/spectral.py",
                "imported_by": "features/pipeline.py",
                "import_line": 769,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "fft_window", "type": "int"},
                    {"name": "cutoff_period", "type": "int"},
                ],
                "key_methods": [
                    {"name": "compute_all", "signature": "def compute_all(self, close: np.ndarray) -> dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break feature column names.",
            },
            {
                "name": "SSADecomposer",
                "type": "class",
                "defined_in": "indicators/ssa.py",
                "imported_by": "features/pipeline.py",
                "import_line": 789,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "window", "type": "int"},
                    {"name": "embed_dim", "type": "int"},
                    {"name": "n_singular", "type": "int"},
                ],
                "key_methods": [
                    {"name": "compute_all", "signature": "def compute_all(self, close: np.ndarray) -> dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break feature column names.",
            },
            {
                "name": "TailRiskAnalyzer",
                "type": "class",
                "defined_in": "indicators/tail_risk.py",
                "imported_by": "features/pipeline.py",
                "import_line": 809,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "window", "type": "int"},
                    {"name": "jump_threshold", "type": "float"},
                ],
                "key_methods": [
                    {"name": "compute_all", "signature": "def compute_all(self, returns: np.ndarray) -> dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break feature column names.",
            },
            {
                "name": "OptimalTransportAnalyzer",
                "type": "class",
                "defined_in": "indicators/ot_divergence.py",
                "imported_by": "features/pipeline.py",
                "import_line": 836,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "window", "type": "int"},
                    {"name": "ref_window", "type": "int"},
                    {"name": "epsilon", "type": "float"},
                    {"name": "max_iter", "type": "int"},
                ],
                "key_methods": [
                    {"name": "compute_all", "signature": "def compute_all(self, returns: np.ndarray) -> dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break feature column names.",
            },
            {
                "name": "EigenvalueAnalyzer",
                "type": "class",
                "defined_in": "indicators/eigenvalue.py",
                "imported_by": "features/pipeline.py",
                "import_line": 1337,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "window", "type": "int"},
                    {"name": "min_assets", "type": "int"},
                    {"name": "regularization", "type": "float"},
                ],
                "key_methods": [
                    {"name": "compute_all", "signature": "def compute_all(self, returns_dict: Dict[str, np.ndarray]) -> dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break feature column names. Requires multi-asset input.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "Extreme transitive amplification: indicators has 1 direct dependent "
            "(features/pipeline.py) but 9 transitive dependents. A signature change "
            "in any indicator class silently propagates through the entire "
            "feature→model→backtest→evaluation chain. 5 advanced analyzers are "
            "conditionally imported — verify they handle ImportError gracefully."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 9: kalshi/ → autopilot/ (promotion gate)
    # =========================================================================
    boundaries.append({
        "id": "kalshi_to_autopilot_9",
        "provider_module": "autopilot",
        "consumer_module": "kalshi",
        "symbols": [
            {
                "name": "PromotionDecision",
                "type": "dataclass",
                "defined_in": "autopilot/promotion_gate.py",
                "imported_by": "kalshi/pipeline.py",
                "import_line": 25,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "candidate", "type": "StrategyCandidate"},
                    {"name": "passed", "type": "bool"},
                    {"name": "score", "type": "float"},
                    {"name": "reasons", "type": "List[str]"},
                    {"name": "metrics", "type": "Dict[str, object]"},
                ],
                "key_methods": [
                    {"name": "to_dict", "signature": "def to_dict(self) -> Dict"},
                ],
                "stability": "stable",
                "breaking_change_impact": "PromotionDecision field changes break Kalshi event promotion pipeline.",
            },
            {
                "name": "PromotionGate",
                "type": "class",
                "defined_in": "autopilot/promotion_gate.py",
                "imported_by": "kalshi/promotion.py",
                "import_line": 12,
                "import_type": "top_level",
                "key_methods": [
                    {
                        "name": "evaluate",
                        "signature": (
                            "def evaluate(self, candidate: StrategyCandidate, "
                            "result: BacktestResult, "
                            "contract_metrics: Optional[Dict[str, object]] = None, "
                            "event_mode: bool = False) -> PromotionDecision"
                        ),
                    },
                    {
                        "name": "evaluate_event_strategy",
                        "signature": (
                            "def evaluate_event_strategy(self, candidate: StrategyCandidate, "
                            "result: BacktestResult, event_metrics: Dict[str, object], "
                            "contract_metrics: Optional[Dict[str, object]] = None) -> PromotionDecision"
                        ),
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": "PromotionGate.evaluate() signature change breaks Kalshi promotion evaluation.",
            },
            {
                "name": "StrategyCandidate",
                "type": "dataclass",
                "defined_in": "autopilot/strategy_discovery.py",
                "imported_by": "kalshi/promotion.py",
                "import_line": 13,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "strategy_id", "type": "str"},
                    {"name": "horizon", "type": "int"},
                    {"name": "entry_threshold", "type": "float"},
                    {"name": "confidence_threshold", "type": "float"},
                    {"name": "use_risk_management", "type": "bool"},
                    {"name": "max_positions", "type": "int"},
                    {"name": "position_size_pct", "type": "float"},
                ],
                "stability": "stable",
                "breaking_change_impact": "StrategyCandidate field changes break Kalshi event strategy construction.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": (
            "Kalshi reuses autopilot's promotion logic. The PromotionGate.evaluate_event_strategy() "
            "method was added specifically for Kalshi event strategies. Verify that "
            "BacktestResult (from backtest/engine.py) is compatible with both standard "
            "and event-mode evaluation."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 10: data/loader.py → validation/data_integrity.py
    # =========================================================================
    boundaries.append({
        "id": "data_to_validation_10",
        "provider_module": "validation",
        "consumer_module": "data",
        "symbols": [
            {
                "name": "DataIntegrityValidator",
                "type": "class",
                "defined_in": "validation/data_integrity.py",
                "imported_by": "data/loader.py",
                "import_line": 567,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "fail_fast", "type": "bool"},
                ],
                "key_methods": [
                    {
                        "name": "validate_universe",
                        "signature": "def validate_universe(self, ohlcv_dict: Dict[str, pd.DataFrame]) -> DataIntegrityCheckResult",
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": (
                    "DataIntegrityValidator is the quality gate that blocks corrupt data "
                    "from entering the pipeline. If validate_universe() raises instead of "
                    "returning DataIntegrityCheckResult, it will halt data loading. "
                    "Loosening quality checks allows corrupt data through."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": (
            "Lazy import at data/loader.py:567 inside load_survivorship_universe(). "
            "Only triggers when DATA_QUALITY_ENABLED=True. The validator delegates "
            "to data/quality.py assess_ohlcv_quality() for per-ticker checks. "
            "Verify DataIntegrityCheckResult.passed semantics match consumer expectations."
        ),
    })

    # =========================================================================
    # CRITICAL BOUNDARY 11: evaluation/ → models/ + backtest/
    # =========================================================================
    boundaries.append({
        "id": "evaluation_to_models_backtest_11",
        "provider_module": "models, backtest",
        "consumer_module": "evaluation",
        "symbols": [
            {
                "name": "compute_ece",
                "type": "function",
                "defined_in": "models/calibration.py",
                "imported_by": "evaluation/calibration_analysis.py",
                "import_line": 92,
                "import_type": "lazy",
                "signature": "def compute_ece(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10) -> float",
                "parameters": [
                    {"name": "predicted_probs", "type": "np.ndarray", "required": True},
                    {"name": "actual_outcomes", "type": "np.ndarray", "required": True},
                    {"name": "n_bins", "type": "int", "required": False},
                ],
                "return_type": "float",
                "stability": "stable",
                "breaking_change_impact": "ECE computation change alters calibration diagnostics.",
            },
            {
                "name": "compute_reliability_curve",
                "type": "function",
                "defined_in": "models/calibration.py",
                "imported_by": "evaluation/calibration_analysis.py",
                "import_line": 92,
                "import_type": "lazy",
                "signature": "def compute_reliability_curve(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10) -> dict",
                "parameters": [
                    {"name": "predicted_probs", "type": "np.ndarray", "required": True},
                    {"name": "actual_outcomes", "type": "np.ndarray", "required": True},
                    {"name": "n_bins", "type": "int", "required": False},
                ],
                "return_type": "dict (keys: bin_centers, observed_freq, avg_predicted, bin_counts)",
                "stability": "stable",
                "breaking_change_impact": "Return dict key changes break calibration visualization.",
            },
            {
                "name": "walk_forward_with_embargo",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "evaluation/engine.py",
                "import_line": 272,
                "import_type": "conditional",
                "signature": (
                    "def walk_forward_with_embargo(returns: pd.Series, "
                    "predictions: np.ndarray, train_window: int = 250, "
                    "embargo: int = 5, test_window: int = 60, "
                    "slide_freq: str = 'weekly', "
                    "risk_free_rate: float = 0.04) -> WalkForwardEmbargoResult"
                ),
                "return_type": "WalkForwardEmbargoResult",
                "stability": "stable",
                "breaking_change_impact": "Walk-forward evaluation change alters overfitting diagnostics.",
            },
            {
                "name": "rolling_ic",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "evaluation/engine.py",
                "import_line": 315,
                "import_type": "conditional",
                "signature": "def rolling_ic(predictions: np.ndarray, returns: pd.Series, window: int = 60) -> pd.Series",
                "return_type": "pd.Series",
                "stability": "stable",
                "breaking_change_impact": "IC computation change alters decay detection.",
            },
            {
                "name": "detect_ic_decay",
                "type": "function",
                "defined_in": "backtest/validation.py",
                "imported_by": "evaluation/engine.py",
                "import_line": 315,
                "import_type": "conditional",
                "signature": "def detect_ic_decay(ic_series: pd.Series, decay_threshold: float = 0.02, window: int = 20) -> tuple",
                "return_type": "tuple[bool, dict]",
                "stability": "stable",
                "breaking_change_impact": "Decay detection threshold change alters model retirement signals.",
            },
        ],
        "shared_artifacts": [
            {
                "path": "results/backtest_*d_summary.json",
                "format": "json",
                "schema_owner": "backtest",
                "writer": "run_backtest.py",
                "readers": ["api/services/backtest_service.py", "api/services/results_service.py", "evaluation/engine.py"],
                "schema_fields": [
                    "horizon", "total_trades", "winning_trades", "losing_trades",
                    "win_rate", "avg_return", "avg_win", "avg_loss",
                    "total_return", "annualized_return", "sharpe", "sortino",
                    "max_drawdown", "profit_factor", "avg_holding_days",
                    "trades_per_year", "regime_breakdown",
                ],
                "breaking_change_impact": (
                    "Summary JSON key changes break API backtest results display "
                    "and evaluation engine metrics extraction."
                ),
            },
        ],
        "boundary_risk": "MEDIUM",
        "audit_notes": (
            "Evaluation module reads from both models (calibration) and backtest (validation). "
            "All imports are lazy/conditional — import failures only surface at evaluation time. "
            "Verify that WalkForwardEmbargoResult fields match what evaluation/engine.py expects."
        ),
    })

    # =========================================================================
    # NON-CRITICAL BOUNDARIES: config imports (grouped by consumer)
    # =========================================================================

    # --- data → config (10 edges) ---
    boundaries.append({
        "id": "data_to_config_12",
        "provider_module": "config",
        "consumer_module": "data",
        "symbols": [
            {
                "name": "DATA_CACHE_DIR, FRAMEWORK_DIR, LOOKBACK_YEARS, MIN_BARS, and 20+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "data/loader.py, data/local_cache.py, data/quality.py, data/feature_store.py, data/survivorship.py, data/intraday_quality.py",
                "import_line": "multiple (34, 22, 25, 31, 371, 500, 601, 701, 748, 25)",
                "import_type": "mixed (top_level + conditional)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "Config constants control cache paths, quality thresholds, and data source "
                    "selection. Path changes break cache lookups. Threshold changes alter "
                    "which data passes quality gates."
                ),
            },
        ],
        "shared_artifacts": [
            {
                "path": "data/cache/**/*.parquet",
                "format": "parquet",
                "schema_owner": "data",
                "writer": "data/local_cache.py",
                "readers": ["data/loader.py", "features/pipeline.py"],
                "schema_fields": ["Open", "High", "Low", "Close", "Volume", "date"],
                "breaking_change_impact": "Column name changes break all downstream feature computation.",
            },
        ],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "10 import edges from data → config. data/loader.py:34 imports 10 constants "
            "at top level. data/survivorship.py and data/local_cache.py use conditional "
            "imports for SURVIVORSHIP_DB. Verify all config paths exist."
        ),
    })

    # --- features → config (8 edges) ---
    boundaries.append({
        "id": "features_to_config_13",
        "provider_module": "config",
        "consumer_module": "features",
        "symbols": [
            {
                "name": "FORWARD_HORIZONS, BENCHMARK, LOOKBACK_YEARS, INTERACTION_PAIRS, spectral/SSA/jump config, and 15+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "features/pipeline.py, features/intraday.py",
                "import_line": "multiple (750, 871, 924, 1527, 1120, 1326, 1460, 21)",
                "import_type": "mixed (lazy + conditional)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "Config constants control feature computation parameters. Changes to "
                    "FORWARD_HORIZONS alter target columns. Changes to indicator config "
                    "alter feature values across the entire pipeline."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": (
            "8 import edges, mostly lazy/conditional inside feature computation methods. "
            "STRUCTURAL_FEATURES_ENABLED at lines 1120 and 1326 gates entire feature blocks."
        ),
    })

    # --- features → data (3 edges) ---
    boundaries.append({
        "id": "features_to_data_14",
        "provider_module": "data",
        "consumer_module": "features",
        "symbols": [
            {
                "name": "load_ohlcv",
                "type": "function",
                "defined_in": "data/loader.py",
                "imported_by": "features/pipeline.py",
                "import_line": 1528,
                "import_type": "lazy",
                "signature": "def load_ohlcv(ticker: str, years: int = 15, use_cache: bool = True, use_wrds: bool = WRDS_ENABLED) -> Optional[pd.DataFrame]",
                "return_type": "Optional[pd.DataFrame]",
                "stability": "stable",
                "breaking_change_impact": "OHLCV loading change affects benchmark relative features.",
            },
            {
                "name": "WRDSProvider",
                "type": "class",
                "defined_in": "data/wrds_provider.py",
                "imported_by": "features/pipeline.py",
                "import_line": 1413,
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "WRDSProvider API change breaks option surface feature computation.",
            },
            {
                "name": "load_intraday_ohlcv",
                "type": "function",
                "defined_in": "data/local_cache.py",
                "imported_by": "features/pipeline.py",
                "import_line": 1458,
                "import_type": "conditional",
                "signature": "def load_intraday_ohlcv(ticker: str, timeframe: str, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]",
                "return_type": "Optional[pd.DataFrame]",
                "stability": "stable",
                "breaking_change_impact": "Intraday data format change breaks microstructure features.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": (
            "All 3 imports are lazy/conditional inside specialized feature methods. "
            "WRDSProvider is only used for OptionMetrics surface features. "
            "load_intraday_ohlcv is only used when INTRADAY_MIN_BARS config is met."
        ),
    })

    # --- features → regime (1 edge) ---
    boundaries.append({
        "id": "features_to_regime_15",
        "provider_module": "regime",
        "consumer_module": "features",
        "symbols": [
            {
                "name": "CorrelationRegimeDetector",
                "type": "class",
                "defined_in": "regime/correlation.py",
                "imported_by": "features/pipeline.py",
                "import_line": 1303,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "window", "type": "int"},
                    {"name": "z_score_lookback", "type": "int"},
                    {"name": "threshold", "type": "float"},
                ],
                "key_methods": [
                    {"name": "get_correlation_features", "signature": "def get_correlation_features(self, returns_dict: Dict[str, pd.Series], ...) -> pd.DataFrame"},
                ],
                "stability": "stable",
                "breaking_change_impact": "Correlation feature column name changes break feature pipeline.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "Conditional import gated by STRUCTURAL_FEATURES_ENABLED config flag.",
    })

    # --- regime → config (11 edges) ---
    boundaries.append({
        "id": "regime_to_config_16",
        "provider_module": "config",
        "consumer_module": "regime",
        "symbols": [
            {
                "name": "REGIME_MODEL_TYPE, REGIME_HMM_STATES, BOCPD_*, SHOCK_VECTOR_*, and 25+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "regime/detector.py, regime/hmm.py, regime/consensus.py, regime/jump_model_pypi.py, regime/online_update.py, regime/uncertainty_gate.py",
                "import_line": "multiple (24, 240, 324, 402, 933, 560, 55, 84, 314, 60, 52)",
                "import_type": "mixed (top_level + lazy + conditional)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "Config constants control regime detection algorithm parameters. "
                    "Changes to HMM states, BOCPD thresholds, or ensemble weights "
                    "alter regime labels propagated to all downstream modules."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "11 import edges from regime → config. regime/detector.py:24 imports 20+ "
            "constants at top level. Regime label changes propagate to backtest, risk, "
            "autopilot, and API. Verify REGIME_NAMES consistency across all consumers."
        ),
    })

    # --- models → config (7 edges) ---
    boundaries.append({
        "id": "models_to_config_17",
        "provider_module": "config",
        "consumer_module": "models",
        "symbols": [
            {
                "name": "MODEL_DIR, MODEL_PARAMS, MAX_FEATURES_SELECTED, CV_FOLDS, REGIME_NAMES, and 15+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "models/trainer.py, models/predictor.py, models/versioning.py, models/governance.py, models/retrain_trigger.py, models/feature_stability.py",
                "import_line": "multiple (77, 21, 22, 10, 11, 31, 26)",
                "import_type": "top_level",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "Config constants control model training hyperparameters, paths, "
                    "and governance settings. MODEL_DIR path change breaks model "
                    "save/load. REGIME_NAMES mismatch breaks regime-specific models."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "7 import edges, all top-level. models/trainer.py:77 imports 17 constants. "
            "MODEL_DIR is used by trainer, predictor, versioning, and retrain_trigger."
        ),
    })

    # --- models → validation (1 edge) ---
    boundaries.append({
        "id": "models_to_validation_18",
        "provider_module": "validation",
        "consumer_module": "models",
        "symbols": [
            {
                "name": "enforce_preconditions",
                "type": "function",
                "defined_in": "validation/preconditions.py",
                "imported_by": "models/trainer.py",
                "import_line": 219,
                "import_type": "lazy",
                "signature": "def enforce_preconditions() -> None",
                "return_type": "None (raises RuntimeError on failure)",
                "stability": "stable",
                "breaking_change_impact": "enforce_preconditions() is called in ModelTrainer.__init__(). Failure blocks all training.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Lazy import in ModelTrainer.__init__() when TRUTH_LAYER_STRICT_PRECONDITIONS=True.",
    })

    # --- backtest → config (5 edges) ---
    boundaries.append({
        "id": "backtest_to_config_19",
        "provider_module": "config",
        "consumer_module": "backtest",
        "symbols": [
            {
                "name": "TRANSACTION_COST_BPS, ENTRY_THRESHOLD, EXEC_*, SHOCK_MODE_*, and 55+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "backtest/engine.py, backtest/optimal_execution.py, backtest/validation.py",
                "import_line": "multiple (26, 2375, 1713, 17, 23)",
                "import_type": "mixed (top_level + lazy)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "backtest/engine.py:26 imports 55+ constants controlling execution "
                    "costs, thresholds, and shock mode. Any change alters backtest results."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "backtest/engine.py:26 has the largest single import statement in the codebase "
            "(55+ constants). Many are EXEC_* execution parameters. Changes to any "
            "default backtest parameter silently alter historical results."
        ),
    })

    # --- backtest → validation (1 edge) ---
    boundaries.append({
        "id": "backtest_to_validation_20",
        "provider_module": "validation",
        "consumer_module": "backtest",
        "symbols": [
            {
                "name": "enforce_preconditions",
                "type": "function",
                "defined_in": "validation/preconditions.py",
                "imported_by": "backtest/engine.py",
                "import_line": 198,
                "import_type": "lazy",
                "signature": "def enforce_preconditions() -> None",
                "return_type": "None (raises RuntimeError on failure)",
                "stability": "stable",
                "breaking_change_impact": "enforce_preconditions() is called in Backtester.__init__(). Failure blocks all backtesting.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Lazy import in Backtester.__init__() when TRUTH_LAYER_STRICT_PRECONDITIONS=True.",
    })

    # --- risk → config (8 edges) ---
    boundaries.append({
        "id": "risk_to_config_21",
        "provider_module": "config",
        "consumer_module": "risk",
        "symbols": [
            {
                "name": "DRAWDOWN_*, HARD_STOP_PCT, ATR_STOP_MULTIPLIER, MAX_PORTFOLIO_VOL, REGIME_*, and 30+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "risk/drawdown.py, risk/position_sizer.py, risk/stop_loss.py, risk/portfolio_risk.py, risk/portfolio_optimizer.py",
                "import_line": "multiple (18, 136, 819, 23, 24, 190, 202)",
                "import_type": "mixed (top_level + conditional + lazy)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "Risk config constants control position sizing, drawdown thresholds, "
                    "stop-loss parameters, and portfolio constraints. Changes silently "
                    "alter risk behavior across backtest and paper trading."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "8 import edges from risk → config. risk/position_sizer.py:136 imports 17+ "
            "constants conditionally. risk/stop_loss.py uses REGIME_STOP_MULTIPLIER "
            "which maps regime codes to stop multipliers."
        ),
    })

    # --- risk → regime (1 edge) ---
    boundaries.append({
        "id": "risk_to_regime_22",
        "provider_module": "regime",
        "consumer_module": "risk",
        "symbols": [
            {
                "name": "UncertaintyGate",
                "type": "class",
                "defined_in": "regime/uncertainty_gate.py",
                "imported_by": "risk/position_sizer.py",
                "import_line": 27,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "UncertaintyGate threshold changes alter position sizing in risk module.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "UncertaintyGate is also imported by backtest/engine.py and autopilot/engine.py. Verify consistent usage.",
    })

    # --- evaluation → config (7 edges) ---
    boundaries.append({
        "id": "evaluation_to_config_23",
        "provider_module": "config",
        "consumer_module": "evaluation",
        "symbols": [
            {
                "name": "EVAL_WF_*, EVAL_IC_*, EVAL_CALIBRATION_*, EVAL_TOP_N_TRADES, and 15+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "evaluation/engine.py, evaluation/calibration_analysis.py, evaluation/fragility.py, evaluation/metrics.py, evaluation/ml_diagnostics.py, evaluation/slicing.py",
                "import_line": "multiple (27, 19, 20, 20, 26, 23, 193)",
                "import_type": "mixed (top_level + lazy)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": "Evaluation config changes alter diagnostic outputs and alert thresholds.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "7 import edges, mostly top-level. evaluation/slicing.py:193 lazy-imports REGIME_NAMES.",
    })

    # --- validation → data (1 edge) ---
    boundaries.append({
        "id": "validation_to_data_24",
        "provider_module": "data",
        "consumer_module": "validation",
        "symbols": [
            {
                "name": "assess_ohlcv_quality",
                "type": "function",
                "defined_in": "data/quality.py",
                "imported_by": "validation/data_integrity.py",
                "import_line": 20,
                "import_type": "top_level",
                "signature": (
                    "def assess_ohlcv_quality(df: pd.DataFrame, "
                    "max_missing_bar_fraction: float = MAX_MISSING_BAR_FRACTION, "
                    "max_zero_volume_fraction: float = MAX_ZERO_VOLUME_FRACTION, "
                    "max_abs_daily_return: float = MAX_ABS_DAILY_RETURN, "
                    "fail_on_error: bool = False) -> DataQualityReport"
                ),
                "return_type": "DataQualityReport",
                "stability": "stable",
                "breaking_change_impact": "Quality assessment change alters which data passes integrity checks.",
            },
            {
                "name": "DataQualityReport",
                "type": "dataclass",
                "defined_in": "data/quality.py",
                "imported_by": "validation/data_integrity.py",
                "import_line": 20,
                "import_type": "top_level",
                "constructor_params": [
                    {"name": "passed", "type": "bool"},
                    {"name": "metrics", "type": "Dict[str, float]"},
                    {"name": "warnings", "type": "List[str]"},
                ],
                "stability": "stable",
                "breaking_change_impact": "DataQualityReport field changes break DataIntegrityValidator.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "Clean provider-consumer relationship. validation delegates quality checks to data/quality.py.",
    })

    # --- autopilot → config (8 edges) ---
    boundaries.append({
        "id": "autopilot_to_config_25",
        "provider_module": "config",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "AUTOPILOT_CYCLE_REPORT, PROMOTION_*, PAPER_*, EXEC_*, STRATEGY_REGISTRY_PATH, and 80+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "autopilot/engine.py, autopilot/paper_trader.py, autopilot/promotion_gate.py, autopilot/registry.py, autopilot/strategy_allocator.py, autopilot/strategy_discovery.py, autopilot/meta_labeler.py",
                "import_line": "multiple (34, 13, 13, 10, 20, 10, 23)",
                "import_type": "top_level",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "autopilot/engine.py:34 imports 18+ constants. "
                    "autopilot/paper_trader.py:13 imports 40+ EXEC_* and PAPER_* constants. "
                    "autopilot/promotion_gate.py:13 imports 29+ PROMOTION_* constants. "
                    "Config changes alter autopilot behavior across all subsystems."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "8 import edges with massive constant surface (80+ unique constants). "
            "paper_trader.py:13 has the second-largest import statement in the codebase. "
            "All imports are top-level so config parse failures block autopilot startup."
        ),
    })

    # --- autopilot → data (2 edges) ---
    boundaries.append({
        "id": "autopilot_to_data_26",
        "provider_module": "data",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "load_survivorship_universe",
                "type": "function",
                "defined_in": "data/loader.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 54,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Universe loading change breaks autopilot data pipeline.",
            },
            {
                "name": "load_universe",
                "type": "function",
                "defined_in": "data/loader.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 54,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Universe loading change breaks autopilot data pipeline.",
            },
            {
                "name": "filter_panel_by_point_in_time_universe",
                "type": "function",
                "defined_in": "data/survivorship.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 55,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Survivorship filter change alters universe composition.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Top-level imports in autopilot/engine.py. Universe data flows into the full pipeline.",
    })

    # --- autopilot → features (1 edge) ---
    boundaries.append({
        "id": "autopilot_to_features_27",
        "provider_module": "features",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "FeaturePipeline",
                "type": "class",
                "defined_in": "features/pipeline.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 56,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Feature column changes break model training/prediction alignment.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Single import. FeaturePipeline is the central feature computation entry point.",
    })

    # --- autopilot → models (5 edges) ---
    boundaries.append({
        "id": "autopilot_to_models_28",
        "provider_module": "models",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "_expanding_walk_forward_folds",
                "type": "function",
                "defined_in": "models/walk_forward.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 33,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Fold generation change alters training splits.",
            },
            {
                "name": "cross_sectional_rank",
                "type": "function",
                "defined_in": "models/cross_sectional.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 57,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Ranking logic change alters signal selection.",
            },
            {
                "name": "EnsemblePredictor",
                "type": "class",
                "defined_in": "models/predictor.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 58,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Predictor API change breaks all prediction paths.",
            },
            {
                "name": "ModelTrainer",
                "type": "class",
                "defined_in": "models/trainer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 59,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Trainer interface change breaks autopilot training.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "5 import edges from autopilot → models. All top-level except "
            "cross_sectional_rank which is also lazy-imported at line 565. "
            "_expanding_walk_forward_folds is a private function used by autopilot."
        ),
    })

    # --- autopilot → regime (2 edges) ---
    boundaries.append({
        "id": "autopilot_to_regime_29",
        "provider_module": "regime",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "RegimeDetector",
                "type": "class",
                "defined_in": "regime/detector.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 60,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Regime detection changes propagate to all autopilot decisions.",
            },
            {
                "name": "UncertaintyGate",
                "type": "class",
                "defined_in": "regime/uncertainty_gate.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 61,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Uncertainty gating changes alter position sizing.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Top-level imports. RegimeDetector and UncertaintyGate are shared with backtest and risk modules.",
    })

    # --- autopilot → risk (7 edges) ---
    boundaries.append({
        "id": "autopilot_to_risk_30",
        "provider_module": "risk",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "optimize_portfolio",
                "type": "function",
                "defined_in": "risk/portfolio_optimizer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 62,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Portfolio optimization change alters all allocation decisions.",
            },
            {
                "name": "PositionSizer",
                "type": "class",
                "defined_in": "risk/position_sizer.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 971,
                "import_type": "lazy",
                "stability": "evolving",
                "breaking_change_impact": "Position sizing changes affect all trade decisions.",
            },
            {
                "name": "CovarianceEstimator, compute_regime_covariance, get_regime_covariance",
                "type": "class + functions",
                "defined_in": "risk/covariance.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 1528,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "Covariance estimation changes alter portfolio optimization inputs.",
            },
            {
                "name": "StopLossManager",
                "type": "class",
                "defined_in": "risk/stop_loss.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 59,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Stop-loss logic changes alter paper trading exit decisions.",
            },
            {
                "name": "PortfolioRiskManager",
                "type": "class",
                "defined_in": "risk/portfolio_risk.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 60,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Risk constraint changes alter which paper trades are allowed.",
            },
            {
                "name": "DrawdownController",
                "type": "class",
                "defined_in": "risk/drawdown.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 99,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "Drawdown threshold changes alter paper trading position sizing.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "7 import edges from autopilot → risk. Mix of top-level (paper_trader) "
            "and lazy (engine). autopilot/paper_trader.py also imports PositionSizer "
            "at line 58 (top-level) and ExecutionModel, ADVTracker, CostCalibrator "
            "from backtest."
        ),
    })

    # --- autopilot → backtest (8 edges) ---
    boundaries.append({
        "id": "autopilot_to_backtest_31",
        "provider_module": "backtest",
        "consumer_module": "autopilot",
        "symbols": [
            {
                "name": "Backtester",
                "type": "class",
                "defined_in": "backtest/engine.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 20,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Backtester interface change breaks autopilot strategy evaluation.",
            },
            {
                "name": "BacktestResult",
                "type": "dataclass",
                "defined_in": "backtest/engine.py",
                "imported_by": "autopilot/promotion_gate.py",
                "import_line": 12,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "BacktestResult field changes break promotion gate evaluation.",
            },
            {
                "name": "ExecutionModel",
                "type": "class",
                "defined_in": "backtest/execution.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 55,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Execution model changes alter paper trade fill simulation.",
            },
            {
                "name": "ADVTracker",
                "type": "class",
                "defined_in": "backtest/adv_tracker.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 56,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "ADV tracking changes affect participation rate calculations.",
            },
            {
                "name": "CostCalibrator",
                "type": "class",
                "defined_in": "backtest/cost_calibrator.py",
                "imported_by": "autopilot/paper_trader.py",
                "import_line": 57,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Cost calibration changes alter execution cost estimates.",
            },
            {
                "name": "walk_forward_validate, run_statistical_tests, combinatorial_purged_cv, superior_predictive_ability, strategy_signal_returns",
                "type": "functions",
                "defined_in": "backtest/validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 26,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Validation function changes alter strategy evaluation outcomes.",
            },
            {
                "name": "capacity_analysis, deflated_sharpe_ratio, probability_of_backtest_overfitting",
                "type": "functions",
                "defined_in": "backtest/advanced_validation.py",
                "imported_by": "autopilot/engine.py",
                "import_line": 21,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Advanced validation changes alter promotion gate decisions.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "8 import edges from autopilot → backtest. All top-level except "
            "ExecutionModel at engine.py:1405 (conditional). This is the largest "
            "single-target dependency from autopilot."
        ),
    })

    # --- data → kalshi (1 edge) ---
    boundaries.append({
        "id": "data_to_kalshi_32",
        "provider_module": "kalshi",
        "consumer_module": "data",
        "symbols": [
            {
                "name": "KalshiProvider",
                "type": "class",
                "defined_in": "kalshi/provider.py",
                "imported_by": "data/provider_registry.py",
                "import_line": 23,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "KalshiProvider interface change breaks data provider registry.",
            },
        ],
        "shared_artifacts": [
            {
                "path": "data/kalshi.duckdb",
                "format": "duckdb",
                "schema_owner": "kalshi",
                "writer": "kalshi/storage.py",
                "readers": ["kalshi/pipeline.py", "kalshi/events.py", "api/services/kalshi_service.py"],
                "schema_fields": [
                    "kalshi_markets", "kalshi_contracts", "kalshi_quotes",
                    "kalshi_fees", "macro_events", "macro_events_versioned",
                    "event_outcomes", "event_outcomes_first_print",
                    "event_outcomes_revised", "kalshi_distributions",
                    "event_market_map_versions", "kalshi_market_specs",
                    "kalshi_contract_specs", "kalshi_data_provenance",
                    "kalshi_coverage_diagnostics", "kalshi_ingestion_logs",
                    "kalshi_daily_health_report", "kalshi_ingestion_checkpoints",
                ],
                "breaking_change_impact": (
                    "DuckDB schema change (DDL) breaks all Kalshi pipeline queries. "
                    "18 tables with versioning and audit trail. Column changes require "
                    "migration of existing databases."
                ),
            },
        ],
        "boundary_risk": "LOW",
        "audit_notes": "Lazy import inside factory function in data/provider_registry.py — optional coupling.",
    })

    # --- kalshi → backtest (2 edges) ---
    boundaries.append({
        "id": "kalshi_to_backtest_33",
        "provider_module": "backtest",
        "consumer_module": "kalshi",
        "symbols": [
            {
                "name": "BacktestResult",
                "type": "dataclass",
                "defined_in": "backtest/engine.py",
                "imported_by": "kalshi/promotion.py",
                "import_line": 14,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "BacktestResult field changes break Kalshi promotion evaluation.",
            },
            {
                "name": "deflated_sharpe_ratio, monte_carlo_validation",
                "type": "functions",
                "defined_in": "backtest/advanced_validation.py",
                "imported_by": "kalshi/walkforward.py",
                "import_line": 12,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Validation function changes alter Kalshi strategy evaluation.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Kalshi reuses backtest validation infrastructure for event strategy evaluation.",
    })

    # --- kalshi → config (1 edge) ---
    boundaries.append({
        "id": "kalshi_to_config_34",
        "provider_module": "config",
        "consumer_module": "kalshi",
        "symbols": [
            {
                "name": "KALSHI_API_BASE_URL, KALSHI_ENV, KALSHI_RATE_LIMIT_*, KALSHI_STALE_*, and 18 constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "kalshi/provider.py",
                "import_line": 14,
                "import_type": "top_level",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": "Kalshi config changes alter API connection and staleness detection.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "kalshi/provider.py:14 imports 18 KALSHI_* constants at top level.",
    })

    # --- kalshi → features (1 edge) ---
    boundaries.append({
        "id": "kalshi_to_features_35",
        "provider_module": "features",
        "consumer_module": "kalshi",
        "symbols": [
            {
                "name": "compute_option_surface_factors",
                "type": "function",
                "defined_in": "features/options_factors.py",
                "imported_by": "kalshi/options.py",
                "import_line": 11,
                "import_type": "top_level",
                "stability": "stable",
                "breaking_change_impact": "Option surface factor computation change breaks Kalshi options integration.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "Single top-level import for option surface factor reuse.",
    })

    # --- api → config (82 edges) ---
    boundaries.append({
        "id": "api_to_config_36",
        "provider_module": "config",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "82 import edges spanning DATA_CACHE_DIR, RESULTS_DIR, MODEL_DIR, UNIVERSE_*, REGIME_*, EXEC_*, and 60+ constants",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "api/config.py, api/main.py, api/orchestrator.py, api/routers/*, api/services/*",
                "import_line": "82 locations across 20+ files",
                "import_type": "mixed (lazy + conditional)",
                "value_type": "mixed",
                "stability": "stable",
                "breaking_change_impact": (
                    "API is the largest consumer of config (82 edges, 52% of all config imports). "
                    "Path constant changes break file lookups. Threshold changes alter API responses. "
                    "All API config imports are lazy or conditional to avoid circular imports."
                ),
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "82 import edges — the largest cross-module dependency in the codebase. "
            "All are lazy or conditional (no top-level config imports in API). "
            "api/services/health_service.py alone has 25+ config import sites. "
            "Verify all lazy imports resolve at runtime."
        ),
    })

    # --- api → data (7 edges) ---
    boundaries.append({
        "id": "api_to_data_37",
        "provider_module": "data",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "load_universe, load_survivorship_universe, get_skip_reasons, get_data_provenance, WRDSProvider",
                "type": "mixed (functions + class)",
                "defined_in": "data/loader.py, data/wrds_provider.py",
                "imported_by": "api/orchestrator.py, api/services/data_service.py, api/routers/system_health.py, api/services/health_service.py",
                "import_line": "multiple (43, 61, 97, 24, 43, 1262, 151)",
                "import_type": "mixed (lazy + conditional)",
                "stability": "stable",
                "breaking_change_impact": "Data loading function changes affect API data endpoints and health checks.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "7 edges. WRDSProvider is imported conditionally for health checks only.",
    })

    # --- api → features (2 edges) ---
    boundaries.append({
        "id": "api_to_features_38",
        "provider_module": "features",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "FeaturePipeline",
                "type": "class",
                "defined_in": "features/pipeline.py",
                "imported_by": "api/orchestrator.py, api/services/data_helpers.py",
                "import_line": "44, 490",
                "import_type": "mixed (lazy + conditional)",
                "stability": "stable",
                "breaking_change_impact": "FeaturePipeline API change breaks API orchestration endpoints.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "2 edges, both lazy/conditional. FeaturePipeline used in API orchestrator.",
    })

    # --- api → regime (4 edges) ---
    boundaries.append({
        "id": "api_to_regime_39",
        "provider_module": "regime",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "RegimeDetector",
                "type": "class",
                "defined_in": "regime/detector.py",
                "imported_by": "api/orchestrator.py, api/services/data_helpers.py",
                "import_line": "45, 225, 292, 491",
                "import_type": "mixed (lazy + conditional)",
                "stability": "stable",
                "breaking_change_impact": "RegimeDetector API change breaks API regime endpoints.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "4 edges, all lazy. RegimeDetector used in orchestrator pipeline steps.",
    })

    # --- api → models (13 edges) ---
    boundaries.append({
        "id": "api_to_models_40",
        "provider_module": "models",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "ModelGovernance, ModelTrainer, ModelRegistry, EnsemblePredictor",
                "type": "classes",
                "defined_in": "models/governance.py, models/trainer.py, models/versioning.py, models/predictor.py",
                "imported_by": "api/orchestrator.py, api/routers/autopilot.py, api/routers/signals.py, api/services/model_service.py",
                "import_line": "multiple (160-162, 224, 291, 30, 30, 17, 91)",
                "import_type": "mixed (lazy + conditional)",
                "stability": "stable",
                "breaking_change_impact": "Model class API changes break API model management and prediction endpoints.",
            },
            {
                "name": "ArbitrageFreeSVIBuilder, generate_synthetic_market_surface",
                "type": "class + function",
                "defined_in": "models/iv/models.py",
                "imported_by": "api/routers/iv_surface.py",
                "import_line": 17,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "IV model changes break API IV surface endpoint.",
            },
            {
                "name": "RETRAIN_MAX_DAYS, RetrainTrigger",
                "type": "constant + class",
                "defined_in": "models/retrain_trigger.py",
                "imported_by": "api/services/backtest_service.py, api/services/data_helpers.py",
                "import_line": "106, 596",
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "Retrain trigger changes affect API model freshness monitoring.",
            },
            {
                "name": "FeatureStabilityTracker",
                "type": "class",
                "defined_in": "models/feature_stability.py",
                "imported_by": "api/services/health_service.py",
                "import_line": 1800,
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "Feature stability tracking changes affect health dashboard.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "HIGH",
        "audit_notes": (
            "13 import edges from api → models. All lazy or conditional. "
            "API is the largest consumer of models module."
        ),
    })

    # --- api → backtest (3 edges) ---
    boundaries.append({
        "id": "api_to_backtest_41",
        "provider_module": "backtest",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "Backtester",
                "type": "class",
                "defined_in": "backtest/engine.py",
                "imported_by": "api/orchestrator.py",
                "import_line": 289,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "Backtester API change breaks API backtest orchestration.",
            },
            {
                "name": "WalkForwardFold, run_statistical_tests",
                "type": "dataclass + function",
                "defined_in": "backtest/validation.py",
                "imported_by": "api/services/data_helpers.py",
                "import_line": "900, 911",
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "Validation dataclass changes break API data helper serialization.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "3 edges, all lazy/conditional. Backtester used in API orchestrator.",
    })

    # --- api → risk (1 edge) ---
    boundaries.append({
        "id": "api_to_risk_42",
        "provider_module": "risk",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "FactorExposureMonitor",
                "type": "class",
                "defined_in": "risk/factor_monitor.py",
                "imported_by": "api/routers/risk.py",
                "import_line": 27,
                "import_type": "conditional",
                "constructor_params": [
                    {"name": "limits", "type": "Optional[Dict[str, Tuple[float, float]]]"},
                    {"name": "lookback_days", "type": "int"},
                    {"name": "beta_lookback", "type": "int"},
                ],
                "key_methods": [
                    {
                        "name": "compute_report",
                        "signature": (
                            "def compute_report(self, positions: Dict[str, float], "
                            "price_data: Dict[str, pd.DataFrame], "
                            "benchmark_returns: Optional[pd.Series] = None) -> FactorExposureReport"
                        ),
                    },
                ],
                "stability": "stable",
                "breaking_change_impact": "FactorExposureMonitor changes break API risk endpoint.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "Single conditional import in API risk router.",
    })

    # --- api → autopilot (1 edge) ---
    boundaries.append({
        "id": "api_to_autopilot_43",
        "provider_module": "autopilot",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "AutopilotEngine",
                "type": "class",
                "defined_in": "autopilot/engine.py",
                "imported_by": "api/jobs/autopilot_job.py",
                "import_line": 12,
                "import_type": "lazy",
                "stability": "stable",
                "breaking_change_impact": "AutopilotEngine interface change breaks API autopilot job execution.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "MEDIUM",
        "audit_notes": "Lazy import in API job runner. AutopilotEngine is the main entry point.",
    })

    # --- api → kalshi (2 edges) ---
    boundaries.append({
        "id": "api_to_kalshi_44",
        "provider_module": "kalshi",
        "consumer_module": "api",
        "symbols": [
            {
                "name": "EventTimeStore",
                "type": "class",
                "defined_in": "kalshi/storage.py",
                "imported_by": "api/services/kalshi_service.py",
                "import_line": "24, 42",
                "import_type": "conditional",
                "stability": "stable",
                "breaking_change_impact": "EventTimeStore API change breaks API Kalshi endpoints.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "2 conditional imports in API Kalshi service.",
    })

    # --- utils → config (1 edge) ---
    boundaries.append({
        "id": "utils_to_config_45",
        "provider_module": "config",
        "consumer_module": "utils",
        "symbols": [
            {
                "name": "ALERT_HISTORY_FILE, ALERT_WEBHOOK_URL",
                "type": "constant",
                "defined_in": "config.py",
                "imported_by": "utils/logging.py",
                "import_line": 114,
                "import_type": "lazy",
                "value_type": "str",
                "stability": "stable",
                "breaking_change_impact": "Alert config changes affect logging webhook behavior.",
            },
        ],
        "shared_artifacts": [],
        "boundary_risk": "LOW",
        "audit_notes": "Single lazy import for alert configuration.",
    })

    # Compute risk counts
    high_risk = sum(1 for b in boundaries if b["boundary_risk"] == "HIGH")
    medium_risk = sum(1 for b in boundaries if b["boundary_risk"] == "MEDIUM")
    low_risk = sum(1 for b in boundaries if b["boundary_risk"] == "LOW")

    return {
        "metadata": {
            "generated": date.today().isoformat(),
            "job": "Job 4 of 7 — Interface Boundary Analysis",
            "total_boundaries": len(boundaries),
            "high_risk_boundaries": high_risk,
            "medium_risk_boundaries": medium_risk,
            "low_risk_boundaries": low_risk,
            "total_cross_module_edges": 308,
            "total_unique_module_pairs": 58,
            "core_module_pairs_documented": len(boundaries),
            "notes": (
                "Entry point (run_*.py) and script boundaries are excluded as they are "
                "consumers only and do not provide interfaces to other modules. "
                "Config boundaries are grouped by consumer module since config.py is "
                "a shared constant provider with 161 incoming edges."
            ),
        },
        "boundaries": boundaries,
    }


def main():
    data = build_contracts()

    # Canonical audit data layout:
    # docs/audit/data/DEPENDENCY_EDGES.json  — Job 2 output
    # docs/audit/data/INTERFACE_CONTRACTS.yaml — Job 3 output
    output_path = Path(__file__).resolve().parent.parent / "docs" / "audit" / "data" / "INTERFACE_CONTRACTS.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    print(f"Wrote {output_path}")
    print(f"  Total boundaries: {data['metadata']['total_boundaries']}")
    print(f"  HIGH risk: {data['metadata']['high_risk_boundaries']}")
    print(f"  MEDIUM risk: {data['metadata']['medium_risk_boundaries']}")
    print(f"  LOW risk: {data['metadata']['low_risk_boundaries']}")


if __name__ == "__main__":
    main()
