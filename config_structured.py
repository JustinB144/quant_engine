"""
Structured configuration for the quant engine using typed dataclasses.

Provides IDE autocomplete, type checking, and organized namespacing
over the flat config.py module. Each subsystem gets its own dataclass.

Usage:
    from config_structured import SystemConfig
    cfg = SystemConfig()
    cfg.regime.n_states        # IDE knows the type and offers autocomplete
    cfg.kelly.max_portfolio_dd  # Clearly scoped to Kelly subsystem
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """Data loading and caching configuration."""

    cache_dir: Path = Path("data/cache")
    wrds_enabled: bool = True
    optionmetrics_enabled: bool = True
    kalshi_enabled: bool = False
    default_universe_source: str = "wrds"
    lookback_years: int = 15
    min_bars: int = 500
    cache_max_staleness_days: int = 21
    cache_trusted_sources: List[str] = field(
        default_factory=lambda: ["wrds", "wrds_delisting", "ibkr"]
    )
    max_missing_bar_fraction: float = 0.05
    max_zero_volume_fraction: float = 0.25
    max_abs_daily_return: float = 0.40


@dataclass
class RegimeConfig:
    """Regime detection configuration."""

    model_type: str = "hmm"  # "hmm", "rule", "jump", or "ensemble" via ensemble_enabled
    n_states: int = 4
    hmm_max_iter: int = 60
    hmm_stickiness: float = 0.92
    min_duration: int = 3
    hmm_prior_weight: float = 0.3
    hmm_covariance_type: str = "full"
    hmm_auto_select_states: bool = True
    hmm_min_states: int = 2
    hmm_max_states: int = 6
    jump_model_enabled: bool = True
    jump_penalty: float = 0.02
    expected_changes_per_year: float = 4.0
    ensemble_enabled: bool = True
    ensemble_consensus_threshold: int = 2
    risk_multiplier: Dict[int, float] = field(
        default_factory=lambda: {0: 1.00, 1: 0.85, 2: 0.95, 3: 0.60}
    )
    stop_multiplier: Dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 0.8, 2: 1.2, 3: 1.5}
    )


@dataclass
class ModelConfig:
    """Model training configuration."""

    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_leaf": 30,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "max_features": "sqrt",
        }
    )
    max_features_selected: int = 30
    max_is_oos_gap: float = 0.05
    cv_folds: int = 5
    holdout_fraction: float = 0.15
    ensemble_diversify: bool = True
    forward_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    max_model_versions: int = 5
    feature_mode: str = "core"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    transaction_cost_bps: float = 20.0
    entry_threshold: float = 0.005
    confidence_threshold: float = 0.6
    max_positions: int = 20
    position_size_pct: float = 0.05
    assumed_capital_usd: float = 1_000_000.0
    max_portfolio_vol: float = 0.30
    max_annualized_turnover: float = 500.0
    max_sector_exposure: float = 0.10
    wf_max_train_dates: Optional[int] = 1260


@dataclass
class KellyConfig:
    """Kelly criterion position sizing configuration."""

    fraction: float = 0.50
    max_portfolio_dd: float = 0.20
    portfolio_blend: float = 0.30
    bayesian_alpha: float = 2.0
    bayesian_beta: float = 2.0
    regime_conditional: bool = True


@dataclass
class DrawdownConfig:
    """Drawdown management configuration."""

    warning_threshold: float = -0.05
    caution_threshold: float = -0.10
    critical_threshold: float = -0.15
    daily_loss_limit: float = -0.03
    weekly_loss_limit: float = -0.05
    recovery_days: int = 10
    size_mult_warning: float = 0.50
    size_mult_caution: float = 0.25


@dataclass
class StopLossConfig:
    """Stop-loss configuration."""

    hard_stop_pct: float = -0.08
    atr_stop_multiplier: float = 2.0
    trailing_atr_multiplier: float = 1.5
    trailing_activation_pct: float = 0.02
    max_holding_days: int = 30


@dataclass
class ValidationConfig:
    """Statistical validation configuration."""

    cpcv_partitions: int = 8
    cpcv_test_partitions: int = 4
    spa_bootstraps: int = 400
    ic_rolling_window: int = 60


@dataclass
class PromotionConfig:
    """Strategy promotion gate thresholds."""

    min_trades: int = 80
    min_win_rate: float = 0.50
    min_sharpe: float = 0.75
    min_profit_factor: float = 1.10
    max_drawdown: float = -0.20
    min_annual_return: float = 0.05
    max_active_strategies: int = 5
    require_advanced_contract: bool = True
    max_dsr_pvalue: float = 0.05
    max_pbo: float = 0.50
    require_statistical_tests: bool = True
    require_cpcv: bool = True
    require_spa: bool = False
    min_wf_oos_corr: float = 0.01
    min_wf_positive_fold_fraction: float = 0.60
    max_wf_is_oos_gap: float = 0.20
    min_regime_positive_fraction: float = 0.50


@dataclass
class HealthConfig:
    """Health monitoring thresholds."""

    min_ic_threshold: float = 0.01
    signal_decay_threshold: float = 0.5
    max_correlation_threshold: float = 0.70
    execution_quality_threshold: float = 2.0
    tail_ratio_threshold: float = 1.0
    cvar_threshold: float = -0.05
    ir_threshold: float = 0.5


@dataclass
class PaperTradingConfig:
    """Paper trading configuration."""

    initial_capital: float = 1_000_000.0
    max_total_positions: int = 30
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.50
    kelly_lookback_trades: int = 200
    kelly_min_size_multiplier: float = 0.25
    kelly_max_size_multiplier: float = 1.50


@dataclass
class ExecutionConfig:
    """Trade execution cost modeling."""

    spread_bps: float = 3.0
    max_participation: float = 0.02
    impact_coeff_bps: float = 25.0
    min_fill_ratio: float = 0.20
    dynamic_costs: bool = True
    almgren_chriss_enabled: bool = True
    almgren_chriss_adv_threshold: float = 0.05


@dataclass
class SystemConfig:
    """Top-level system configuration aggregating all subsystems.

    Provides a single entry point with IDE autocomplete for all config
    domains. Each subsystem is a typed dataclass.
    """

    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
