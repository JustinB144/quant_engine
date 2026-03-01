"""
Structured configuration for the quant engine using typed dataclasses.

This is the AUTHORITATIVE source of truth for all configuration values.
``config.py`` imports from here for backward compatibility.

Provides IDE autocomplete, type checking, and organized namespacing.
Each subsystem gets its own dataclass.

Usage:
    from config_structured import get_config
    cfg = get_config()
    cfg.regime.n_states        # IDE knows the type and offers autocomplete
    cfg.kelly.max_portfolio_dd  # Clearly scoped to Kelly subsystem
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Execution Contract Enums ──────────────────────────────────────────


class ReturnType(Enum):
    """Return computation method."""
    LOG = "log"
    SIMPLE = "simple"


class PriceType(Enum):
    """Price type used for entry/exit baseline."""
    CLOSE = "close"
    OPEN = "open"


class EntryType(Enum):
    """Entry price execution model."""
    NEXT_BAR_OPEN = "next_bar_open"
    MARKET_ON_OPEN = "market_on_open"
    LIMIT_10BP = "limit_10bp"


@dataclass
class PreconditionsConfig:
    """Locked-down execution contract for the Truth Layer.

    Validates that core execution assumptions are explicit and sensible
    before any modeling or backtesting begins.
    """
    ret_type: ReturnType = ReturnType.LOG
    label_h: int = 5
    px_type: PriceType = PriceType.CLOSE
    entry_price_type: EntryType = EntryType.NEXT_BAR_OPEN

    def __post_init__(self):
        # Coerce string values to enums for backward compatibility
        if isinstance(self.ret_type, str):
            self.ret_type = ReturnType(self.ret_type)
        if isinstance(self.px_type, str):
            self.px_type = PriceType(self.px_type)
        if isinstance(self.entry_price_type, str):
            self.entry_price_type = EntryType(self.entry_price_type)

        if not isinstance(self.label_h, int) or self.label_h < 1:
            raise ValueError(f"LABEL_H must be a positive integer, got {self.label_h}")
        if self.label_h > 60:
            raise ValueError(
                f"LABEL_H={self.label_h} exceeds 60 trading days; "
                "this is unrealistic for intraday/daily prediction — check config"
            )


# ── Cost Stress Config ───────────────────────────────────────────────


@dataclass
class CostStressConfig:
    """Configuration for transaction cost stress testing."""
    multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    enabled: bool = False


@dataclass
class DataConfig:
    """Data loading and caching configuration."""

    cache_dir: Path = Path("data/cache")
    cache_alpaca_dir: Path = Path("data/cache_alpaca")
    wrds_enabled: bool = True
    optionmetrics_enabled: bool = False
    kalshi_enabled: bool = False
    default_universe_source: str = "wrds"
    lookback_years: int = 15
    min_bars: int = 500
    cache_max_staleness_days: int = 21
    cache_trusted_sources: List[str] = field(
        default_factory=lambda: ["wrds", "wrds_delisting", "ibkr", "wrds_taq"]
    )
    max_missing_bar_fraction: float = 0.05
    max_zero_volume_fraction: float = 0.25
    max_abs_daily_return: float = 0.40


MAX_HMM_STATES = 6  # Upper bound for HMM auto-select state count


@dataclass
class RegimeConfig:
    """Regime detection configuration."""

    model_type: str = "jump"  # "hmm", "rule", "jump", or "ensemble" via ensemble_enabled
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
    trade_policy: Dict[int, Dict[str, Any]] = field(
        default_factory=lambda: {
            0: {"enabled": True,  "min_confidence": 0.0},
            1: {"enabled": True,  "min_confidence": 0.0},
            2: {"enabled": False, "min_confidence": 0.70},
            3: {"enabled": True,  "min_confidence": 0.60},
        }
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
    structural_weight_enabled: bool = True
    structural_weight_changepoint_penalty: float = 0.5
    structural_weight_jump_penalty: float = 0.5
    structural_weight_stress_penalty: float = 0.3


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
    spa_bootstraps: int = 1000
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
    max_pbo: float = 0.45
    require_statistical_tests: bool = True
    require_cpcv: bool = True
    require_spa: bool = False
    min_wf_oos_corr: float = 0.01
    min_wf_positive_fold_fraction: float = 0.60
    max_wf_is_oos_gap: float = 0.20
    min_regime_positive_fraction: float = 0.50
    max_stress_drawdown: float = 0.15
    min_stress_sharpe: float = -0.50
    max_transition_drawdown: float = 0.10
    stress_regimes: List[int] = field(default_factory=lambda: [2, 3])


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

    # Ensemble disagreement tracking (SPEC-H02)
    ensemble_disagreement_lookback: int = 20
    ensemble_disagreement_warn_threshold: float = 0.015
    ensemble_disagreement_critical_threshold: float = 0.03

    # Execution quality monitoring (SPEC-H03)
    exec_quality_lookback: int = 50
    exec_quality_warn_surprise_bps: float = 2.0
    exec_quality_critical_surprise_bps: float = 5.0


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
    almgren_chriss_enabled: bool = False
    almgren_chriss_adv_threshold: float = 0.05


@dataclass
class SystemConfig:
    """Top-level system configuration aggregating all subsystems.

    Provides a single entry point with IDE autocomplete for all config
    domains. Each subsystem is a typed dataclass.
    """

    preconditions: PreconditionsConfig = field(default_factory=PreconditionsConfig)
    cost_stress: CostStressConfig = field(default_factory=CostStressConfig)
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


# ── Module-level singleton ──────────────────────────────────────────

_CONFIG: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Return the singleton SystemConfig instance.

    On first call, instantiates the default SystemConfig. Subsequent
    calls return the same instance so all callers share one source of
    truth.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = SystemConfig()
    return _CONFIG
