"""
Universe Configuration — centralized sector, liquidity, and borrowability metadata.

Loads universe metadata from ``config_data/universe.yaml`` and provides typed
query methods for portfolio risk checks.  Replaces the hardcoded
``SECTOR_MAP`` dict that was previously inlined in ``portfolio_risk.py``.

Spec 07, Task 1.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when universe.yaml is malformed or missing required fields."""


class UniverseConfig:
    """Load, validate, and query universe metadata from YAML config.

    Parameters
    ----------
    path : str or Path, optional
        Path to ``universe.yaml``.  Defaults to ``<project_root>/config_data/universe.yaml``.

    Raises
    ------
    ConfigError
        If the file is missing, unparsable, or fails schema validation.
    """

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = Path(__file__).parent.parent / "config_data" / "universe.yaml"
        self._path = Path(path)
        if not self._path.exists():
            raise ConfigError(
                f"Universe config not found at {self._path}. "
                "Create config_data/universe.yaml or provide a valid path."
            )
        self._raw = self._load_yaml()
        self._validate()
        self._build_lookups()

    # ── Loading and validation ────────────────────────────────────────────

    def _load_yaml(self) -> dict:
        try:
            with open(self._path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse universe.yaml: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError("universe.yaml must be a YAML mapping at the top level.")
        return data

    def _validate(self) -> None:
        """Validate required sections and types."""
        required_sections = ["sectors", "constraint_base", "stress_multipliers"]
        for section in required_sections:
            if section not in self._raw:
                raise ConfigError(f"universe.yaml missing required section: '{section}'")

        # Validate sectors: must be a dict of {sector_name: list_of_tickers}
        sectors = self._raw["sectors"]
        if not isinstance(sectors, dict):
            raise ConfigError("'sectors' must be a mapping of sector_name -> [tickers]")
        for sector_name, tickers in sectors.items():
            if not isinstance(tickers, list):
                raise ConfigError(
                    f"Sector '{sector_name}' must map to a list of tickers, got {type(tickers).__name__}"
                )

        # Validate constraint_base
        cb = self._raw["constraint_base"]
        required_constraints = [
            "sector_cap", "correlation_limit", "gross_exposure",
            "single_name_cap", "annualized_turnover_max",
        ]
        for key in required_constraints:
            if key not in cb:
                raise ConfigError(f"constraint_base missing required key: '{key}'")
            if not isinstance(cb[key], (int, float)):
                raise ConfigError(f"constraint_base.{key} must be numeric, got {type(cb[key]).__name__}")

        # Validate stress_multipliers
        sm = self._raw["stress_multipliers"]
        for regime_key in ("normal", "stress"):
            if regime_key not in sm:
                raise ConfigError(f"stress_multipliers missing required key: '{regime_key}'")

        # Validate liquidity_tiers if present
        if "liquidity_tiers" in self._raw:
            lt = self._raw["liquidity_tiers"]
            if not isinstance(lt, dict):
                raise ConfigError("'liquidity_tiers' must be a mapping")
            for tier_name, tier_def in lt.items():
                if not isinstance(tier_def, dict):
                    raise ConfigError(f"Liquidity tier '{tier_name}' must be a mapping")
                for required_key in ("market_cap_min", "dollar_volume_min"):
                    if required_key not in tier_def:
                        raise ConfigError(
                            f"Liquidity tier '{tier_name}' missing '{required_key}'"
                        )

        # Validate borrowability if present
        if "borrowability" in self._raw:
            borrow = self._raw["borrowability"]
            if not isinstance(borrow, dict):
                raise ConfigError("'borrowability' must be a mapping")
            for key in ("hard_to_borrow", "restricted"):
                if key in borrow and not isinstance(borrow[key], list):
                    raise ConfigError(f"borrowability.{key} must be a list")

    def _build_lookups(self) -> None:
        """Build reverse lookup maps for fast queries."""
        # Ticker -> sector
        self._ticker_to_sector: Dict[str, str] = {}
        for sector_name, tickers in self._raw["sectors"].items():
            for ticker in tickers:
                self._ticker_to_sector[str(ticker).upper()] = str(sector_name)

        # Sector -> tickers
        self._sector_to_tickers: Dict[str, List[str]] = {}
        for sector_name, tickers in self._raw["sectors"].items():
            self._sector_to_tickers[str(sector_name)] = [
                str(t).upper() for t in tickers
            ]

        # Borrowability sets
        borrow = self._raw.get("borrowability", {})
        self._hard_to_borrow = {
            str(t).upper() for t in borrow.get("hard_to_borrow", [])
        }
        self._restricted = {
            str(t).upper() for t in borrow.get("restricted", [])
        }

        # Liquidity tiers (sorted descending by market_cap_min for tier resolution)
        self._liquidity_tiers: List[Tuple[str, float, float]] = []
        for tier_name, tier_def in sorted(
            self._raw.get("liquidity_tiers", {}).items(),
            key=lambda x: x[1].get("market_cap_min", 0),
            reverse=True,
        ):
            self._liquidity_tiers.append((
                tier_name,
                float(tier_def["market_cap_min"]),
                float(tier_def["dollar_volume_min"]),
            ))

    # ── Public query API ─────────────────────────────────────────────────

    def get_sector(self, ticker: str) -> str:
        """Return the sector name for *ticker*, or ``'other'`` if unmapped."""
        return self._ticker_to_sector.get(str(ticker).upper(), "other")

    def get_sector_constituents(self, sector: str) -> List[str]:
        """Return all tickers belonging to *sector*."""
        return list(self._sector_to_tickers.get(sector, []))

    def get_all_sectors(self) -> List[str]:
        """Return all sector names."""
        return list(self._sector_to_tickers.keys())

    def get_liquidity_tier(
        self, market_cap: float, dollar_volume: float,
    ) -> str:
        """Classify into a liquidity tier based on market cap and dollar volume.

        Returns the highest tier where both ``market_cap >= tier.market_cap_min``
        and ``dollar_volume >= tier.dollar_volume_min``.  Falls back to ``'Micro'``
        if no tier qualifies.
        """
        for tier_name, cap_min, vol_min in self._liquidity_tiers:
            if market_cap >= cap_min and dollar_volume >= vol_min:
                return tier_name
        return "Micro"

    def is_hard_to_borrow(self, ticker: str) -> bool:
        """Return True if *ticker* is on the hard-to-borrow list."""
        return str(ticker).upper() in self._hard_to_borrow

    def is_restricted(self, ticker: str) -> bool:
        """Return True if *ticker* is restricted (no short selling)."""
        return str(ticker).upper() in self._restricted

    # ── Constraint accessors ─────────────────────────────────────────────

    @property
    def constraint_base(self) -> dict:
        """Return the base constraint parameters."""
        return dict(self._raw["constraint_base"])

    @property
    def stress_multipliers(self) -> dict:
        """Return the stress multiplier parameters."""
        return dict(self._raw["stress_multipliers"])

    @property
    def factor_limits(self) -> dict:
        """Return factor exposure limit configuration."""
        return dict(self._raw.get("factor_limits", {}))

    @property
    def backoff_policy(self) -> dict:
        """Return backoff policy configuration."""
        return dict(self._raw.get("backoff_policy", {
            "mode": "continuous",
            "thresholds": [0.70, 0.80, 0.90, 0.95],
            "backoff_factors": [0.9, 0.7, 0.5, 0.25],
        }))

    def get_stress_multiplier_set(self, is_stress: bool) -> dict:
        """Return the multiplier set for the given regime type.

        Parameters
        ----------
        is_stress : bool
            If True, returns stress multipliers; otherwise normal multipliers.
        """
        key = "stress" if is_stress else "normal"
        return dict(self._raw["stress_multipliers"].get(key, {}))

    def get_factor_bounds(self, factor: str, is_stress: bool) -> Optional[Tuple[float, float]]:
        """Return (low, high) bounds for a factor, or None if unconstrained.

        Parameters
        ----------
        factor : str
            Factor name (e.g., ``"beta"``, ``"volatility"``).
        is_stress : bool
            If True, returns stress-regime bounds.
        """
        fl = self._raw.get("factor_limits", {})
        if factor not in fl:
            return None
        regime_key = "stress" if is_stress else "normal"
        bounds = fl[factor].get(regime_key)
        if bounds is None:
            return None
        if isinstance(bounds, list) and len(bounds) == 2:
            return (float(bounds[0]), float(bounds[1]))
        return None

    # ── Environment overrides ────────────────────────────────────────────

    def apply_env_overrides(self) -> None:
        """Override constraint values from environment variables.

        Recognized variables:
            - ``CONSTRAINT_STRESS_SECTOR_CAP``
            - ``CONSTRAINT_STRESS_CORRELATION_LIMIT``
            - ``FACTOR_BETA_MIN``
            - ``FACTOR_BETA_MAX``
        """
        env_map = {
            "CONSTRAINT_STRESS_SECTOR_CAP": ("stress_multipliers", "stress", "sector_cap"),
            "CONSTRAINT_STRESS_CORRELATION_LIMIT": ("stress_multipliers", "stress", "correlation_limit"),
        }
        for env_var, path in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    float_val = float(val)
                    section = self._raw[path[0]]
                    section[path[1]][path[2]] = float_val
                    logger.info("Applied env override %s=%s", env_var, val)
                except (ValueError, KeyError) as e:
                    logger.warning("Failed to apply env override %s: %s", env_var, e)

        # Factor bound overrides
        factor_env = {
            "FACTOR_BETA_MIN": ("beta", 0),
            "FACTOR_BETA_MAX": ("beta", 1),
        }
        for env_var, (factor, idx) in factor_env.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    float_val = float(val)
                    fl = self._raw.setdefault("factor_limits", {})
                    fb = fl.setdefault(factor, {"normal": [0.8, 1.2], "stress": [0.9, 1.1]})
                    for regime_key in ("normal", "stress"):
                        if fb.get(regime_key) is not None:
                            fb[regime_key][idx] = float_val
                    logger.info("Applied env override %s=%s", env_var, val)
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning("Failed to apply env override %s: %s", env_var, e)
