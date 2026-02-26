"""SPEC-B05: Verify quant_engine package installation structure.

After ``pip install -e .``, every subpackage must be importable through the
``quant_engine`` namespace.  These tests confirm the pyproject.toml
``package-dir`` mapping and explicit ``packages`` list are correct.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# All packages declared in pyproject.toml [tool.setuptools] packages
# ---------------------------------------------------------------------------
EXPECTED_PACKAGES = [
    "quant_engine",
    "quant_engine.api",
    "quant_engine.api.cache",
    "quant_engine.api.deps",
    "quant_engine.api.jobs",
    "quant_engine.api.routers",
    "quant_engine.api.schemas",
    "quant_engine.api.services",
    "quant_engine.autopilot",
    "quant_engine.backtest",
    "quant_engine.config_data",
    "quant_engine.data",
    "quant_engine.data.providers",
    "quant_engine.evaluation",
    "quant_engine.features",
    "quant_engine.indicators",
    "quant_engine.kalshi",
    "quant_engine.kalshi.tests",
    "quant_engine.models",
    "quant_engine.models.iv",
    "quant_engine.regime",
    "quant_engine.risk",
    "quant_engine.utils",
    "quant_engine.validation",
]

# Top-level modules inside the quant_engine package
EXPECTED_MODULES = [
    "quant_engine.config",
    "quant_engine.config_structured",
    "quant_engine.reproducibility",
]


class TestPackageInstallation:
    """Verify every declared package is importable."""

    @pytest.mark.parametrize("package", EXPECTED_PACKAGES)
    def test_package_importable(self, package: str) -> None:
        mod = importlib.import_module(package)
        assert hasattr(mod, "__file__") or hasattr(mod, "__path__")

    @pytest.mark.parametrize("module", EXPECTED_MODULES)
    def test_module_importable(self, module: str) -> None:
        mod = importlib.import_module(module)
        assert mod.__file__ is not None


class TestCriticalImportPaths:
    """Verify the most-used import paths in the codebase resolve."""

    def test_data_loader(self) -> None:
        from quant_engine.data.loader import load_ohlcv, load_universe
        assert callable(load_ohlcv)
        assert callable(load_universe)

    def test_models_predictor(self) -> None:
        from quant_engine.models.predictor import EnsemblePredictor
        assert EnsemblePredictor is not None

    def test_backtest_engine(self) -> None:
        from quant_engine.backtest.engine import Backtester
        assert Backtester is not None

    def test_regime_detector(self) -> None:
        from quant_engine.regime.detector import RegimeDetector
        assert RegimeDetector is not None

    def test_risk_position_sizer(self) -> None:
        from quant_engine.risk.position_sizer import PositionSizer
        assert PositionSizer is not None

    def test_features_pipeline(self) -> None:
        from quant_engine.features.pipeline import FeaturePipeline
        assert FeaturePipeline is not None

    def test_autopilot_engine(self) -> None:
        from quant_engine.autopilot.engine import AutopilotEngine
        assert AutopilotEngine is not None

    def test_evaluation_engine(self) -> None:
        from quant_engine.evaluation.engine import EvaluationEngine
        assert EvaluationEngine is not None

    def test_api_main(self) -> None:
        from quant_engine.api.main import create_app
        assert callable(create_app)

    def test_config(self) -> None:
        from quant_engine.config import validate_config
        assert callable(validate_config)

    def test_reproducibility(self) -> None:
        from quant_engine.reproducibility import build_run_manifest
        assert callable(build_run_manifest)


class TestEggInfoCorrect:
    """Verify the installed package metadata is correct."""

    def test_top_level_package(self) -> None:
        """After pip install -e ., quant_engine is the sole top-level package."""
        import quant_engine
        assert quant_engine.__file__ is not None
        assert "quant_engine" in quant_engine.__file__

    def test_no_stale_top_level_api(self) -> None:
        """The 'api' package should NOT be importable as a standalone
        top-level package via the installed distribution.  It should
        only be reachable as quant_engine.api.

        Note: 'import api' may still work when running from the project
        root (pytest adds CWD to sys.path), so we check the egg-info
        metadata instead.
        """
        from pathlib import Path

        egg_info = Path(__file__).resolve().parent.parent / "quant_engine.egg-info" / "top_level.txt"
        if egg_info.exists():
            top_levels = egg_info.read_text().strip().splitlines()
            assert "api" not in top_levels, (
                "The 'api' package should not be a standalone top-level in egg-info; "
                "it should only exist under quant_engine.api"
            )
            assert "models" not in top_levels, (
                "The 'models' package should not be a standalone top-level in egg-info; "
                "it should only exist under quant_engine.models"
            )
            assert "quant_engine" in top_levels

    def test_subprocess_import(self) -> None:
        """Verify import works in a clean subprocess (no CWD pollution)."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import quant_engine; from quant_engine.data.loader import load_ohlcv; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed in subprocess: {result.stderr}"
        assert "OK" in result.stdout
