"""Tests for SPEC-B06: Verify all api/ module imports resolve without errors.

Every module in the api/ package should be importable without
ModuleNotFoundError, regardless of how the server is started.
"""
import importlib

import pytest


# All api modules that were fixed in SPEC-B06 (absolute → relative imports)
FIXED_MODULES = [
    "quant_engine.api.main",
    "quant_engine.api.routers.iv_surface",
    "quant_engine.api.routers.benchmark",
    "quant_engine.api.routers.dashboard",
    "quant_engine.api.routers.diagnostics",
    "quant_engine.api.services.backtest_service",
    "quant_engine.api.services.model_service",
    "quant_engine.api.services.health_service",
    "quant_engine.api.services.regime_service",
    "quant_engine.api.jobs.train_job",
    "quant_engine.api.jobs.predict_job",
    "quant_engine.api.jobs.backtest_job",
]

# All router modules registered in api/routers/__init__.py
ROUTER_MODULES = [
    "quant_engine.api.routers.jobs",
    "quant_engine.api.routers.system_health",
    "quant_engine.api.routers.dashboard",
    "quant_engine.api.routers.data_explorer",
    "quant_engine.api.routers.model_lab",
    "quant_engine.api.routers.signals",
    "quant_engine.api.routers.backtests",
    "quant_engine.api.routers.benchmark",
    "quant_engine.api.routers.logs",
    "quant_engine.api.routers.autopilot",
    "quant_engine.api.routers.config_mgmt",
    "quant_engine.api.routers.iv_surface",
    "quant_engine.api.routers.regime",
    "quant_engine.api.routers.risk",
    "quant_engine.api.routers.diagnostics",
]


@pytest.mark.parametrize("module_path", FIXED_MODULES)
def test_fixed_module_imports(module_path: str):
    """Each module that had broken imports should now import cleanly."""
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("module_path", ROUTER_MODULES)
def test_router_module_has_router(module_path: str):
    """Each router module should be importable and expose a 'router' attribute."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "router"), f"{module_path} missing 'router' attribute"


def test_no_bare_api_imports():
    """Verify no api/ files still use 'from api.' absolute imports internally."""
    from pathlib import Path

    api_root = Path(__file__).resolve().parent.parent.parent / "api"
    violations = []

    for py_file in api_root.rglob("*.py"):
        with open(py_file) as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                # Check for bare 'from api.' imports (not 'from quant_engine.api.')
                if "from api." in stripped and "quant_engine.api." not in stripped:
                    violations.append(f"{py_file.relative_to(api_root.parent)}:{lineno}: {stripped}")

    assert not violations, (
        "Found bare 'from api.' imports that should use relative imports:\n"
        + "\n".join(violations)
    )


def test_no_bare_models_imports():
    """Verify no api/ files still use 'from models.' absolute imports."""
    from pathlib import Path

    api_root = Path(__file__).resolve().parent.parent.parent / "api"
    violations = []

    for py_file in api_root.rglob("*.py"):
        with open(py_file) as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # 'from models.' without 'quant_engine.models.' prefix
                if "from models." in stripped and "quant_engine.models." not in stripped:
                    violations.append(f"{py_file.relative_to(api_root.parent)}:{lineno}: {stripped}")

    assert not violations, (
        "Found bare 'from models.' imports that should use 'quant_engine.models.':\n"
        + "\n".join(violations)
    )


def test_create_app_no_import_errors():
    """Creating the app should not raise any import errors from router registration."""
    from quant_engine.api.config import ApiSettings
    from quant_engine.api.main import create_app

    app = create_app(ApiSettings(job_db_path=":memory:"))
    assert app is not None
    # Verify iv-surface route is registered (it was broken before)
    paths = {r.path for r in app.routes}
    assert "/api/iv-surface/arb-free-svi" in paths, "iv-surface route not registered"
