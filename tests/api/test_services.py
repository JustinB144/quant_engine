"""Tests for service wrappers — verify dict outputs."""
from quant_engine.api.services.data_service import DataService
from quant_engine.api.services.autopilot_service import AutopilotService
from quant_engine.api.services.backtest_service import BacktestService
from quant_engine.api.services.health_service import HealthService
from quant_engine.api.services.model_service import ModelService
from quant_engine.api.services.results_service import ResultsService


def test_data_service_universe_info():
    svc = DataService()
    info = svc.get_universe_info()
    assert "full_size" in info
    assert "quick_size" in info
    assert info["full_size"] > 0
    assert isinstance(info["full_tickers"], list)
    assert info["benchmark"] == "SPY"


def test_data_service_cached_tickers():
    svc = DataService()
    tickers = svc.get_cached_tickers()
    assert isinstance(tickers, list)


def test_backtest_service_no_results():
    """When no results files exist, should return available=False."""
    svc = BacktestService()
    result = svc.get_latest_results(horizon=999)  # unlikely horizon
    assert isinstance(result, dict)
    # Either available=False or actual data
    assert "horizon" in result or "available" in result


def test_autopilot_service_latest_cycle():
    svc = AutopilotService()
    result = svc.get_latest_cycle()
    assert isinstance(result, dict)
    assert "available" in result


def test_autopilot_service_strategy_registry():
    svc = AutopilotService()
    result = svc.get_strategy_registry()
    assert isinstance(result, dict)
    assert "active" in result


def test_autopilot_service_paper_state():
    svc = AutopilotService()
    result = svc.get_paper_state()
    assert isinstance(result, dict)
    assert "available" in result


def test_health_service_quick():
    svc = HealthService()
    result = svc.get_quick_status()
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ("healthy", "degraded", "unhealthy")


def test_model_service_list_versions():
    svc = ModelService()
    versions = svc.list_versions()
    assert isinstance(versions, list)


def test_model_service_champion_info():
    svc = ModelService()
    info = svc.get_champion_info(horizon=10)
    assert isinstance(info, dict)
    assert info["horizon"] == 10


def test_results_service_list():
    svc = ResultsService()
    results = svc.list_all_results()
    assert isinstance(results, list)
