# Source API Reference (Python)

Source-derived Python module inventory for the current repository tree. This file is generated from Python AST parsing and filesystem scanning.

## FastAPI App And Router Registration

- App factory: `api/main.py` (`create_app`, `run_server`)
- Lazy router loader: `api/routers/__init__.py` (`all_routers`)
- Mounted router modules (from `api/routers/__init__.py`):
  - `quant_engine.api.routers.jobs`
  - `quant_engine.api.routers.system_health`
  - `quant_engine.api.routers.dashboard`
  - `quant_engine.api.routers.data_explorer`
  - `quant_engine.api.routers.model_lab`
  - `quant_engine.api.routers.signals`
  - `quant_engine.api.routers.backtests`
  - `quant_engine.api.routers.benchmark`
  - `quant_engine.api.routers.logs`
  - `quant_engine.api.routers.autopilot`
  - `quant_engine.api.routers.config_mgmt`
  - `quant_engine.api.routers.iv_surface`

## Package `(root)`

### `__init__.py`

- LOC: 6
- Module intent: Quant Engine - Continuous Feature ML Trading System
- Top-level functions: none
- Classes: none

### `config.py`

- LOC: 464
- Module intent: Central configuration for the quant engine.
- Imports (2): `from pathlib import Path`, `from typing import Dict`
- Top-level functions:
  - `validate_config()` (line 387): Check config for common misconfigurations.
- Classes: none

### `config_structured.py`

- LOC: 235
- Module intent: Structured configuration for the quant engine using typed dataclasses.
- Imports (4): `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`
- Top-level functions: none
- Classes:
  - `DataConfig` (line 21): Data loading and caching configuration.
  - `RegimeConfig` (line 41): Regime detection configuration.
  - `ModelConfig` (line 68): Model training configuration.
  - `BacktestConfig` (line 92): Backtesting configuration.
  - `KellyConfig` (line 108): Kelly criterion position sizing configuration.
  - `DrawdownConfig` (line 120): Drawdown management configuration.
  - `StopLossConfig` (line 134): Stop-loss configuration.
  - `ValidationConfig` (line 145): Statistical validation configuration.
  - `PromotionConfig` (line 155): Strategy promotion gate thresholds.
  - `HealthConfig` (line 178): Health monitoring thresholds.
  - `PaperTradingConfig` (line 191): Paper trading configuration.
  - `ExecutionConfig` (line 204): Trade execution cost modeling.
  - `SystemConfig` (line 217): Top-level system configuration aggregating all subsystems.

### `reproducibility.py`

- LOC: 333
- Module intent: Reproducibility locks for run manifests.
- Imports (8): `from __future__ import annotations`, `import hashlib`, `import json`, `import subprocess`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`, `import pandas as pd`
- Top-level functions:
  - `_get_git_commit()` (line 22): Return the current git commit hash, or 'unknown' if not in a repo.
  - `_dataframe_checksum(df)` (line 36): Compute a lightweight checksum of a DataFrame's shape and sample.
  - `build_run_manifest(run_type, config_snapshot, datasets, mapping_version, extra)` (line 47): Build a reproducibility manifest for a pipeline run.
  - `write_run_manifest(manifest, output_dir, filename)` (line 99): Write manifest to JSON file. Returns the output path.
  - `verify_manifest(manifest_path, config_snapshot)` (line 117): Verify current environment matches a stored manifest.
  - `replay_manifest(manifest_path, output_dir)` (line 234): Re-run a historical cycle and compare to stored results.
- Classes: none

### `run_autopilot.py`

- LOC: 89
- Module intent: Run one full autopilot cycle:
- Imports (6): `import argparse`, `import sys`, `import time`, `from pathlib import Path`, `from quant_engine.autopilot.engine import AutopilotEngine`, `from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, AUTOPILOT_FEATURE_MODE`
- Top-level functions:
  - `main()` (line 23): Run the command-line entry point.
- Classes: none

### `run_backtest.py`

- LOC: 428
- Module intent: Backtest the trained model on historical data.
- Imports (16): `import argparse`, `import json`, `import logging`, `import sys`, `import time`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, RESULTS_DIR, REGIME_NAMES, ENTRY_THRESHOLD, CPCV_PARTITIONS, CPCV_TEST_PARTITIONS, SPA_BOOTSTRAPS, SURVIVORSHIP_UNIVERSE_NAME, FEATURE_MODE_DEFAULT, WF_MAX_TRAIN_DATES`, `from quant_engine.data.loader import load_universe, load_survivorship_universe`, `from quant_engine.data.survivorship import filter_panel_by_point_in_time_universe`, `from quant_engine.features.pipeline import FeaturePipeline` (+4 more)
- Top-level functions:
  - `main()` (line 50): Run the command-line entry point.
- Classes: none

### `run_kalshi_event_pipeline.py`

- LOC: 227
- Module intent: Run the integrated Kalshi event-time pipeline inside quant_engine.
- Imports (10): `import argparse`, `import json`, `from pathlib import Path`, `import pandas as pd`, `from quant_engine.config import KALSHI_DB_PATH, KALSHI_DISTANCE_LAGS, KALSHI_DISTRIBUTION_FREQ, KALSHI_ENABLED, KALSHI_FAR_EVENT_MINUTES, KALSHI_FAR_EVENT_STALE_MINUTES, KALSHI_NEAR_EVENT_MINUTES, KALSHI_NEAR_EVENT_STALE_MINUTES, KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER, KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD, KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD, KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER, KALSHI_STALE_MARKET_TYPE_MULTIPLIERS, KALSHI_SNAPSHOT_HORIZONS, KALSHI_STALE_AFTER_MINUTES, KALSHI_TAIL_THRESHOLDS, RESULTS_DIR`, `from quant_engine.kalshi.distribution import DistributionConfig`, `from quant_engine.kalshi.events import EventFeatureConfig, build_event_labels`, `from quant_engine.kalshi.pipeline import KalshiPipeline`, `from quant_engine.kalshi.promotion import EventPromotionConfig`, `from quant_engine.kalshi.walkforward import EventWalkForwardConfig`
- Top-level functions:
  - `_read_df(path)` (line 37): Read a CSV or Parquet file into a DataFrame based on the file extension.
  - `main()` (line 45): Run Kalshi ingestion, distribution building, event features, walk-forward evaluation, and reporting tasks.
- Classes: none

### `run_predict.py`

- LOC: 202
- Module intent: Generate predictions using trained ensemble model.
- Imports (12): `import argparse`, `import json`, `import sys`, `import time`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD, RESULTS_DIR, REGIME_NAMES, FEATURE_MODE_DEFAULT`, `from quant_engine.data.loader import load_universe`, `from quant_engine.features.pipeline import FeaturePipeline`, `from quant_engine.regime.detector import RegimeDetector`, `from quant_engine.models.predictor import EnsemblePredictor`
- Top-level functions:
  - `main()` (line 33): Load a trained model version and generate prediction outputs for the configured universe.
- Classes: none

### `run_rehydrate_cache_metadata.py`

- LOC: 101
- Module intent: Backfill cache metadata sidecars for existing OHLCV cache files.
- Imports (5): `import argparse`, `import sys`, `from pathlib import Path`, `from quant_engine.config import DATA_CACHE_DIR, FRAMEWORK_DIR`, `from quant_engine.data.local_cache import rehydrate_cache_metadata`
- Top-level functions:
  - `_parse_root_source(items)` (line 21): Parse repeated --root-source values of the form "<path>=<source>" into a mapping.
  - `main()` (line 36): Backfill cache metadata sidecars for existing cached OHLCV files and print a summary report.
- Classes: none

### `run_retrain.py`

- LOC: 290
- Module intent: Retrain the quant engine model — checks triggers and retrains if needed.
- Imports (14): `import argparse`, `import sys`, `import time`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, FORWARD_HORIZONS, FEATURE_MODE_DEFAULT, RETRAIN_REGIME_CHANGE_DAYS, RESULTS_DIR`, `from quant_engine.data.loader import load_universe, load_survivorship_universe`, `from quant_engine.features.pipeline import FeaturePipeline`, `from quant_engine.regime.detector import RegimeDetector`, `from quant_engine.models.governance import ModelGovernance`, `from quant_engine.models.trainer import ModelTrainer` (+2 more)
- Top-level functions:
  - `_check_regime_change_trigger(predictions_df, trained_regime, days_threshold)` (line 36): Check whether the market regime has changed for a sustained period.
  - `main()` (line 67): Evaluate retraining conditions and run controlled retraining/promotion updates when triggered.
- Classes: none

### `run_server.py`

- LOC: 80
- Module intent: Combined API + frontend static serving entry point.
- Imports (5): `from __future__ import annotations`, `import argparse`, `import logging`, `import sys`, `from pathlib import Path`
- Top-level functions:
  - `main()` (line 25): No module docstring.
- Classes: none

### `run_train.py`

- LOC: 195
- Module intent: Train the regime-conditional ensemble model.
- Imports (13): `import argparse`, `import sys`, `import time`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, FORWARD_HORIZONS, LOOKBACK_YEARS, FEATURE_MODE_DEFAULT`, `from quant_engine.data.loader import load_universe, load_survivorship_universe`, `from quant_engine.features.pipeline import FeaturePipeline`, `from quant_engine.regime.detector import RegimeDetector`, `from quant_engine.models.governance import ModelGovernance`, `from quant_engine.models.trainer import ModelTrainer` (+1 more)
- Top-level functions:
  - `main()` (line 35): Train a model for the requested horizon/workflow settings and persist versioned artifacts.
- Classes: none

### `run_wrds_daily_refresh.py`

- LOC: 348
- Module intent: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
- Imports (8): `import argparse`, `import sys`, `import time`, `from datetime import datetime`, `from pathlib import Path`, `import pandas as pd`, `from quant_engine.config import DATA_CACHE_DIR, UNIVERSE_FULL, BENCHMARK`, `from quant_engine.data.local_cache import list_cached_tickers, save_ohlcv`
- Top-level functions:
  - `_build_ticker_list(tickers_arg)` (line 30): Build the full ticker list from cached + UNIVERSE_FULL + BENCHMARK.
  - `_verify_file(path)` (line 41): Verify OHLCV quality for a single parquet file. Returns dict of results.
  - `_verify_all(cache_dir)` (line 93): Run verification on all _1d.parquet files in cache.
  - `_cleanup_old_daily(cache_dir, downloaded_tickers)` (line 148): Remove old {TICKER}_daily_{dates}.parquet and .meta.json files.
  - `main()` (line 169): Run the WRDS daily refresh workflow and emit a summary of refreshed datasets and outputs.
- Classes: none

## Package `api`

### `api/__init__.py`

- LOC: 1
- Module intent: FastAPI backend for the quant engine.
- Top-level functions: none
- Classes: none

### `api/ab_testing.py`

- LOC: 214
- Module intent: A/B testing framework for strategy evaluation.
- Imports (9): `from __future__ import annotations`, `import json`, `import logging`, `import uuid`, `from dataclasses import asdict, dataclass, field`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `ABVariant` (line 24): One arm of an A/B test.
    - Methods (3): `n_trades`, `mean_return`, `sharpe`
  - `ABTest` (line 54): An A/B test comparing two strategy variants.
    - Methods (5): `__post_init__`, `record_trade`, `assign_variant`, `get_results`, `to_dict`
  - `ABTestRegistry` (line 156): Manages A/B tests with JSON persistence.
    - Methods (6): `__init__`, `create_test`, `get_test`, `list_tests`, `complete_test`, `_save`

### `api/cache/__init__.py`

- LOC: 4
- Module intent: TTL cache with event-driven invalidation.
- Imports (1): `from .manager import CacheManager`
- Top-level functions: none
- Classes: none

### `api/cache/invalidation.py`

- LOC: 32
- Module intent: Event-driven cache invalidation helpers.
- Imports (2): `from __future__ import annotations`, `from .manager import CacheManager`
- Top-level functions:
  - `invalidate_on_train(cache)` (line 7): Clear caches affected by a new model training run.
  - `invalidate_on_backtest(cache)` (line 15): Clear caches affected by a new backtest run.
  - `invalidate_on_data_refresh(cache)` (line 21): Clear caches affected by fresh market data.
  - `invalidate_on_config_change(cache)` (line 30): Clear all caches when runtime config is patched.
- Classes: none

### `api/cache/manager.py`

- LOC: 62
- Module intent: In-memory TTL cache manager.
- Imports (4): `from __future__ import annotations`, `import fnmatch`, `import time`, `from typing import Any, Dict, Optional`
- Top-level functions: none
- Classes:
  - `CacheManager` (line 21): Simple in-memory dict cache with per-key TTL.
    - Methods (6): `__init__`, `get`, `set`, `invalidate`, `invalidate_pattern`, `invalidate_all`

### `api/config.py`

- LOC: 75
- Module intent: Runtime-adjustable configuration for the API layer.
- Imports (4): `from __future__ import annotations`, `import logging`, `from typing import Any, Dict, Set`, `from pydantic_settings import BaseSettings`
- Top-level functions: none
- Classes:
  - `ApiSettings` (line 27): Immutable settings loaded from environment / .env file.
  - `RuntimeConfig` (line 39): Thin wrapper around engine ``config.py`` module-level variables.
    - Methods (3): `__init__`, `get_adjustable`, `patch`

### `api/deps/__init__.py`

- LOC: 16
- Module intent: Dependency injection providers.
- Imports (1): `from .providers import get_cache, get_job_runner, get_job_store, get_runtime_config, get_settings`
- Top-level functions: none
- Classes: none

### `api/deps/providers.py`

- LOC: 54
- Module intent: Singleton dependency providers for FastAPI ``Depends()``.
- Imports (3): `from __future__ import annotations`, `from functools import lru_cache`, `from ..config import ApiSettings, RuntimeConfig`
- Top-level functions:
  - `get_settings()` (line 10): No module docstring.
  - `get_runtime_config()` (line 15): No module docstring.
  - `get_job_store()` (line 27): Return the singleton ``JobStore``.
  - `get_job_runner()` (line 37): Return the singleton ``JobRunner``.
  - `get_cache()` (line 47): Return the singleton ``CacheManager``.
- Classes: none

### `api/errors.py`

- LOC: 77
- Module intent: Custom exceptions and FastAPI error handler registration.
- Imports (6): `from __future__ import annotations`, `import logging`, `import traceback`, `from fastapi import FastAPI, Request`, `from fastapi.responses import JSONResponse`, `from .schemas.envelope import ApiResponse`
- Top-level functions:
  - `_make_handler(status_code)` (line 49): Create a handler that wraps an exception in ApiResponse.
  - `register_error_handlers(app)` (line 59): Register custom exception handlers on the FastAPI app.
- Classes:
  - `DataNotFoundError` (line 18): Requested data or ticker not available.
  - `TrainingFailedError` (line 22): Model training could not complete.
  - `JobNotFoundError` (line 26): Requested job ID does not exist.
  - `ConfigValidationError` (line 30): Runtime config patch contains invalid keys or values.
  - `ServiceUnavailableError` (line 34): An engine dependency is not ready (e.g. WRDS offline).

### `api/jobs/__init__.py`

- LOC: 6
- Module intent: SQLite-backed job queue for long-running compute.
- Imports (3): `from .models import JobRecord, JobStatus`, `from .store import JobStore`, `from .runner import JobRunner`
- Top-level functions: none
- Classes: none

### `api/jobs/autopilot_job.py`

- LOC: 31
- Module intent: Autopilot job executor.
- Imports (2): `from __future__ import annotations`, `from typing import Any, Callable, Dict, Optional`
- Top-level functions:
  - `execute_autopilot_job(params, progress_callback)` (line 7): Run a single autopilot cycle and return results dict.
- Classes: none

### `api/jobs/backtest_job.py`

- LOC: 29
- Module intent: Backtest job executor.
- Imports (2): `from __future__ import annotations`, `from typing import Any, Callable, Dict, Optional`
- Top-level functions:
  - `execute_backtest_job(params, progress_callback)` (line 7): Run a full backtest pipeline and return results dict.
- Classes: none

### `api/jobs/models.py`

- LOC: 32
- Module intent: Job data models.
- Imports (5): `from __future__ import annotations`, `import enum`, `from datetime import datetime, timezone`, `from typing import Any, Dict, Optional`, `from pydantic import BaseModel, Field`
- Top-level functions: none
- Classes:
  - `JobStatus` (line 11): No module docstring.
  - `JobRecord` (line 19): Persistent representation of a compute job.

### `api/jobs/predict_job.py`

- LOC: 28
- Module intent: Predict job executor.
- Imports (2): `from __future__ import annotations`, `from typing import Any, Callable, Dict, Optional`
- Top-level functions:
  - `execute_predict_job(params, progress_callback)` (line 7): Run prediction pipeline and return results dict.
- Classes: none

### `api/jobs/runner.py`

- LOC: 115
- Module intent: Async job runner with concurrency control and SSE event streaming.
- Imports (8): `from __future__ import annotations`, `import asyncio`, `import logging`, `import traceback`, `from datetime import datetime, timezone`, `from typing import Any, AsyncGenerator, Callable, Dict, Optional`, `from .models import JobStatus`, `from .store import JobStore`
- Top-level functions: none
- Classes:
  - `JobRunner` (line 16): Executes job functions in background threads with bounded concurrency.
    - Methods (7): `__init__`, `submit`, `_run`, `_on_progress`, `cancel`, `subscribe_events`, `_emit`

### `api/jobs/store.py`

- LOC: 145
- Module intent: SQLite-backed persistence for job records.
- Imports (6): `from __future__ import annotations`, `import json`, `import uuid`, `from typing import Any, Dict, List, Optional`, `import aiosqlite`, `from .models import JobRecord, JobStatus`
- Top-level functions: none
- Classes:
  - `JobStore` (line 13): Async SQLite store for job lifecycle tracking.
    - Methods (10): `__init__`, `initialize`, `close`, `create_job`, `get_job`, `list_jobs`, `update_status`, `update_progress`, `cancel_job`, `_row_to_record`

### `api/jobs/train_job.py`

- LOC: 30
- Module intent: Train job executor.
- Imports (2): `from __future__ import annotations`, `from typing import Any, Callable, Dict, Optional`
- Top-level functions:
  - `execute_train_job(params, progress_callback)` (line 7): Run a full train pipeline and return results dict.
- Classes: none

### `api/main.py`

- LOC: 148
- Module intent: FastAPI application factory and server entry point.
- Imports (9): `from __future__ import annotations`, `import asyncio`, `import logging`, `from contextlib import asynccontextmanager`, `from fastapi import FastAPI`, `from fastapi.middleware.cors import CORSMiddleware`, `from .config import ApiSettings`, `from .deps.providers import get_cache, get_job_store, get_settings`, `from .errors import register_error_handlers`
- Top-level functions:
  - `async _retrain_monitor_loop()` (line 21): Background task that periodically checks retrain triggers.
  - `async _lifespan(app)` (line 46): Startup / shutdown lifecycle.
  - `create_app(settings)` (line 101): Build and return the FastAPI application.
  - `run_server()` (line 138): CLI entry point: ``python -m quant_engine.api.main``.
- Classes: none

### `api/orchestrator.py`

- LOC: 372
- Module intent: Unified pipeline orchestrator — data -> features -> regimes -> compute.
- Imports (6): `from __future__ import annotations`, `import logging`, `from dataclasses import dataclass, field`, `from typing import Any, Dict, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `PipelineState` (line 19): Intermediate state passed between orchestrator stages.
  - `PipelineOrchestrator` (line 29): Chains engine modules into a reproducible pipeline.
    - Methods (4): `load_and_prepare`, `train`, `predict`, `backtest`

### `api/routers/__init__.py`

- LOC: 39
- Module intent: Route modules — imported lazily by the app factory.
- Imports (4): `from __future__ import annotations`, `import logging`, `from typing import List`, `from fastapi import APIRouter`
- Top-level functions:
  - `all_routers()` (line 28): Import and return every available router, skipping broken ones.
- Classes: none

### `api/routers/autopilot.py`

- LOC: 83
- Module intent: Autopilot endpoints — cycle reports, strategies, paper state, run-cycle.
- Imports (10): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..deps.providers import get_job_runner, get_job_store`, `from ..jobs.runner import JobRunner`, `from ..jobs.store import JobStore`, `from ..schemas.compute import AutopilotRequest`, `from ..schemas.envelope import ApiResponse`, `from ..services.autopilot_service import AutopilotService`
- Route decorators detected:
  - `GET` line 44: `"/latest-cycle"`
  - `GET` line 54: `"/strategies"`
  - `GET` line 63: `"/paper-state"`
  - `POST` line 73: `"/run-cycle"`
- Top-level functions:
  - `_get_autopilot_meta()` (line 19): Collect transparency metadata for autopilot responses.
  - `async latest_cycle()` (line 45): No module docstring.
  - `async strategies()` (line 55): No module docstring.
  - `async paper_state()` (line 64): No module docstring.
  - `async run_cycle(req, store, runner)` (line 74): No module docstring.
- Classes: none

### `api/routers/backtests.py`

- LOC: 89
- Module intent: Backtest result + compute endpoints.
- Imports (12): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.invalidation import invalidate_on_backtest`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache, get_job_runner, get_job_store`, `from ..jobs.runner import JobRunner`, `from ..jobs.store import JobStore`, `from ..schemas.compute import BacktestRequest`, `from ..schemas.envelope import ApiResponse`, `from ..services.backtest_service import BacktestService`
- Route decorators detected:
  - `GET` line 33: `"/latest"`
  - `GET` line 51: `"/latest/trades"`
  - `GET` line 60: `"/latest/equity-curve"`
  - `POST` line 77: `"/run"`
- Top-level functions:
  - `_extract_backtest_meta(data)` (line 21): Extract transparency fields from backtest data for ResponseMeta.
  - `async latest_backtest(horizon, cache)` (line 34): No module docstring.
  - `async latest_trades(horizon, limit, offset)` (line 52): No module docstring.
  - `async equity_curve(horizon, cache)` (line 61): No module docstring.
  - `async run_backtest(req, store, runner, cache)` (line 78): No module docstring.
- Classes: none

### `api/routers/benchmark.py`

- LOC: 128
- Module intent: Benchmark comparison endpoints.
- Imports (7): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache`, `from ..schemas.envelope import ApiResponse`
- Route decorators detected:
  - `GET` line 93: `"/comparison"`
  - `GET` line 105: `"/equity-curves"`
  - `GET` line 118: `"/rolling-metrics"`
- Top-level functions:
  - `_compute_comparison()` (line 16): Build benchmark vs. strategy comparison (sync).
  - `_compute_equity_curves()` (line 47): Build cumulative equity curves for strategy and benchmark (sync).
  - `_compute_rolling_metrics()` (line 70): Build rolling 60D correlation, alpha, beta, relative strength (sync).
  - `async benchmark_comparison(cache)` (line 94): No module docstring.
  - `async benchmark_equity_curves(cache)` (line 106): Strategy + SPY cumulative return time series for equity comparison chart.
  - `async benchmark_rolling_metrics(cache)` (line 119): Rolling 60D correlation, alpha, beta, relative strength time series.
- Classes: none

### `api/routers/config_mgmt.py`

- LOC: 146
- Module intent: Runtime config management endpoints.
- Imports (9): `from __future__ import annotations`, `from typing import Any, Dict`, `from fastapi import APIRouter, Body, Depends`, `from fastapi.responses import JSONResponse`, `from ..cache.invalidation import invalidate_on_config_change`, `from ..cache.manager import CacheManager`, `from ..config import RuntimeConfig`, `from ..deps.providers import get_cache, get_runtime_config`, `from ..schemas.envelope import ApiResponse`
- Route decorators detected:
  - `GET` line 102: `""`
  - `GET` line 107: `"/validate"`
  - `GET` line 124: `"/status"`
  - `PATCH` line 134: `""`
- Top-level functions:
  - `_annotated(value, status, reason)` (line 18): Build a single config entry with status annotation.
  - `_build_config_status()` (line 26): Build the full annotated config status response.
  - `async get_config(rc)` (line 103): No module docstring.
  - `async validate_config_endpoint()` (line 108): Run config validation and return any issues found.
  - `async get_config_status()` (line 125): Return all config values with active/placeholder/inactive status annotations.
  - `async patch_config(updates, rc, cache)` (line 135): No module docstring.
- Classes: none

### `api/routers/dashboard.py`

- LOC: 170
- Module intent: Dashboard endpoints — KPIs, regime overview, time series analytics.
- Imports (9): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache`, `from ..schemas.envelope import ApiResponse`, `from ..services.backtest_service import BacktestService`, `from ..services.regime_service import RegimeService`
- Route decorators detected:
  - `GET` line 18: `"/summary"`
  - `GET` line 39: `"/regime"`
  - `GET` line 121: `"/returns-distribution"`
  - `GET` line 134: `"/rolling-risk"`
  - `GET` line 147: `"/equity"`
  - `GET` line 160: `"/attribution"`
- Top-level functions:
  - `async dashboard_summary(cache)` (line 19): No module docstring.
  - `async dashboard_regime(cache)` (line 40): No module docstring.
  - `_compute_returns_distribution()` (line 52): Build return histogram data with VaR/CVaR lines (sync).
  - `_compute_rolling_risk()` (line 67): Build rolling vol, Sharpe, drawdown time series (sync).
  - `_compute_equity_with_benchmark()` (line 82): Build equity curve with benchmark overlay (sync).
  - `_compute_attribution()` (line 104): Build factor attribution analysis (sync).
  - `async returns_distribution(cache)` (line 122): Daily return histogram data with VaR/CVaR lines.
  - `async rolling_risk(cache)` (line 135): Rolling volatility, Sharpe, and drawdown time series.
  - `async equity_with_benchmark(cache)` (line 148): Equity curve with benchmark overlay time series.
  - `async attribution(cache)` (line 161): Factor attribution: tech-minus-def and momentum-spread decomposition.
- Classes: none

### `api/routers/data_explorer.py`

- LOC: 56
- Module intent: Data explorer endpoints — universe + per-ticker OHLCV.
- Imports (9): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache`, `from ..errors import DataNotFoundError`, `from ..schemas.envelope import ApiResponse`, `from ..services.data_service import DataService`
- Route decorators detected:
  - `GET` line 18: `"/universe"`
  - `GET` line 31: `"/status"`
  - `GET` line 48: `"/ticker/{ticker}"`
- Top-level functions:
  - `async get_universe(cache)` (line 19): No module docstring.
  - `async get_data_status(cache)` (line 32): Per-ticker cache health: source, freshness, bar counts, timeframes.
  - `async get_ticker(ticker, years)` (line 49): No module docstring.
- Classes: none

### `api/routers/iv_surface.py`

- LOC: 71
- Module intent: IV Surface computation endpoints.
- Imports (5): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter`, `from ..schemas.envelope import ApiResponse`
- Route decorators detected:
  - `GET` line 65: `"/arb-free-svi"`
- Top-level functions:
  - `_compute_arb_free_svi()` (line 14): Build arb-free SVI surface from synthetic market data.
  - `async arb_free_svi_surface()` (line 66): Compute arbitrage-free SVI surface from synthetic market data.
- Classes: none

### `api/routers/jobs.py`

- LOC: 72
- Module intent: Job management endpoints.
- Imports (11): `from __future__ import annotations`, `import json`, `from fastapi import APIRouter, Depends`, `from fastapi.responses import JSONResponse`, `from sse_starlette.sse import EventSourceResponse`, `from ..deps.providers import get_job_runner, get_job_store`, `from ..errors import JobNotFoundError`, `from ..jobs.models import JobRecord`, `from ..jobs.store import JobStore`, `from ..jobs.runner import JobRunner`, `from ..schemas.envelope import ApiResponse`
- Route decorators detected:
  - `GET` line 25: `""`
  - `GET` line 34: `"/{job_id}"`
  - `GET` line 45: `"/{job_id}/events"`
  - `POST` line 62: `"/{job_id}/cancel"`
- Top-level functions:
  - `_not_found(job_id)` (line 20): No module docstring.
  - `async list_jobs(limit, store)` (line 26): No module docstring.
  - `async get_job(job_id, store)` (line 35): No module docstring.
  - `async job_events(job_id, store, runner)` (line 46): No module docstring.
  - `async cancel_job(job_id, store, runner)` (line 63): No module docstring.
- Classes: none

### `api/routers/logs.py`

- LOC: 38
- Module intent: Log retrieval endpoint.
- Imports (5): `from __future__ import annotations`, `import logging`, `from collections import deque`, `from fastapi import APIRouter`, `from ..schemas.envelope import ApiResponse`
- Route decorators detected:
  - `GET` line 35: `""`
- Top-level functions:
  - `async get_logs(last_n)` (line 36): No module docstring.
- Classes:
  - `_BufferHandler` (line 17): Captures log records into the ring buffer.
    - Methods (1): `emit`

### `api/routers/model_lab.py`

- LOC: 90
- Module intent: Model lab endpoints — versions, health, feature importance, train, predict.
- Imports (12): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.invalidation import invalidate_on_train`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache, get_job_runner, get_job_store`, `from ..jobs.runner import JobRunner`, `from ..jobs.store import JobStore`, `from ..schemas.compute import PredictRequest, TrainRequest`, `from ..schemas.envelope import ApiResponse`, `from ..services.model_service import ModelService`
- Route decorators detected:
  - `GET` line 21: `"/versions"`
  - `GET` line 30: `"/health"`
  - `GET` line 43: `"/features/importance"`
  - `GET` line 56: `"/features/correlations"`
  - `POST` line 65: `"/train"`
  - `POST` line 80: `"/predict"`
- Top-level functions:
  - `async list_versions()` (line 22): No module docstring.
  - `async model_health(cache)` (line 31): No module docstring.
  - `async feature_importance(cache)` (line 44): No module docstring.
  - `async feature_correlations()` (line 57): No module docstring.
  - `async train_model(req, store, runner, cache)` (line 66): No module docstring.
  - `async predict_model(req, store, runner)` (line 81): No module docstring.
- Classes: none

### `api/routers/signals.py`

- LOC: 51
- Module intent: Signal / prediction endpoints.
- Imports (8): `from __future__ import annotations`, `import asyncio`, `import time`, `from fastapi import APIRouter, Depends`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache`, `from ..schemas.envelope import ApiResponse`, `from ..services.results_service import ResultsService`
- Route decorators detected:
  - `GET` line 36: `"/latest"`
- Top-level functions:
  - `_get_signal_meta_fields()` (line 17): Collect transparency metadata for signal responses.
  - `async latest_signals(horizon, cache)` (line 37): No module docstring.
- Classes: none

### `api/routers/system_health.py`

- LOC: 111
- Module intent: System health endpoints.
- Imports (11): `from __future__ import annotations`, `import asyncio`, `import json`, `import time`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from fastapi import APIRouter, Depends`, `from ..cache.manager import CacheManager`, `from ..deps.providers import get_cache`, `from ..schemas.envelope import ApiResponse`, `from ..services.health_service import HealthService`
- Route decorators detected:
  - `GET` line 20: `"/api/health"`
  - `GET` line 33: `"/api/health/detailed"`
  - `GET` line 46: `"/api/v1/system/model-age"`
  - `GET` line 81: `"/api/v1/system/data-mode"`
- Top-level functions:
  - `async quick_health(cache)` (line 21): No module docstring.
  - `async detailed_health(cache)` (line 34): No module docstring.
  - `async model_age()` (line 47): Return age of the currently deployed model in days and the version ID.
  - `async data_mode()` (line 82): Return current data source mode (wrds, cache, demo) and any active fallbacks.
- Classes: none

### `api/schemas/__init__.py`

- LOC: 4
- Module intent: Pydantic schemas for API request/response models.
- Imports (1): `from .envelope import ApiResponse, ResponseMeta`
- Top-level functions: none
- Classes: none

### `api/schemas/autopilot.py`

- LOC: 30
- Module intent: Autopilot schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `CycleReport` (line 9): Latest autopilot cycle results.
  - `StrategyInfo` (line 16): Active strategy from registry.
  - `PaperState` (line 27): Paper trading state.

### `api/schemas/backtests.py`

- LOC: 56
- Module intent: Backtest schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `BacktestSummary` (line 9): Backtest run summary.
  - `TradeRecord` (line 26): Single trade from backtest.
  - `EquityCurvePoint` (line 43): Single point on an equity curve.
  - `RegimeBreakdown` (line 50): Per-regime performance stats.

### `api/schemas/compute.py`

- LOC: 56
- Module intent: Request schemas for compute (POST) endpoints.
- Imports (3): `from __future__ import annotations`, `from typing import List, Optional`, `from pydantic import BaseModel, Field`
- Top-level functions: none
- Classes:
  - `TrainRequest` (line 9): Request body for POST /api/models/train.
  - `BacktestRequest` (line 21): Request body for POST /api/backtests/run.
  - `PredictRequest` (line 33): Request body for POST /api/models/predict.
  - `AutopilotRequest` (line 44): Request body for POST /api/autopilot/run-cycle.
  - `JobCreatedResponse` (line 51): Response for job submission endpoints.

### `api/schemas/dashboard.py`

- LOC: 38
- Module intent: Dashboard-related schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `DashboardKPIs` (line 9): Key performance indicators for the dashboard summary.
  - `RegimeInfo` (line 24): Current regime detection results.
  - `EquityPoint` (line 34): Single point on an equity curve.

### `api/schemas/data_explorer.py`

- LOC: 37
- Module intent: Data explorer schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `UniverseInfo` (line 9): Universe configuration summary.
  - `OHLCVBar` (line 19): Single OHLCV bar.
  - `TickerDetail` (line 30): Detailed ticker response with OHLCV bars.

### `api/schemas/envelope.py`

- LOC: 54
- Module intent: Standard API response envelope with provenance metadata.
- Imports (4): `from __future__ import annotations`, `from datetime import datetime, timezone`, `from typing import Any, Generic, List, Optional, TypeVar`, `from pydantic import BaseModel, Field`
- Top-level functions: none
- Classes:
  - `ResponseMeta` (line 12): Provenance metadata attached to every API response.
  - `ApiResponse` (line 29): Generic API response wrapper.
    - Methods (3): `success`, `fail`, `from_cached`

### `api/schemas/model_lab.py`

- LOC: 43
- Module intent: Model lab schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `ModelVersionInfo` (line 9): Summary of a model version.
  - `ModelHealth` (line 27): Model health assessment.
  - `FeatureImportance` (line 39): Feature importance results.

### `api/schemas/signals.py`

- LOC: 26
- Module intent: Signal schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `SignalRow` (line 9): Single prediction signal.
  - `SignalsSummary` (line 20): Signals overview for a horizon.

### `api/schemas/system_health.py`

- LOC: 47
- Module intent: System health schemas.
- Imports (3): `from __future__ import annotations`, `from typing import Any, Dict, List, Optional`, `from pydantic import BaseModel`
- Top-level functions: none
- Classes:
  - `QuickStatus` (line 9): Lightweight health check response.
  - `AlertEvent` (line 17): A single health check item.
  - `SystemHealthDetail` (line 27): Full system health assessment.

### `api/services/__init__.py`

- LOC: 20
- Module intent: Engine wrapper services — sync functions returning plain dicts.
- Imports (8): `from .data_service import DataService`, `from .regime_service import RegimeService`, `from .model_service import ModelService`, `from .backtest_service import BacktestService`, `from .autopilot_service import AutopilotService`, `from .health_service import HealthService`, `from .kalshi_service import KalshiService`, `from .results_service import ResultsService`
- Top-level functions: none
- Classes: none

### `api/services/autopilot_service.py`

- LOC: 62
- Module intent: Wraps autopilot engine and results for API consumption.
- Imports (4): `from __future__ import annotations`, `import json`, `import logging`, `from typing import Any, Dict, List`
- Top-level functions: none
- Classes:
  - `AutopilotService` (line 11): Reads autopilot state and cycle reports.
    - Methods (3): `get_latest_cycle`, `get_strategy_registry`, `get_paper_state`

### `api/services/backtest_service.py`

- LOC: 139
- Module intent: Wraps backtest results for API consumption.
- Imports (6): `from __future__ import annotations`, `import json`, `import logging`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `BacktestService` (line 14): Reads backtest result files from the results/ directory.
    - Methods (6): `get_latest_results`, `get_latest_trades`, `get_equity_curve`, `_compute_model_staleness`, `_get_sizing_method`, `_get_walk_forward_mode`

### `api/services/data_helpers.py`

- LOC: 1034
- Module intent: Data loading and computation functions extracted from dash_ui/data/loaders.py.
- Imports (9): `from __future__ import annotations`, `import json`, `import logging`, `from dataclasses import dataclass, field`, `from datetime import datetime, timedelta`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `load_trades(path)` (line 26): Load and clean backtest trade CSV.
  - `build_portfolio_returns(trades)` (line 43): Build daily portfolio returns from trade-level data.
  - `_read_close_returns(path)` (line 60): Read close returns from a parquet file.
  - `load_benchmark_returns(cache_dir, ref_index)` (line 69): Load benchmark (SPY) returns from parquet cache.
  - `build_equity_curves(strategy_returns, benchmark_returns, max_points)` (line 125): Build aligned cumulative return series for strategy and benchmark.
  - `compute_rolling_metrics(strategy_returns, benchmark_returns, window, max_points)` (line 165): Compute rolling correlation, alpha, beta, and relative strength.
  - `compute_returns_distribution(returns, bins)` (line 251): Compute histogram data and risk lines for a returns series.
  - `compute_rolling_risk(returns, vol_window, sharpe_window, max_points)` (line 284): Compute rolling volatility, Sharpe, and drawdown time series.
  - `compute_attribution(strategy_returns, cache_dir)` (line 324): Compute factor attribution: tech-minus-def and momentum-spread.
  - `_load_proxy_returns(cache_dir, symbols)` (line 400): Try to load daily close returns for any of the given symbol tickers.
  - `compute_risk_metrics(returns)` (line 429): Compute portfolio risk metrics from daily returns.
  - `compute_regime_payload(cache_dir)` (line 463): Run HMM regime detection and return structured results.
  - `compute_model_health(model_dir, trades)` (line 536): Assess model health from registry and trade data.
  - `load_feature_importance(model_dir)` (line 590): Load feature importance from latest model metadata.
  - `score_to_status(score)` (line 656): Convert numeric score to PASS/WARN/FAIL status.
  - `collect_health_data()` (line 665): Run full system health assessment.
  - `_check_data_integrity()` (line 702): Check survivorship bias and data quality.
  - `_check_promotion_contract()` (line 774): Verify promotion gate configuration.
  - `_check_walkforward()` (line 855): Verify walk-forward validation setup.
  - `_check_execution()` (line 901): Audit execution cost model.
  - `_check_complexity()` (line 950): Audit feature and knob complexity.
  - `_check_strengths()` (line 1004): Identify what's working well.
- Classes:
  - `HealthCheck` (line 622): Single health check result.
  - `SystemHealthPayload` (line 632): Full system health assessment.

### `api/services/data_service.py`

- LOC: 208
- Module intent: Wraps data.loader for API consumption.
- Imports (6): `from __future__ import annotations`, `import json`, `import logging`, `from datetime import date, datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`
- Top-level functions: none
- Classes:
  - `DataService` (line 13): Thin wrapper around ``data.loader`` — all methods are synchronous.
    - Methods (5): `load_universe`, `load_single_ticker`, `get_cached_tickers`, `get_universe_info`, `get_cache_status`

### `api/services/health_service.py`

- LOC: 686
- Module intent: System health assessment for API consumption.
- Imports (8): `from __future__ import annotations`, `import json`, `import logging`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `HealthService` (line 24): Computes system health from model age, cache freshness, and runtime metrics.
    - Methods (19): `get_quick_status`, `get_detailed_health`, `compute_comprehensive_health`, `_domain_score`, `_check_signal_decay`, `_check_feature_importance_drift`, `_check_regime_transition_health`, `_check_prediction_distribution`, `_check_survivorship_bias`, `_check_correlation_regime`, `_check_execution_quality`, `_check_tail_risk`, `_check_information_ratio`, `_check_cv_gap_trend`, `_check_data_quality_anomalies`, `_check_ensemble_disagreement`, `_check_market_microstructure`, `_check_retraining_effectiveness`, `_check_capital_utilization`

### `api/services/kalshi_service.py`

- LOC: 50
- Module intent: Wraps kalshi.storage for API consumption.
- Imports (3): `from __future__ import annotations`, `import logging`, `from typing import Any, Dict, List`
- Top-level functions: none
- Classes:
  - `KalshiService` (line 10): Kalshi event market data — conditionally enabled.
    - Methods (3): `__init__`, `get_events`, `get_distributions`

### `api/services/model_service.py`

- LOC: 98
- Module intent: Wraps models.* modules for API consumption.
- Imports (5): `from __future__ import annotations`, `import json`, `import logging`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`
- Top-level functions: none
- Classes:
  - `ModelService` (line 12): Synchronous model metadata / health wrapper.
    - Methods (5): `list_versions`, `get_model_health`, `get_feature_importance`, `get_feature_correlations`, `get_champion_info`

### `api/services/regime_service.py`

- LOC: 50
- Module intent: Wraps regime.detector for API consumption.
- Imports (4): `from __future__ import annotations`, `import logging`, `from typing import Any, Dict`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `RegimeService` (line 12): Synchronous regime detection wrapper.
    - Methods (2): `detect_current_regime`, `get_regime_names`

### `api/services/results_service.py`

- LOC: 85
- Module intent: Reads/writes to the results/ directory.
- Imports (6): `from __future__ import annotations`, `import json`, `import logging`, `from pathlib import Path`, `from typing import Any, Dict, List`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `ResultsService` (line 14): Unified access to persisted result artefacts.
    - Methods (3): `get_latest_backtest`, `get_latest_predictions`, `list_all_results`

## Package `autopilot`

### `autopilot/__init__.py`

- LOC: 20
- Module intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Imports (5): `from .strategy_discovery import StrategyCandidate, StrategyDiscovery`, `from .promotion_gate import PromotionDecision, PromotionGate`, `from .registry import StrategyRegistry`, `from .paper_trader import PaperTrader`, `from .engine import AutopilotEngine`
- Top-level functions: none
- Classes: none

### `autopilot/engine.py`

- LOC: 990
- Module intent: End-to-end autopilot cycle:
- Imports (24): `import json`, `import logging`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import re`, `import numpy as np`, `import pandas as pd`, `from ..backtest.engine import Backtester`, `from ..backtest.advanced_validation import capacity_analysis, deflated_sharpe_ratio, probability_of_backtest_overfitting`, `from ..backtest.validation import walk_forward_validate, run_statistical_tests, combinatorial_purged_cv, superior_predictive_ability, strategy_signal_returns`, `from ..models.walk_forward import _expanding_walk_forward_folds`, `from ..config import AUTOPILOT_CYCLE_REPORT, AUTOPILOT_FEATURE_MODE, BACKTEST_ASSUMED_CAPITAL_USD, CPCV_PARTITIONS, CPCV_TEST_PARTITIONS, EXEC_MAX_PARTICIPATION, REQUIRE_PERMNO, SURVIVORSHIP_UNIVERSE_NAME, WF_MAX_TRAIN_DATES` (+12 more)
- Top-level functions: none
- Classes:
  - `HeuristicPredictor` (line 61): Lightweight fallback predictor used when sklearn-backed model artifacts
    - Methods (3): `__init__`, `_rolling_zscore`, `predict`
  - `AutopilotEngine` (line 136): Coordinates discovery, promotion, and paper execution.
    - Methods (15): `__init__`, `_log`, `_is_permno_key`, `_assert_permno_price_data`, `_assert_permno_prediction_panel`, `_assert_permno_latest_predictions`, `_load_data`, `_build_regimes`, `_train_baseline`, `_ensure_predictor`, `_predict_universe`, `_walk_forward_predictions`, `_evaluate_candidates`, `_compute_optimizer_weights`, `run_cycle`

### `autopilot/paper_trader.py`

- LOC: 529
- Module intent: Stateful paper-trading engine for promoted strategies.
- Imports (9): `import json`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from ..config import PAPER_STATE_PATH, PAPER_INITIAL_CAPITAL, PAPER_MAX_TOTAL_POSITIONS, TRANSACTION_COST_BPS, PAPER_USE_KELLY_SIZING, PAPER_KELLY_FRACTION, PAPER_KELLY_LOOKBACK_TRADES, PAPER_KELLY_MIN_SIZE_MULTIPLIER, PAPER_KELLY_MAX_SIZE_MULTIPLIER, REGIME_RISK_MULTIPLIER`, `from ..risk.position_sizer import PositionSizer`, `from .registry import ActiveStrategy`
- Top-level functions: none
- Classes:
  - `PaperTrader` (line 28): Executes paper entries/exits from promoted strategy definitions.
    - Methods (14): `__init__`, `_load_state`, `_save_state`, `_resolve_as_of`, `_latest_predictions_by_id`, `_latest_predictions_by_ticker`, `_current_price`, `_position_id`, `_mark_to_market`, `_trade_return`, `_historical_trade_stats`, `_market_risk_stats`, `_position_size_pct`, `run_cycle`

### `autopilot/promotion_gate.py`

- LOC: 281
- Module intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Imports (6): `from dataclasses import dataclass, asdict`, `from typing import Dict, List, Optional`, `import numpy as np`, `from ..backtest.engine import BacktestResult`, `from ..config import PROMOTION_MIN_TRADES, PROMOTION_MIN_WIN_RATE, PROMOTION_MIN_SHARPE, PROMOTION_MIN_PROFIT_FACTOR, PROMOTION_MAX_DRAWDOWN, PROMOTION_MIN_ANNUAL_RETURN, PROMOTION_REQUIRE_ADVANCED_CONTRACT, PROMOTION_MAX_DSR_PVALUE, PROMOTION_MAX_PBO, PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED, PROMOTION_MAX_CAPACITY_UTILIZATION, PROMOTION_MIN_WF_OOS_CORR, PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION, PROMOTION_MAX_WF_IS_OOS_GAP, PROMOTION_MIN_REGIME_POSITIVE_FRACTION, PROMOTION_EVENT_MAX_WORST_EVENT_LOSS, PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE, PROMOTION_EVENT_MIN_REGIME_STABILITY, PROMOTION_REQUIRE_STATISTICAL_TESTS, PROMOTION_REQUIRE_CPCV, PROMOTION_REQUIRE_SPA`, `from .strategy_discovery import StrategyCandidate`
- Top-level functions: none
- Classes:
  - `PromotionDecision` (line 37): Serializable promotion-gate decision for a single strategy candidate evaluation.
    - Methods (1): `to_dict`
  - `PromotionGate` (line 52): Applies hard risk/quality constraints before a strategy can be paper-deployed.
    - Methods (4): `__init__`, `evaluate`, `evaluate_event_strategy`, `rank`

### `autopilot/registry.py`

- LOC: 110
- Module intent: Persistent strategy registry for promoted candidates.
- Imports (7): `import json`, `from dataclasses import dataclass, asdict`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Dict, List`, `from ..config import STRATEGY_REGISTRY_PATH, PROMOTION_MAX_ACTIVE_STRATEGIES`, `from .promotion_gate import PromotionDecision`
- Top-level functions: none
- Classes:
  - `ActiveStrategy` (line 15): Persisted record for a currently active promoted strategy.
    - Methods (1): `to_dict`
  - `StrategyRegistry` (line 29): Maintains promoted strategy state and historical promotion decisions.
    - Methods (5): `__init__`, `_load`, `_save`, `get_active`, `apply_promotions`

### `autopilot/strategy_discovery.py`

- LOC: 79
- Module intent: Strategy discovery for execution-layer parameter variants.
- Imports (3): `from dataclasses import dataclass, asdict`, `from typing import Dict, List`, `from ..config import ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD, POSITION_SIZE_PCT, DISCOVERY_ENTRY_MULTIPLIERS, DISCOVERY_CONFIDENCE_OFFSETS, DISCOVERY_RISK_VARIANTS, DISCOVERY_MAX_POSITIONS_VARIANTS`
- Top-level functions: none
- Classes:
  - `StrategyCandidate` (line 22): Execution-parameter variant generated for backtest and promotion evaluation.
    - Methods (1): `to_dict`
  - `StrategyDiscovery` (line 37): Generates a deterministic candidate grid for backtest validation.
    - Methods (2): `__init__`, `generate`

## Package `backtest`

### `backtest/__init__.py`

- LOC: 4
- Module intent: Backtesting package exports and namespace initialization.
- Top-level functions: none
- Classes: none

### `backtest/advanced_validation.py`

- LOC: 581
- Module intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- Imports (5): `from dataclasses import dataclass, field`, `from math import erf`, `from typing import List, Optional, Dict`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `deflated_sharpe_ratio(observed_sharpe, n_trials, n_returns, skewness, kurtosis, annualization_factor)` (line 94): Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
  - `probability_of_backtest_overfitting(returns_matrix, n_partitions)` (line 176): Probability of Backtest Overfitting (Bailey et al., 2017).
  - `monte_carlo_validation(trade_returns, n_simulations, holding_days, method)` (line 291): Monte Carlo validation of strategy performance.
  - `capacity_analysis(trades, price_data, capital_usd, max_participation_rate, impact_coefficient_bps)` (line 367): Estimate strategy capacity and market impact.
  - `run_advanced_validation(trade_returns, trades, price_data, n_strategy_variants, holding_days, returns_matrix, verbose)` (line 460): Run all advanced validation tests.
  - `_print_report(report)` (line 541): Pretty-print advanced validation report.
- Classes:
  - `DeflatedSharpeResult` (line 38): Result of Deflated Sharpe Ratio test.
  - `PBOResult` (line 49): Probability of Backtest Overfitting result.
  - `MonteCarloResult` (line 59): Monte Carlo simulation result.
  - `CapacityResult` (line 73): Strategy capacity analysis.
  - `AdvancedValidationReport` (line 84): Complete advanced validation report.

### `backtest/engine.py`

- LOC: 1869
- Module intent: Backtester — converts model predictions into simulated trades.
- Imports (9): `from dataclasses import dataclass, field`, `from datetime import datetime`, `from typing import Dict, List, Optional, Tuple`, `import re`, `import numpy as np`, `import pandas as pd`, `import logging`, `from ..config import TRANSACTION_COST_BPS, ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_POSITIONS, POSITION_SIZE_PCT, BACKTEST_ASSUMED_CAPITAL_USD, EXEC_SPREAD_BPS, EXEC_MAX_PARTICIPATION, EXEC_IMPACT_COEFF_BPS, EXEC_MIN_FILL_RATIO, REGIME_RISK_MULTIPLIER, EXEC_DYNAMIC_COSTS, EXEC_DOLLAR_VOLUME_REF_USD, EXEC_VOL_REF, EXEC_VOL_SPREAD_BETA, EXEC_GAP_SPREAD_BETA, EXEC_RANGE_SPREAD_BETA, EXEC_VOL_IMPACT_BETA, REQUIRE_PERMNO, ALMGREN_CHRISS_ENABLED, ALMGREN_CHRISS_ADV_THRESHOLD, ALMGREN_CHRISS_RISK_AVERSION, MAX_ANNUALIZED_TURNOVER, ALMGREN_CHRISS_FALLBACK_VOL, REGIME_2_TRADE_ENABLED, REGIME_2_SUPPRESSION_MIN_CONFIDENCE`, `from .execution import ExecutionModel`
- Top-level functions: none
- Classes:
  - `Trade` (line 48): Trade record produced by the backtester for one simulated position lifecycle.
  - `BacktestResult` (line 71): Aggregate backtest outputs including metrics, curves, and trade history.
  - `Backtester` (line 105): Simulates trading from model predictions.
    - Methods (21): `__init__`, `_init_risk_components`, `_almgren_chriss_cost_bps`, `_simulate_entry`, `_simulate_exit`, `_execution_context`, `_effective_return_series`, `_delisting_adjustment_multiplier`, `_trade_realized_return`, `_is_permno_key`, `_assert_permno_inputs`, `run`, `_process_signals`, `_process_signals_risk_managed`, `_compute_metrics`, `_build_daily_equity`, `_compute_turnover`, `_compute_regime_performance`, `_compute_tca`, `_print_result` (+1 more)

### `backtest/execution.py`

- LOC: 273
- Module intent: Execution simulator with spread, market impact, and participation limits.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Dict, List`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `calibrate_cost_model(fills, actual_prices)` (line 153): Calibrate execution-cost model parameters from historical fills.
- Classes:
  - `ExecutionFill` (line 17): Simulated execution fill outcome returned by the execution model.
  - `ExecutionModel` (line 27): Simple market-impact model for backtests.
    - Methods (2): `__init__`, `simulate`

### `backtest/optimal_execution.py`

- LOC: 201
- Module intent: Almgren-Chriss (2001) optimal execution model.
- Imports (3): `from __future__ import annotations`, `import numpy as np`, `from ..config import ALMGREN_CHRISS_RISK_AVERSION`
- Top-level functions:
  - `almgren_chriss_trajectory(total_shares, n_intervals, daily_volume, daily_volatility, risk_aversion, temporary_impact, permanent_impact)` (line 20): Compute the optimal execution trajectory using the Almgren-Chriss model.
  - `estimate_execution_cost(trajectory, reference_price, daily_volume, daily_volatility, temporary_impact, permanent_impact)` (line 106): Estimate the total execution cost for a given trade trajectory.
- Classes: none

### `backtest/validation.py`

- LOC: 749
- Module intent: Walk-forward validation and statistical tests.
- Imports (7): `from dataclasses import dataclass, field`, `from itertools import combinations`, `from math import erf`, `from typing import List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from ..config import IC_ROLLING_WINDOW`
- Top-level functions:
  - `walk_forward_validate(predictions, actuals, n_folds, entry_threshold, max_overfit_ratio, purge_gap, embargo, max_train_samples)` (line 192): Walk-forward validation of prediction quality with purge gap.
  - `_benjamini_hochberg(pvals, alpha)` (line 337): Benjamini-Hochberg procedure for multiple testing correction.
  - `run_statistical_tests(predictions, actuals, trade_returns, entry_threshold, holding_days)` (line 361): Statistical tests for prediction quality.
  - `_partition_bounds(n_obs, n_partitions)` (line 500): Return contiguous [start, end) bounds for temporal partitions.
  - `combinatorial_purged_cv(predictions, actuals, entry_threshold, n_partitions, n_test_partitions, purge_gap, embargo, max_combinations)` (line 515): Combinatorial Purged Cross-Validation for signal robustness.
  - `strategy_signal_returns(predictions, actuals, entry_threshold, confidence, min_confidence)` (line 658): Build per-sample strategy return series from prediction signals.
  - `superior_predictive_ability(strategy_returns, benchmark_returns, n_bootstraps, block_size, random_state)` (line 684): Single-strategy SPA-style block-bootstrap test on differential returns.
- Classes:
  - `WalkForwardFold` (line 111): Per-fold walk-forward validation metrics for one temporal split.
  - `WalkForwardResult` (line 122): Aggregate walk-forward validation summary and overfitting diagnostics.
  - `StatisticalTests` (line 136): Bundle of statistical significance tests for prediction quality and signal returns.
  - `CPCVResult` (line 163): Combinatorial purged cross-validation summary metrics and pass/fail status.
  - `SPAResult` (line 180): Superior Predictive Ability (SPA) test result bundle.

## Package `data`

### `data/__init__.py`

- LOC: 32
- Module intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- Imports (5): `from .loader import load_ohlcv, load_universe, load_survivorship_universe, load_with_delistings`, `from .local_cache import save_ohlcv, load_ibkr_data, list_cached_tickers, cache_universe`, `from .provider_registry import get_provider, list_providers, register_provider`, `from .quality import DataQualityReport, assess_ohlcv_quality, generate_quality_report, flag_degraded_stocks`, `from .feature_store import FeatureStore`
- Top-level functions: none
- Classes: none

### `data/alternative.py`

- LOC: 652
- Module intent: Alternative data framework — WRDS-backed implementation.
- Imports (7): `from __future__ import annotations`, `import logging`, `from datetime import datetime, timedelta`, `from pathlib import Path`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_get_wrds()` (line 31): Return the cached WRDSProvider singleton, or None.
  - `compute_alternative_features(ticker, provider, cache_dir)` (line 521): Gather all available alternative data and return as a feature DataFrame.
- Classes:
  - `AlternativeDataProvider` (line 51): WRDS-backed alternative data provider.
    - Methods (7): `__init__`, `_resolve_permno`, `get_earnings_surprise`, `get_options_flow`, `get_short_interest`, `get_insider_transactions`, `get_institutional_ownership`

### `data/feature_store.py`

- LOC: 312
- Module intent: Point-in-time feature store for backtest acceleration.
- Imports (8): `from __future__ import annotations`, `import json`, `import logging`, `from datetime import datetime`, `from pathlib import Path`, `from typing import Dict, List, Optional`, `import pandas as pd`, `from ..config import ROOT_DIR`
- Top-level functions: none
- Classes:
  - `FeatureStore` (line 38): Point-in-time feature store for backtest acceleration.
    - Methods (9): `__init__`, `_version_dir`, `_ts_tag`, `_parquet_path`, `_meta_path`, `save_features`, `load_features`, `list_available`, `invalidate`

### `data/loader.py`

- LOC: 732
- Module intent: Data loader — self-contained data loading with multiple sources.
- Imports (9): `import logging`, `from datetime import date, timedelta`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from ..config import CACHE_MAX_STALENESS_DAYS, CACHE_TRUSTED_SOURCES, CACHE_WRDS_SPAN_ADVANTAGE_DAYS, DATA_QUALITY_ENABLED, LOOKBACK_YEARS, MIN_BARS, OPTIONMETRICS_ENABLED, REQUIRE_PERMNO, WRDS_ENABLED`, `from .local_cache import list_cached_tickers, load_ohlcv_with_meta as cache_load_with_meta, save_ohlcv as cache_save`, `from .provider_registry import get_provider`, `from .quality import assess_ohlcv_quality`
- Top-level functions:
  - `_permno_from_meta(meta)` (line 51): Internal helper for permno from meta.
  - `_ticker_from_meta(meta)` (line 64): Internal helper for ticker from meta.
  - `_attach_id_attrs(df, permno, ticker)` (line 72): Internal helper for attach id attrs.
  - `_cache_source(meta)` (line 88): Internal helper for cache source.
  - `_cache_is_usable(cached, meta, years, require_recent, require_trusted)` (line 94): Internal helper for cache is usable.
  - `_cached_universe_subset(candidates)` (line 131): Prefer locally cached symbols to keep offline runs deterministic.
  - `_normalize_ohlcv(df)` (line 154): Return a sorted, deterministic OHLCV frame or None if invalid.
  - `_harmonize_return_columns(df)` (line 169): Standardize return columns so backtests can consume total-return streams.
  - `_merge_option_surface_from_prefetch(df, permno, option_surface)` (line 205): Merge pre-fetched OptionMetrics surface rows into a single PERMNO panel.
  - `load_ohlcv(ticker, years, use_cache, use_wrds)` (line 236): Load daily OHLCV data for a single ticker.
  - `get_data_provenance()` (line 385): Return a summary of data source provenance and any fallbacks that occurred.
  - `get_skip_reasons()` (line 394): Return per-ticker skip reasons from the most recent load_universe() call.
  - `load_universe(tickers, years, verbose, use_cache, use_wrds)` (line 403): Load OHLCV data for multiple symbols. Returns {permno: DataFrame}.
  - `load_survivorship_universe(as_of_date, years, verbose)` (line 463): Load a survivorship-bias-free universe using WRDS CRSP.
  - `load_with_delistings(tickers, years, verbose)` (line 590): Load OHLCV data including delisting returns from CRSP.
- Classes: none

### `data/local_cache.py`

- LOC: 702
- Module intent: Local data cache for daily OHLCV data.
- Imports (7): `import json`, `import logging`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Dict, List, Mapping, Optional, Tuple`, `import pandas as pd`, `from ..config import DATA_CACHE_DIR, FRAMEWORK_DIR`
- Top-level functions:
  - `_ensure_cache_dir()` (line 28): Create cache directory if it doesn't exist.
  - `_normalize_ohlcv_columns(df)` (line 34): Normalize OHLCV column names to quant_engine's canonical schema.
  - `_to_daily_ohlcv(df)` (line 71): Convert any candidate frame into validated daily OHLCV.
  - `_read_csv_ohlcv(path)` (line 93): Internal helper to read csv ohlcv from storage.
  - `_candidate_csv_paths(cache_root, ticker)` (line 111): Internal helper for candidate csv paths.
  - `_cache_meta_path(data_path, ticker)` (line 128): Return the metadata sidecar path for a cache data file.
  - `_read_cache_meta(data_path, ticker)` (line 149): Internal helper to read cache meta from storage.
  - `_write_cache_meta(data_path, ticker, df, source, meta)` (line 168): Internal helper to write cache meta to storage.
  - `save_ohlcv(ticker, df, cache_dir, source, meta)` (line 195): Save OHLCV DataFrame to local cache.
  - `load_ohlcv_with_meta(ticker, cache_dir)` (line 231): Load OHLCV and sidecar metadata from cache roots.
  - `load_ohlcv(ticker, cache_dir)` (line 305): Load OHLCV DataFrame from local cache.
  - `load_intraday_ohlcv(ticker, timeframe, cache_dir)` (line 321): Load intraday OHLCV data from cache.
  - `list_intraday_timeframes(ticker, cache_dir)` (line 390): Return list of available intraday timeframes for a ticker in the cache.
  - `list_cached_tickers(cache_dir)` (line 404): List all tickers available in cache roots.
  - `_daily_cache_files(root)` (line 423): Return de-duplicated daily-cache candidate files for one root.
  - `_ticker_from_cache_path(path)` (line 446): Internal helper for ticker from cache path.
  - `_timeframe_from_cache_path(path)` (line 461): Determine the canonical timeframe from a cache file path.
  - `_all_cache_files(root)` (line 472): Return de-duplicated daily + intraday cache candidate files for one root.
  - `rehydrate_cache_metadata(cache_roots, source_by_root, default_source, only_missing, overwrite_source, dry_run)` (line 494): Backfill metadata sidecars for existing cache files without rewriting price data.
  - `load_ibkr_data(data_dir)` (line 640): Scan a directory of IBKR-downloaded files (CSV or parquet).
  - `cache_universe(data, cache_dir, source)` (line 695): Save all tickers in a data dict to the local cache.
- Classes: none

### `data/provider_base.py`

- LOC: 14
- Module intent: Shared provider protocol for pluggable data connectors.
- Imports (2): `from __future__ import annotations`, `from typing import Protocol`
- Top-level functions: none
- Classes:
  - `DataProvider` (line 9): Protocol defining the minimal interface expected from pluggable data providers.
    - Methods (1): `available`

### `data/provider_registry.py`

- LOC: 53
- Module intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- Imports (3): `from __future__ import annotations`, `from typing import Callable, Dict, List`, `from .provider_base import DataProvider`
- Top-level functions:
  - `_wrds_factory()` (line 14): Lazily import and construct the WRDS provider.
  - `_kalshi_factory()` (line 21): Lazily import and construct the Kalshi provider.
  - `get_provider(name)` (line 34): Construct a registered provider instance by name.
  - `list_providers()` (line 43): Return the names of supported data providers available through the registry.
  - `register_provider(name, factory)` (line 48): Register or override a provider factory under a normalized key.
- Classes: none

### `data/quality.py`

- LOC: 263
- Module intent: Data quality checks for OHLCV time series.
- Imports (5): `from dataclasses import dataclass, asdict`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import MAX_MISSING_BAR_FRACTION, MAX_ZERO_VOLUME_FRACTION, MAX_ABS_DAILY_RETURN`
- Top-level functions:
  - `_expected_trading_days(start, end)` (line 32): Return expected trading days between *start* and *end* (inclusive).
  - `assess_ohlcv_quality(df, max_missing_bar_fraction, max_zero_volume_fraction, max_abs_daily_return)` (line 56): assess ohlcv quality.
  - `generate_quality_report(ohlcv_dict)` (line 122): Return a per-stock quality summary DataFrame.
  - `flag_degraded_stocks(ohlcv_dict)` (line 230): Return a list of tickers whose data quality is below threshold.
- Classes:
  - `DataQualityReport` (line 45): Structured result of OHLCV quality checks with metrics and warning tags.
    - Methods (1): `to_dict`

### `data/survivorship.py`

- LOC: 935
- Module intent: Survivorship Bias Controls (Tasks 112-117)
- Imports (9): `import numpy as np`, `import pandas as pd`, `from dataclasses import dataclass, field`, `from typing import Dict, List, Optional, Tuple, Set`, `from datetime import datetime, date`, `from enum import Enum`, `import json`, `import sqlite3`, `import os`
- Top-level functions:
  - `hydrate_universe_history_from_snapshots(snapshots, universe_name, db_path, verbose)` (line 353): Build point-in-time universe intervals from snapshot rows.
  - `hydrate_sp500_history_from_wrds(start_date, end_date, db_path, freq, verbose)` (line 450): Pull historical S&P 500 snapshots from WRDS and hydrate local PIT DB.
  - `filter_panel_by_point_in_time_universe(panel, universe_name, db_path, verbose)` (line 482): Filter MultiIndex panel rows by point-in-time universe membership.
  - `reconstruct_historical_universe(universe_name, as_of_date, tracker)` (line 908): Task 115: Quick function to reconstruct historical universe.
  - `calculate_survivorship_bias_impact(prices, start_date, end_date, universe_name)` (line 920): Task 117: Quick function to calculate survivorship bias impact.
- Classes:
  - `DelistingReason` (line 25): Reason for stock delisting.
  - `UniverseMember` (line 38): Task 112: Track a symbol's membership in a universe.
    - Methods (2): `is_active_on`, `to_dict`
  - `UniverseChange` (line 72): Task 114: Track a change to universe membership.
    - Methods (1): `to_dict`
  - `DelistingEvent` (line 98): Task 113: Track delisting event with proper returns.
    - Methods (1): `to_dict`
  - `SurvivorshipReport` (line 128): Task 117: Report comparing returns with/without survivorship adjustment.
    - Methods (1): `to_dict`
  - `UniverseHistoryTracker` (line 171): Task 112, 114, 115: Track historical universe membership.
    - Methods (8): `__init__`, `_init_db`, `add_member`, `record_change`, `get_universe_on_date`, `get_changes_in_period`, `bulk_load_universe`, `clear_universe`
  - `DelistingHandler` (line 555): Task 113, 116: Handle delisting events properly.
    - Methods (9): `__init__`, `_init_db`, `record_delisting`, `preserve_price_history`, `get_dead_company_prices`, `get_delisting_event`, `get_delisting_return`, `is_delisted`, `get_all_delisted_symbols`
  - `SurvivorshipBiasController` (line 728): Task 117: Main controller for survivorship bias analysis.
    - Methods (4): `__init__`, `get_survivorship_free_universe`, `calculate_bias_impact`, `format_report`

### `data/wrds_provider.py`

- LOC: 1615
- Module intent: wrds_provider.py
- Imports (9): `import logging`, `import os`, `import threading`, `import warnings`, `from datetime import datetime`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `import re as _re`
- Top-level functions:
  - `_sanitize_ticker_list(tickers)` (line 76): Build a SQL-safe IN-clause string from ticker symbols.
  - `_sanitize_permno_list(permnos)` (line 92): Build a SQL-safe IN-clause string from PERMNO values.
  - `_read_pgpass_password()` (line 107): Read the WRDS password from ~/.pgpass so the wrds library doesn't
  - `_get_connection()` (line 141): Get or create a cached WRDS connection. Returns None if unavailable.
  - `get_wrds_provider()` (line 1602): Get or create the default WRDSProvider singleton.
  - `wrds_available()` (line 1610): Quick check: is WRDS accessible?
- Classes:
  - `WRDSProvider` (line 216): WRDS data provider for the auto-discovery pipeline.
    - Methods (21): `__init__`, `available`, `_query`, `_query_silent`, `get_sp500_universe`, `get_sp500_history`, `resolve_permno`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_optionmetrics_link`, `_nearest_iv`, `get_option_surface_features`, `get_fundamentals`, `get_earnings_surprises`, `get_institutional_ownership`, `get_taqmsec_ohlcv`, `query_options_volume`, `query_short_interest`, `query_insider_transactions`, `_permno_to_ticker` (+1 more)

## Package `features`

### `features/__init__.py`

- LOC: 4
- Module intent: Feature engineering package namespace.
- Imports (1): `from .pipeline import FEATURE_METADATA, get_feature_type`
- Top-level functions: none
- Classes: none

### `features/harx_spillovers.py`

- LOC: 242
- Module intent: HARX Volatility Spillover features (Tier 6.1).
- Imports (4): `from __future__ import annotations`, `from typing import Dict, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_realized_volatility(returns, window, min_periods)` (line 38): Rolling realized volatility (annualised std of returns).
  - `_ols_lstsq(X, y)` (line 48): OLS via numpy lstsq.  Returns coefficient vector.
  - `compute_harx_spillovers(returns_by_asset, rv_daily_window, rv_weekly_window, rv_monthly_window, regression_window, min_regression_obs)` (line 54): Compute HARX cross-market volatility spillover features.
- Classes: none

### `features/intraday.py`

- LOC: 243
- Module intent: Intraday microstructure features from WRDS TAQmsec tick data.
- Imports (5): `from __future__ import annotations`, `from typing import Any, Dict, Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import MARKET_OPEN, MARKET_CLOSE`
- Top-level functions:
  - `compute_intraday_features(ticker, date, wrds_provider)` (line 24): Compute intraday microstructure features for a single ticker on a date.
  - `compute_rolling_vwap(df, window)` (line 203): Compute causal rolling VWAP and deviation features.
- Classes: none

### `features/lob_features.py`

- LOC: 311
- Module intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- Imports (4): `from __future__ import annotations`, `from typing import Dict, Optional, Tuple`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_inter_bar_durations(index)` (line 34): Compute inter-bar durations in seconds from a DatetimeIndex.
  - `_estimate_poisson_lambda(durations)` (line 51): Estimate trade arrival rate (lambda) from inter-arrival durations.
  - `_signed_volume(bars)` (line 72): Approximate trade direction using candle body (close - open).
  - `compute_lob_features(intraday_bars, freq)` (line 88): Compute Markov LOB proxy features for a single stock-day.
  - `compute_lob_features_batch(intraday_data, freq)` (line 239): Compute LOB features for multiple stock-days in batch.
- Classes: none

### `features/macro.py`

- LOC: 244
- Module intent: FRED macro indicator features for quant_engine.
- Imports (7): `from __future__ import annotations`, `import hashlib`, `import logging`, `from pathlib import Path`, `from typing import Dict, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_cache_key(series_id, start, end)` (line 43): Generate a deterministic cache filename.
- Classes:
  - `MacroFeatureProvider` (line 50): FRED API integration for macro indicator features.
    - Methods (5): `__init__`, `_fetch_series_fredapi`, `_fetch_series_requests`, `_fetch_series`, `get_macro_features`

### `features/options_factors.py`

- LOC: 134
- Module intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- Imports (4): `from __future__ import annotations`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_pick_numeric(df, candidates)` (line 12): Internal helper for pick numeric.
  - `_rolling_percentile_rank(series, window, min_periods)` (line 20): Internal helper for rolling percentile rank.
  - `compute_option_surface_factors(df)` (line 38): Compute minimal high-signal option surface features.
  - `compute_iv_shock_features(df, window)` (line 92): Causal IV change features (G3).
- Classes: none

### `features/pipeline.py`

- LOC: 1272
- Module intent: Feature Pipeline — computes model features from OHLCV data.
- Imports (7): `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`, `from ..indicators import ATR, NATR, BollingerBandWidth, HistoricalVolatility, BBWidthPercentile, NATRPercentile, VolatilitySqueeze, RSI, MACD, MACDSignal, MACDHistogram, ROC, Stochastic, StochasticD, WilliamsR, CCI, SMA, EMA, PriceVsSMA, SMASlope, ADX, Aroon, EMAAlignment, TrendStrength, PriceVsEMAStack, MarketRegime, VolatilityRegime, VolumeRatio, OBV, OBVSlope, MFI, RVOL, NetVolumeTrend, VolumeForce, AccumulationDistribution, HigherHighs, LowerLows, CandleBody, CandleDirection, GapPercent, DistanceFromHigh, DistanceFromLow, PricePercentile, PivotHigh, PivotLow, NBarHighBreak, NBarLowBreak, RangeBreakout, ATRTrailingStop, ATRChannel, RiskPerATR, VWAP, PriceVsVWAP, VWAPBands, ValueAreaHigh, ValueAreaLow, POC, PriceVsPOC, ValueAreaPosition, AboveValueArea, BelowValueArea, ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility, VolatilityCone, VolOfVol, GARCHVolatility, VolTermStructure, HurstExponent, MeanReversionHalfLife, ZScore, VarianceRatio, Autocorrelation, KalmanTrend, ShannonEntropy, ApproximateEntropy, AmihudIlliquidity, KyleLambda, RollSpread, FractalDimension, DFA, DominantCycle, ReturnSkewness, ReturnKurtosis, CUSUMDetector, RegimePersistence`, `from .research_factors import ResearchFactorConfig, compute_cross_asset_research_factors, compute_single_asset_research_factors`, `from .options_factors import compute_option_surface_factors`, `from .wave_flow import compute_wave_flow_decomposition`
- Top-level functions:
  - `get_feature_type(feature_name)` (line 384): Return the causality type for a feature.
  - `_filter_causal_features(features)` (line 400): Keep only features with type CAUSAL or END_OF_DAY (drop RESEARCH_ONLY).
  - `_build_indicator_set()` (line 406): Instantiate all indicators with default parameters.
  - `_build_minimal_indicator_set()` (line 487): Lean indicator set for the 'minimal' feature mode.
  - `_get_indicators(minimal)` (line 535): Return the indicator set (cached at module level).
  - `compute_indicator_features(df, verbose, minimal)` (line 548): Compute indicator-based features as continuous columns.
  - `compute_raw_features(df)` (line 579): Compute raw OHLCV-derived features (returns, volume, gaps, etc.).
  - `compute_har_volatility_features(df)` (line 621): Compute HAR (Heterogeneous Autoregressive) realized volatility features.
  - `compute_multiscale_features(df)` (line 664): Compute momentum, RSI, and volatility features at multiple time scales.
  - `compute_interaction_features(features, pairs)` (line 700): Generate interaction features from pairs of continuous indicators.
  - `compute_targets(df, horizons, benchmark_close)` (line 753): Compute forward return targets for supervised learning.
  - `_winsorize_expanding(df, lower_q, upper_q)` (line 802): Winsorize features using expanding-window quantiles (no look-ahead).
- Classes:
  - `FeaturePipeline` (line 827): End-to-end feature computation pipeline.
    - Methods (4): `__init__`, `compute`, `compute_universe`, `_load_benchmark_close`

### `features/research_factors.py`

- LOC: 985
- Module intent: Research-derived factor construction for quant_engine.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Dict, List, Mapping, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_rolling_zscore(series, window, min_periods)` (line 43): Causal rolling z-score.
  - `_safe_pct_change(series, periods)` (line 50): Internal helper for safe pct change.
  - `_required_ohlcv(df)` (line 56): Internal helper for required ohlcv.
  - `compute_order_flow_impact_factors(df, config)` (line 66): Order-flow imbalance and price-impact proxies (Cont et al. inspired).
  - `compute_markov_queue_features(df, config)` (line 134): Markov-style queue imbalance features (de Larrard style state framing).
  - `compute_time_series_momentum_factors(df, config)` (line 204): Vol-scaled time-series momentum factors (Moskowitz/Ooi/Pedersen style).
  - `compute_vol_scaled_momentum(df, horizons, vol_window)` (line 252): Volatility-scaled time-series momentum enhancements.
  - `_rolling_levy_area(dx, dy, window, min_periods)` (line 344): Rolling Levy area for a 2D path of increments.
  - `compute_signature_path_features(df, config)` (line 370): Signature-inspired path features for returns-volume trajectory.
  - `compute_vol_surface_factors(df, config)` (line 405): Volatility term-structure factors inspired by implied-vol surface dynamics.
  - `compute_single_asset_research_factors(df, config)` (line 474): Compute all single-asset research factors.
  - `_standardize_block(block)` (line 492): Column-wise z-score with NaN-safe handling.
  - `_lagged_weight_matrix(values, t, window, min_obs)` (line 501): Build positive lagged correlation weights:
  - `compute_cross_asset_research_factors(price_data, config)` (line 540): Compute cross-asset network momentum and volatility spillover factors.
  - `_dtw_distance_numpy(x, y)` (line 652): Pure numpy DTW distance computation using dynamic programming.
  - `_dtw_avg_lag_from_path(path)` (line 692): Extract average lag from DTW alignment path.
  - `compute_dtw_lead_lag(returns, window, max_lag)` (line 703): DTW-based lead-lag detection across a universe of assets.
  - `_numpy_order2_signature(price_inc, volume_inc)` (line 858): Pure numpy computation of truncated order-2 path signature for a 2D path
  - `compute_path_signatures(df, windows, order)` (line 901): Compute truncated path signatures of (price, volume) paths.
- Classes:
  - `ResearchFactorConfig` (line 28): Configuration for research-derived factor generation.

### `features/version.py`

- LOC: 168
- Module intent: Feature versioning system.
- Imports (8): `from __future__ import annotations`, `import hashlib`, `import json`, `import logging`, `from dataclasses import dataclass, field`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`
- Top-level functions: none
- Classes:
  - `FeatureVersion` (line 23): Immutable snapshot of a feature pipeline configuration.
    - Methods (6): `__post_init__`, `n_features`, `compute_hash`, `to_dict`, `diff`, `is_compatible`
  - `FeatureRegistry` (line 77): Registry tracking feature versions over time with JSON persistence.
    - Methods (8): `__init__`, `register`, `get_version`, `get_latest`, `list_versions`, `check_compatibility`, `_load`, `_save`

### `features/wave_flow.py`

- LOC: 144
- Module intent: Wave-Flow Decomposition for quant_engine.
- Imports (4): `from __future__ import annotations`, `from typing import List`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `compute_wave_flow_decomposition(df, short_window, long_window, regime_threshold)` (line 26): Decompose the return series into flow (secular trend) and wave (oscillatory)
- Classes: none

## Package `indicators`

### `indicators/__init__.py`

- LOC: 89
- Module intent: Quant Engine Indicators — self-contained copy of the technical indicator library.
- Imports (1): `from .indicators import Indicator, ATR, NATR, BollingerBandWidth, HistoricalVolatility, BBWidthPercentile, NATRPercentile, VolatilitySqueeze, RSI, MACD, MACDSignal, MACDHistogram, ROC, Stochastic, StochasticD, WilliamsR, CCI, SMA, EMA, PriceVsSMA, SMASlope, ADX, Aroon, EMAAlignment, TrendStrength, PriceVsEMAStack, MarketRegime, VolatilityRegime, VolumeRatio, OBV, OBVSlope, MFI, RVOL, NetVolumeTrend, VolumeForce, AccumulationDistribution, HigherHighs, LowerLows, CandleBody, CandleDirection, GapPercent, DistanceFromHigh, DistanceFromLow, PricePercentile, PivotHigh, PivotLow, NBarHighBreak, NBarLowBreak, RangeBreakout, ATRTrailingStop, ATRChannel, RiskPerATR, VWAP, PriceVsVWAP, VWAPBands, AnchoredVWAP, PriceVsAnchoredVWAP, MultiVWAPPosition, ValueAreaHigh, ValueAreaLow, POC, PriceVsPOC, ValueAreaPosition, AboveValueArea, BelowValueArea, Beast666Proximity, Beast666Distance, ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility, VolatilityCone, VolOfVol, GARCHVolatility, VolTermStructure, HurstExponent, MeanReversionHalfLife, ZScore, VarianceRatio, Autocorrelation, KalmanTrend, ShannonEntropy, ApproximateEntropy, AmihudIlliquidity, KyleLambda, RollSpread, FractalDimension, DFA, DominantCycle, ReturnSkewness, ReturnKurtosis, CUSUMDetector, RegimePersistence, get_all_indicators`
- Top-level functions: none
- Classes: none

### `indicators/indicators.py`

- LOC: 2904
- Module intent: Technical Indicator Library
- Imports (4): `import numpy as np`, `import pandas as pd`, `from typing import Optional, Tuple`, `from abc import ABC, abstractmethod`
- Top-level functions:
  - `get_all_indicators()` (line 2744): Return dictionary of all indicator classes.
  - `create_indicator(name)` (line 2899): Create an indicator by name with given parameters.
- Classes:
  - `Indicator` (line 14): Base class for all indicators.
    - Methods (2): `name`, `calculate`
  - `ATR` (line 33): Average True Range - measures volatility.
    - Methods (3): `__init__`, `name`, `calculate`
  - `NATR` (line 60): Normalized ATR - ATR as percentage of close price.
    - Methods (3): `__init__`, `name`, `calculate`
  - `BollingerBandWidth` (line 80): Bollinger Band Width - measures volatility squeeze.
    - Methods (3): `__init__`, `name`, `calculate`
  - `HistoricalVolatility` (line 106): Historical volatility (standard deviation of returns).
    - Methods (3): `__init__`, `name`, `calculate`
  - `RSI` (line 129): Relative Strength Index.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MACD` (line 156): MACD Line (difference between fast and slow EMA).
    - Methods (3): `__init__`, `name`, `calculate`
  - `MACDSignal` (line 177): MACD Signal Line.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MACDHistogram` (line 199): MACD Histogram (MACD - Signal).
    - Methods (3): `__init__`, `name`, `calculate`
  - `ROC` (line 222): Rate of Change.
    - Methods (3): `__init__`, `name`, `calculate`
  - `Stochastic` (line 241): Stochastic %K.
    - Methods (3): `__init__`, `name`, `calculate`
  - `StochasticD` (line 262): Stochastic %D (smoothed %K).
    - Methods (3): `__init__`, `name`, `calculate`
  - `WilliamsR` (line 283): Williams %R.
    - Methods (3): `__init__`, `name`, `calculate`
  - `CCI` (line 304): Commodity Channel Index.
    - Methods (3): `__init__`, `name`, `calculate`
  - `SMA` (line 331): Simple Moving Average.
    - Methods (3): `__init__`, `name`, `calculate`
  - `EMA` (line 348): Exponential Moving Average.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PriceVsSMA` (line 365): Price distance from SMA (as percentage).
    - Methods (3): `__init__`, `name`, `calculate`
  - `SMASlope` (line 384): Slope of SMA (rate of change).
    - Methods (3): `__init__`, `name`, `calculate`
  - `ADX` (line 405): Average Directional Index - trend strength.
    - Methods (3): `__init__`, `name`, `calculate`
  - `Aroon` (line 443): Aroon Oscillator.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolumeRatio` (line 475): Current volume vs average volume.
    - Methods (3): `__init__`, `name`, `calculate`
  - `OBV` (line 493): On-Balance Volume.
    - Methods (2): `name`, `calculate`
  - `OBVSlope` (line 513): OBV rate of change.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MFI` (line 534): Money Flow Index.
    - Methods (3): `__init__`, `name`, `calculate`
  - `HigherHighs` (line 565): Count of higher highs in lookback period.
    - Methods (3): `__init__`, `name`, `calculate`
  - `LowerLows` (line 584): Count of lower lows in lookback period.
    - Methods (3): `__init__`, `name`, `calculate`
  - `CandleBody` (line 603): Candle body size as percentage of range.
    - Methods (2): `name`, `calculate`
  - `CandleDirection` (line 618): Candle direction streak (positive = up candles, negative = down).
    - Methods (3): `__init__`, `name`, `calculate`
  - `GapPercent` (line 636): Gap from previous close as percentage.
    - Methods (2): `name`, `calculate`
  - `DistanceFromHigh` (line 653): Distance from N-period high as percentage.
    - Methods (3): `__init__`, `name`, `calculate`
  - `DistanceFromLow` (line 671): Distance from N-period low as percentage.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PricePercentile` (line 689): Current price percentile within N-period range.
    - Methods (3): `__init__`, `name`, `calculate`
  - `BBWidthPercentile` (line 713): Bollinger Band Width Percentile - identifies squeeze conditions.
    - Methods (3): `__init__`, `name`, `calculate`
  - `NATRPercentile` (line 740): NATR Percentile - where current volatility sits vs history.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolatilitySqueeze` (line 767): Volatility Squeeze indicator - BB inside Keltner Channel.
    - Methods (3): `__init__`, `name`, `calculate`
  - `RVOL` (line 808): Relative Volume - current volume vs same time period average.
    - Methods (3): `__init__`, `name`, `calculate`
  - `NetVolumeTrend` (line 830): Net Volume Trend - accumulation/distribution pressure.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolumeForce` (line 861): Volume Force Index - measures buying/selling pressure.
    - Methods (3): `__init__`, `name`, `calculate`
  - `AccumulationDistribution` (line 883): Accumulation/Distribution Line slope.
    - Methods (3): `__init__`, `name`, `calculate`
  - `EMAAlignment` (line 911): EMA Alignment - checks if EMAs are properly stacked.
    - Methods (3): `__init__`, `name`, `calculate`
  - `TrendStrength` (line 944): Combined trend strength using multiple factors.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PriceVsEMAStack` (line 980): Price position relative to EMA stack.
    - Methods (2): `name`, `calculate`
  - `PivotHigh` (line 1008): Pivot High breakout - price breaks above N-bar high.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PivotLow` (line 1044): Pivot Low breakdown - price breaks below N-bar low.
    - Methods (3): `__init__`, `name`, `calculate`
  - `NBarHighBreak` (line 1080): Simple N-bar high breakout.
    - Methods (3): `__init__`, `name`, `calculate`
  - `NBarLowBreak` (line 1102): Simple N-bar low breakdown.
    - Methods (3): `__init__`, `name`, `calculate`
  - `RangeBreakout` (line 1124): Range Breakout - price breaks out of N-day range.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ATRTrailingStop` (line 1154): Distance from ATR trailing stop.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ATRChannel` (line 1186): Position within ATR channel.
    - Methods (3): `__init__`, `name`, `calculate`
  - `RiskPerATR` (line 1216): Recent price range in ATR units.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MarketRegime` (line 1248): Market regime based on price action.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolatilityRegime` (line 1279): Volatility regime classification.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VWAP` (line 1314): Volume Weighted Average Price - rolling calculation.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PriceVsVWAP` (line 1337): Price distance from VWAP as percentage.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VWAPBands` (line 1359): VWAP Standard Deviation Bands.
    - Methods (3): `__init__`, `name`, `calculate`
  - `AnchoredVWAP` (line 1394): Anchored VWAP - VWAP calculated from N days ago.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PriceVsAnchoredVWAP` (line 1423): Price distance from Anchored VWAP.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MultiVWAPPosition` (line 1445): Position relative to multiple VWAP anchors.
    - Methods (2): `name`, `calculate`
  - `ValueAreaHigh` (line 1474): Value Area High approximation.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ValueAreaLow` (line 1518): Value Area Low approximation.
    - Methods (3): `__init__`, `name`, `calculate`
  - `POC` (line 1561): Point of Control approximation.
    - Methods (3): `__init__`, `name`, `calculate`
  - `PriceVsPOC` (line 1613): Price distance from Point of Control.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ValueAreaPosition` (line 1635): Position within Value Area.
    - Methods (3): `__init__`, `name`, `calculate`
  - `AboveValueArea` (line 1666): Binary: 1 if price above VAH, 0 otherwise.
    - Methods (3): `__init__`, `name`, `calculate`
  - `BelowValueArea` (line 1688): Binary: 1 if price below VAL, 0 otherwise.
    - Methods (3): `__init__`, `name`, `calculate`
  - `Beast666Proximity` (line 1714): Beast 666 Proximity Score (0-100).
    - Methods (3): `__init__`, `name`, `calculate`
  - `Beast666Distance` (line 1811): Signed percent distance from the nearest 666 level.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ParkinsonVolatility` (line 1864): Parkinson range-based volatility estimator. More efficient than close-to-close.
    - Methods (3): `__init__`, `name`, `calculate`
  - `GarmanKlassVolatility` (line 1884): Garman-Klass OHLC volatility estimator. ~8x more efficient than close-to-close.
    - Methods (3): `__init__`, `name`, `calculate`
  - `YangZhangVolatility` (line 1905): Yang-Zhang volatility combining overnight and Rogers-Satchell intraday components.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolatilityCone` (line 1939): Percentile rank of current realized vol vs its historical distribution.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolOfVol` (line 1963): Volatility of volatility - rolling std of rolling volatility.
    - Methods (3): `__init__`, `name`, `calculate`
  - `GARCHVolatility` (line 1983): Simplified GARCH(1,1) volatility with fixed parameters.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VolTermStructure` (line 2021): Ratio of short-term to long-term realized vol. >1 = backwardation (fear).
    - Methods (3): `__init__`, `name`, `calculate`
  - `HurstExponent` (line 2046): Hurst exponent via R/S analysis. H>0.5 trending, H<0.5 mean-reverting.
    - Methods (3): `__init__`, `name`, `calculate`
  - `MeanReversionHalfLife` (line 2106): Ornstein-Uhlenbeck half-life via OLS. Lower = faster mean reversion.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ZScore` (line 2144): Z-Score: standardized deviation from rolling mean.
    - Methods (3): `__init__`, `name`, `calculate`
  - `VarianceRatio` (line 2164): Lo-MacKinlay variance ratio. VR>1 = trending, VR<1 = mean-reverting.
    - Methods (3): `__init__`, `name`, `calculate`
  - `Autocorrelation` (line 2186): Serial correlation of returns at lag k. Positive = momentum, negative = mean-reversion.
    - Methods (3): `__init__`, `name`, `calculate`
  - `KalmanTrend` (line 2218): 1D Kalman filter for price trend extraction.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ShannonEntropy` (line 2266): Shannon entropy of return distribution. High = uncertain, low = predictable.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ApproximateEntropy` (line 2296): Approximate Entropy (ApEn). Low = regular/predictable, high = complex/random.
    - Methods (3): `__init__`, `name`, `calculate`
  - `AmihudIlliquidity` (line 2347): Amihud illiquidity ratio: |return| / dollar_volume. Higher = less liquid.
    - Methods (3): `__init__`, `name`, `calculate`
  - `KyleLambda` (line 2367): Kyle's lambda price impact coefficient via rolling regression.
    - Methods (3): `__init__`, `name`, `calculate`
  - `RollSpread` (line 2404): Roll's implied bid-ask spread in basis points.
    - Methods (3): `__init__`, `name`, `calculate`
  - `FractalDimension` (line 2430): Higuchi fractal dimension. D~1 = smooth/trending, D~2 = rough/noisy.
    - Methods (3): `__init__`, `name`, `calculate`
  - `DFA` (line 2486): Detrended Fluctuation Analysis. alpha>0.5 = persistent, alpha<0.5 = anti-persistent.
    - Methods (3): `__init__`, `name`, `calculate`
  - `DominantCycle` (line 2562): FFT-based dominant cycle period in bars.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ReturnSkewness` (line 2616): Rolling skewness of returns. Negative = left tail risk.
    - Methods (3): `__init__`, `name`, `calculate`
  - `ReturnKurtosis` (line 2633): Rolling excess kurtosis. High = fat tails (tail risk).
    - Methods (3): `__init__`, `name`, `calculate`
  - `CUSUMDetector` (line 2654): CUSUM change-point detection. Output = bars since last regime change / period.
    - Methods (3): `__init__`, `name`, `calculate`
  - `RegimePersistence` (line 2703): Consecutive bars in the same trend regime (price vs SMA).
    - Methods (3): `__init__`, `name`, `calculate`

## Package `kalshi`

### `kalshi/__init__.py`

- LOC: 58
- Module intent: Kalshi vertical for intraday event-market research.
- Imports (12): `from .client import KalshiClient`, `from .storage import EventTimeStore`, `from .provider import KalshiProvider`, `from .pipeline import KalshiPipeline`, `from .router import KalshiDataRouter`, `from .quality import QualityDimensions, StalePolicy`, `from .mapping_store import EventMarketMappingStore, EventMarketMappingRecord`, `from .distribution import DistributionConfig, build_distribution_panel`, `from .options import add_options_disagreement_features, build_options_reference_panel`, `from .promotion import EventPromotionConfig, evaluate_event_promotion`, `from .events import EventFeatureConfig, add_reference_disagreement_features, asof_join, build_asset_response_labels, build_event_feature_panel, build_event_labels`, `from .walkforward import EventWalkForwardConfig, EventWalkForwardResult, evaluate_event_contract_metrics, run_event_walkforward`
- Top-level functions: none
- Classes: none

### `kalshi/client.py`

- LOC: 655
- Module intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Imports (14): `from __future__ import annotations`, `import base64`, `import logging`, `import os`, `import subprocess`, `import tempfile`, `import threading`, `import time`, `from dataclasses import dataclass`, `from datetime import datetime, timezone`, `from typing import Callable, Dict, Iterable, List, Optional`, `from urllib.parse import urlparse` (+2 more)
- Top-level functions:
  - `_normalize_env(value)` (line 23): Internal helper for normalize env.
- Classes:
  - `RetryPolicy` (line 32): HTTP retry settings for Kalshi API requests.
  - `RateLimitPolicy` (line 40): Token-bucket rate-limit settings for Kalshi API access.
  - `RequestLimiter` (line 46): Lightweight token-bucket limiter with runtime limit updates.
    - Methods (5): `__init__`, `_refill`, `acquire`, `update_rate`, `update_from_account_limits`
  - `KalshiSigner` (line 126): Signs Kalshi requests using RSA-PSS SHA256.
    - Methods (7): `__init__`, `available`, `_canonical_path`, `_load_private_key`, `_sign_with_cryptography`, `_sign_with_openssl`, `sign`
  - `KalshiClient` (line 292): Kalshi HTTP wrapper with:
    - Methods (16): `__init__`, `available`, `_join_url`, `_auth_headers`, `_request_with_retries`, `_request`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`

### `kalshi/disagreement.py`

- LOC: 113
- Module intent: Cross-market disagreement engine for Kalshi event features.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `compute_disagreement(kalshi_entropy, kalshi_tail_mass, kalshi_variance, options_iv, options_skew, entropy_history, tail_history)` (line 32): Compute cross-market disagreement signals.
  - `disagreement_as_feature_dict(signals)` (line 104): Convert disagreement signals to a flat dict for feature panel merging.
- Classes:
  - `DisagreementSignals` (line 22): Container for cross-market disagreement features derived from Kalshi and options references.

### `kalshi/distribution.py`

- LOC: 935
- Module intent: Contract -> probability distribution builder for Kalshi markets.
- Imports (6): `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple`, `import numpy as np`, `import pandas as pd`, `from .quality import QualityDimensions, StalePolicy, compute_quality_dimensions, dynamic_stale_cutoff_minutes`
- Top-level functions:
  - `_is_tz_aware_datetime(series)` (line 22): Return whether tz aware datetime satisfies the expected condition.
  - `_to_utc_timestamp(value)` (line 63): Internal helper for to utc timestamp.
  - `_prob_from_mid(mid, price_scale)` (line 71): Internal helper for prob from mid.
  - `_entropy(p)` (line 85): Internal helper for entropy.
  - `_isotonic_nonincreasing(y)` (line 94): Pool-adjacent-violators for nonincreasing constraints.
  - `_isotonic_nondecreasing(y)` (line 118): Internal helper for isotonic nondecreasing.
  - `_resolve_threshold_direction(row)` (line 131): Resolve threshold contract semantics:
  - `_resolve_threshold_direction_with_confidence(row)` (line 141): Resolve threshold contract semantics with confidence scoring.
  - `_validate_bins(contracts)` (line 196): Validate bin structure for non-overlapping, ordered coverage.
  - `_tail_thresholds(event_type, config, support)` (line 257): Internal helper for tail thresholds.
  - `_latest_quotes_asof(quotes, asof_ts, stale_minutes)` (line 273): Internal helper for latest quotes asof.
  - `_normalize_mass(mass)` (line 293): Internal helper for normalize mass.
  - `_moments(support, mass)` (line 303): Internal helper for moments.
  - `_cdf_from_pmf(support, mass)` (line 320): Internal helper for cdf from pmf.
  - `_pmf_on_grid(support, mass, grid)` (line 331): Internal helper for pmf on grid.
  - `_distribution_distances(support_a, mass_a, support_b, mass_b)` (line 339): Internal helper for distribution distances.
  - `_tail_probs_from_mass(support, mass, thresholds)` (line 379): Internal helper for tail probs from mass.
  - `_tail_probs_from_threshold_curve(thresholds_x, prob_curve, direction, fixed_thresholds)` (line 398): Internal helper for tail probs from threshold curve.
  - `_estimate_liquidity_proxy(quotes, asof_ts)` (line 425): Estimate a stable liquidity proxy from recent quote stream.
  - `build_distribution_snapshot(contracts, quotes, asof_ts, config, event_ts, event_type)` (line 462): Build distribution snapshot.
  - `_lag_slug(lag)` (line 819): Internal helper for lag slug.
  - `_add_distance_features(panel, lags)` (line 824): Internal helper for add distance features.
  - `build_distribution_panel(markets, contracts, quotes, snapshot_times, config)` (line 868): Build market-level distribution snapshots across times.
- Classes:
  - `DistributionConfig` (line 29): Configuration for Kalshi contract-to-distribution reconstruction and snapshot feature extraction.
  - `DirectionResult` (line 124): Result of threshold direction resolution with confidence metadata.
  - `BinValidationResult` (line 188): Result of bin overlap/gap/ordering validation.

### `kalshi/events.py`

- LOC: 517
- Module intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from typing import Dict, Iterable, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_to_utc_ts(series)` (line 30): Internal helper for to utc ts.
  - `_ensure_asof_before_release(df)` (line 35): Internal helper for ensure asof before release.
  - `asof_join(left, right, by, left_ts_col, right_ts_col)` (line 44): Strict backward as-of join (no forward-peeking).
  - `build_event_snapshot_grid(macro_events, horizons)` (line 79): Build event snapshot grid.
  - `_merge_event_market_map(grid, event_market_map)` (line 112): Internal helper for merge event market map.
  - `_add_revision_speed_features(joined, feature_cols)` (line 148): Internal helper for add revision speed features.
  - `add_reference_disagreement_features(event_panel, reference_features, ts_col)` (line 186): Optional cross-market disagreement block via strict backward as-of join.
  - `build_event_feature_panel(macro_events, event_market_map, kalshi_distributions, config)` (line 228): Build event-centric panel indexed by (event_id, asof_ts).
  - `build_asset_time_feature_panel(asset_frame, kalshi_distributions, market_id, asof_col)` (line 325): Optional continuous panel keyed by (asset_id, ts) with strict as-of joins.
  - `build_event_labels(macro_events, event_outcomes_first_print, event_outcomes_revised, label_mode)` (line 359): Build event outcome labels with explicit as-of awareness.
  - `build_asset_response_labels(macro_events, asset_prices, ts_col, price_col, window_start, window_end, entry_delay, exit_horizon, price_source)` (line 444): Event-to-asset response labels with realistic execution windows.
- Classes:
  - `EventTimestampMeta` (line 14): Authoritative event timestamp metadata (D2).
  - `EventFeatureConfig` (line 22): Configuration for event snapshot horizons and event-panel quality filtering.

### `kalshi/mapping_store.py`

- LOC: 75
- Module intent: Versioned event-to-market mapping persistence.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Iterable, Optional`, `import pandas as pd`, `from .storage import EventTimeStore`
- Top-level functions: none
- Classes:
  - `EventMarketMappingRecord` (line 15): Versioned mapping row linking a macro event to a Kalshi market over an effective time window.
  - `EventMarketMappingStore` (line 26): Persistence helper for versioned event-to-market mappings stored in EventTimeStore.
    - Methods (5): `__init__`, `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`

### `kalshi/microstructure.py`

- LOC: 127
- Module intent: Market microstructure diagnostics for Kalshi event markets.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `compute_microstructure(quotes, window_hours, asof_ts)` (line 31): Compute microstructure diagnostics from a quote panel.
  - `microstructure_as_feature_dict(diag)` (line 117): Convert diagnostics to a flat dict for feature panel merging.
- Classes:
  - `MicrostructureDiagnostics` (line 20): Microstructure summary metrics computed from a quote panel window.

### `kalshi/options.py`

- LOC: 145
- Module intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Imports (5): `from __future__ import annotations`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`, `from ..features.options_factors import compute_option_surface_factors`
- Top-level functions:
  - `_to_utc_ts(series)` (line 17): Internal helper for to utc ts.
  - `build_options_reference_panel(options_frame, ts_col)` (line 22): Build a normalized options reference panel with:
  - `add_options_disagreement_features(event_panel, options_reference, options_ts_col)` (line 72): Strict backward as-of join of options reference features into event panel.
- Classes: none

### `kalshi/pipeline.py`

- LOC: 167
- Module intent: Orchestration helpers for the Kalshi event-market vertical.
- Imports (14): `from __future__ import annotations`, `from dataclasses import dataclass`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Optional`, `import pandas as pd`, `from .distribution import DistributionConfig`, `from .events import EventFeatureConfig, build_event_feature_panel`, `from .options import add_options_disagreement_features, build_options_reference_panel`, `from .promotion import EventPromotionConfig, evaluate_event_promotion`, `from .provider import KalshiProvider`, `from .storage import EventTimeStore` (+2 more)
- Top-level functions: none
- Classes:
  - `KalshiPipeline` (line 29): High-level orchestration wrapper for Kalshi sync, feature, walk-forward, and promotion workflows.
    - Methods (8): `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`

### `kalshi/promotion.py`

- LOC: 178
- Module intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Imports (9): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Dict, Iterable, Optional`, `import numpy as np`, `import pandas as pd`, `from ..autopilot.promotion_gate import PromotionDecision, PromotionGate`, `from ..autopilot.strategy_discovery import StrategyCandidate`, `from ..backtest.engine import BacktestResult`, `from .walkforward import EventWalkForwardResult, evaluate_event_contract_metrics`
- Top-level functions:
  - `_to_backtest_result(event_returns, horizon_days)` (line 33): Internal helper for to backtest result.
  - `evaluate_event_promotion(walkforward_result, config, gate, extra_contract_metrics)` (line 127): Evaluate Kalshi event strategy promotion from walk-forward outputs.
- Classes:
  - `EventPromotionConfig` (line 22): Strategy metadata used to evaluate an event strategy through the shared promotion gate.

### `kalshi/provider.py`

- LOC: 647
- Module intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Imports (12): `from __future__ import annotations`, `import hashlib`, `import json`, `from datetime import datetime, timezone`, `from typing import Dict, Iterable, List, Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import KALSHI_API_BASE_URL, KALSHI_DISTANCE_LAGS, KALSHI_ENV, KALSHI_FAR_EVENT_MINUTES, KALSHI_FAR_EVENT_STALE_MINUTES, KALSHI_HISTORICAL_API_BASE_URL, KALSHI_HISTORICAL_CUTOFF_TS, KALSHI_NEAR_EVENT_MINUTES, KALSHI_NEAR_EVENT_STALE_MINUTES, KALSHI_RATE_LIMIT_BURST, KALSHI_RATE_LIMIT_RPS, KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER, KALSHI_STALE_AFTER_MINUTES, KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD, KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD, KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER, KALSHI_STALE_MARKET_TYPE_MULTIPLIERS, KALSHI_TAIL_THRESHOLDS`, `from .client import KalshiClient, RateLimitPolicy`, `from .distribution import DistributionConfig, build_distribution_panel`, `from .mapping_store import EventMarketMappingStore`, `from .storage import EventTimeStore`
- Top-level functions:
  - `_to_iso_utc(value)` (line 40): Normalize a timestamp-like value to an ISO-8601 UTC string, returning None on failure.
  - `_safe_hash_text(text)` (line 55): Return a stable SHA-256 hash for text fields used in spec/provenance snapshots.
  - `_asof_date(value)` (line 60): Convert a timestamp-like value to a UTC calendar date string for daily rollups.
- Classes:
  - `KalshiProvider` (line 70): Provider interface similar to WRDSProvider, but for event-market data.
    - Methods (17): `__init__`, `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes`, `compute_and_store_distributions`, `materialize_daily_health_report`, `get_daily_health_report`, `store_clock_check`

### `kalshi/quality.py`

- LOC: 206
- Module intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Imports (4): `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from typing import Dict, Iterable, Optional`, `import numpy as np`
- Top-level functions:
  - `_finite(values)` (line 50): Return only finite numeric values from an iterable as a NumPy array.
  - `dynamic_stale_cutoff_minutes(time_to_event_minutes, policy, market_type, liquidity_proxy)` (line 56): Dynamic stale-cutoff schedule:
  - `compute_quality_dimensions(expected_contracts, observed_contracts, spreads, quote_ages_seconds, volumes, open_interests, violation_magnitude)` (line 104): Multi-dimensional quality model for distribution snapshots.
  - `passes_hard_gates(quality, stale_cutoff_seconds, min_coverage, max_median_spread)` (line 164): Hard validity gates (C1).  Must-pass criteria — failing any gate means
  - `quality_as_feature_dict(quality)` (line 193): Expose soft quality dimensions as separate learnable feature columns (C2).
- Classes:
  - `QualityDimensions` (line 16): Component-level quality metrics for a Kalshi distribution snapshot.
  - `StalePolicy` (line 27): Parameters controlling dynamic stale-quote cutoff schedules for Kalshi snapshots.

### `kalshi/regimes.py`

- LOC: 142
- Module intent: Regime tagging for Kalshi event strategies.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Dict, Optional, Sequence`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `classify_inflation_regime(cpi_yoy, high_threshold, low_threshold)` (line 28): Classify inflation regime from CPI year-over-year.
  - `classify_policy_regime(fed_funds_change_bps, tightening_threshold_bps, easing_threshold_bps)` (line 43): Classify monetary policy regime from Fed funds rate changes.
  - `classify_vol_regime(vix_level, high_threshold, low_threshold)` (line 58): Classify volatility regime from VIX level.
  - `tag_event_regime(cpi_yoy, fed_funds_change_bps, vix_level)` (line 73): Tag an event with macro regime classifications.
  - `evaluate_strategy_by_regime(event_returns, regime_tags)` (line 86): Evaluate strategy performance breakdown by regime.
  - `regime_stability_score(breakdown)` (line 120): Score strategy stability across regimes (0-1).
- Classes:
  - `EventRegimeTag` (line 21): Macro regime labels attached to an event for regime-stability analysis.

### `kalshi/router.py`

- LOC: 102
- Module intent: Routing helpers for live vs historical Kalshi endpoints.
- Imports (5): `from __future__ import annotations`, `from dataclasses import dataclass`, `from typing import Dict, Optional`, `from urllib.parse import urlparse`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `RouteDecision` (line 14): Resolved endpoint route decision (base URL, path, and historical/live choice).
  - `KalshiDataRouter` (line 21): Chooses live vs historical endpoint roots by cutoff timestamp.
    - Methods (6): `__init__`, `_to_utc_ts`, `update_cutoff`, `_extract_end_ts`, `_clean_path`, `resolve`

### `kalshi/storage.py`

- LOC: 649
- Module intent: Event-time storage layer for Kalshi + macro event research.
- Imports (7): `from __future__ import annotations`, `import json`, `import sqlite3`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Iterable, List, Mapping, Optional, Sequence`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `EventTimeStore` (line 35): Intraday/event-time storage with a stable schema.
    - Methods (27): `__init__`, `_execute`, `_executemany`, `_table_columns`, `_norm_ts`, `_clean_value`, `_insert_or_replace`, `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs`, `upsert_data_provenance` (+7 more)

### `kalshi/tests/__init__.py`

- LOC: 1
- Module intent: Kalshi package-local tests.
- Top-level functions: none
- Classes: none

### `kalshi/tests/test_bin_validity.py`

- LOC: 105
- Module intent: Bin overlap/gap detection test (Instructions I.3).
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.kalshi.distribution import _validate_bins`
- Top-level functions: none
- Classes:
  - `BinValidityTests` (line 13): Tests for bin overlap/gap/ordering validation.
    - Methods (9): `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`

### `kalshi/tests/test_distribution.py`

- LOC: 41
- Module intent: Kalshi test module for distribution behavior and regressions.
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.kalshi.distribution import DistributionConfig, build_distribution_snapshot`
- Top-level functions: none
- Classes:
  - `DistributionLocalTests` (line 12): Test cases covering Kalshi subsystem behavior and safety constraints.
    - Methods (1): `test_bin_distribution_probability_mass_is_normalized`

### `kalshi/tests/test_leakage.py`

- LOC: 46
- Module intent: Kalshi test module for leakage behavior and regressions.
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel`
- Top-level functions: none
- Classes:
  - `LeakageLocalTests` (line 12): Test cases covering Kalshi subsystem behavior and safety constraints.
    - Methods (1): `test_feature_rows_strictly_pre_release`

### `kalshi/tests/test_no_leakage.py`

- LOC: 117
- Module intent: No-leakage test at panel level (Instructions I.4).
- Imports (4): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel`
- Top-level functions: none
- Classes:
  - `NoLeakageTests` (line 16): Panel-level look-ahead bias detection.
    - Methods (3): `_build_synthetic_panel`, `test_all_asof_before_release`, `test_single_event_no_leakage`

### `kalshi/tests/test_signature_kat.py`

- LOC: 141
- Module intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Imports (3): `import base64`, `import unittest`, `from quant_engine.kalshi.client import KalshiSigner`
- Top-level functions: none
- Classes:
  - `SignatureKATTests` (line 46): Known-answer tests for Kalshi request signing.
    - Methods (5): `_skip_if_no_crypto`, `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`

### `kalshi/tests/test_stale_quotes.py`

- LOC: 152
- Module intent: Stale quote cutoff test (Instructions I.5).
- Imports (2): `import unittest`, `from quant_engine.kalshi.quality import StalePolicy, dynamic_stale_cutoff_minutes`
- Top-level functions: none
- Classes:
  - `StaleQuoteCutoffTests` (line 15): Tests for dynamic stale-cutoff schedule.
    - Methods (10): `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`

### `kalshi/tests/test_threshold_direction.py`

- LOC: 126
- Module intent: Threshold direction correctness test (Instructions I.2).
- Imports (2): `import unittest`, `from quant_engine.kalshi.distribution import _resolve_threshold_direction, _resolve_threshold_direction_with_confidence`
- Top-level functions: none
- Classes:
  - `ThresholdDirectionTests` (line 15): Tests for threshold direction resolution.
    - Methods (17): `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below`, `test_title_guess_or_higher`, `test_title_guess_or_lower`, `test_no_direction_signal`, `test_empty_row`, `test_legacy_resolve_returns_string`

### `kalshi/tests/test_walkforward_purge.py`

- LOC: 159
- Module intent: Walk-forward purge/embargo test (Instructions I.6).
- Imports (4): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.kalshi.walkforward import EventWalkForwardConfig, _prepare_panel, run_event_walkforward`
- Top-level functions: none
- Classes:
  - `WalkForwardPurgeTests` (line 18): Tests that walk-forward purge/embargo prevents data leakage.
    - Methods (5): `_build_synthetic_data`, `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`

### `kalshi/walkforward.py`

- LOC: 492
- Module intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Imports (6): `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from typing import Any, Dict, List, Optional, Sequence`, `import numpy as np`, `import pandas as pd`, `from ..backtest.advanced_validation import deflated_sharpe_ratio, monte_carlo_validation`
- Top-level functions:
  - `_bootstrap_mean_ci(values, n_bootstrap, random_seed)` (line 103): Estimate a bootstrap mean and 95% confidence interval for event returns.
  - `_event_regime_stability(event_returns, event_types)` (line 118): Score return consistency across event types on a 0-1 stability scale.
  - `evaluate_event_contract_metrics(result, n_bootstrap, max_events_per_day)` (line 147): Advanced validation contract metrics for event strategies:
  - `_corr(a, b)` (line 238): Compute a finite-sample-safe correlation between two numeric arrays.
  - `_fit_ridge(X, y, alpha)` (line 252): Fit a simple ridge regression coefficient vector for event walk-forward evaluation.
  - `_predict(X, beta)` (line 262): Apply a fitted linear coefficient vector to a feature matrix.
  - `_prepare_panel(panel, labels, label_col)` (line 267): Normalize and merge feature panel and labels into a walk-forward-ready event dataset.
  - `run_event_walkforward(panel, labels, config, label_col)` (line 315): Run purge/embargo-aware walk-forward evaluation on an event feature panel.
- Classes:
  - `EventWalkForwardConfig` (line 16): Configuration for event-level walk-forward splits, purge/embargo, and trial accounting.
  - `EventWalkForwardFold` (line 36): Per-fold event walk-forward metrics for fit quality and event-return diagnostics.
  - `EventWalkForwardResult` (line 52): Aggregate event walk-forward outputs and OOS traces used in promotion checks.
    - Methods (5): `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`

## Package `models`

### `models/__init__.py`

- LOC: 20
- Module intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- Imports (6): `from .governance import ModelGovernance, ChampionRecord`, `from .cross_sectional import cross_sectional_rank`, `from .calibration import ConfidenceCalibrator`, `from .neural_net import TabularNet`, `from .walk_forward import walk_forward_select`, `from .feature_stability import FeatureStabilityTracker`
- Top-level functions: none
- Classes: none

### `models/calibration.py`

- LOC: 327
- Module intent: Confidence Calibration --- Platt scaling and isotonic regression.
- Imports (3): `from __future__ import annotations`, `from typing import Optional`, `import numpy as np`
- Top-level functions:
  - `compute_ece(predicted_probs, actual_outcomes, n_bins)` (line 228): Compute Expected Calibration Error.
  - `compute_reliability_curve(predicted_probs, actual_outcomes, n_bins)` (line 277): Compute reliability curve data for calibration diagnostics.
- Classes:
  - `_LinearRescaler` (line 41): Maps raw scores to [0, 1] via min-max linear rescaling.
    - Methods (3): `__init__`, `fit`, `transform`
  - `ConfidenceCalibrator` (line 69): Post-hoc confidence calibration via Platt scaling or isotonic regression.
    - Methods (8): `__init__`, `fit`, `_fit_sklearn`, `transform`, `fit_transform`, `is_fitted`, `backend`, `__repr__`

### `models/cross_sectional.py`

- LOC: 136
- Module intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- Imports (3): `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `cross_sectional_rank(predictions, date_col, prediction_col, asset_col, long_quantile, short_quantile)` (line 18): Rank stocks cross-sectionally by predicted return at each date.
- Classes: none

### `models/feature_stability.py`

- LOC: 313
- Module intent: Feature Stability Monitoring — tracks feature importance rankings across
- Imports (7): `import json`, `import time`, `from dataclasses import dataclass, field, asdict`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `from ..config import RESULTS_DIR`
- Top-level functions: none
- Classes:
  - `StabilityReport` (line 47): Summary returned by :meth:`FeatureStabilityTracker.check_stability`.
    - Methods (1): `to_dict`
  - `FeatureStabilityTracker` (line 69): Record and compare feature importance rankings over training cycles.
    - Methods (6): `__init__`, `_load`, `_save`, `record_importance`, `_spearman_rank_correlation`, `check_stability`

### `models/governance.py`

- LOC: 155
- Module intent: Champion/challenger governance for model versions.
- Imports (7): `import json`, `from dataclasses import dataclass, asdict`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Dict, Optional`, `from ..config import CHAMPION_REGISTRY`, `from ..config import GOVERNANCE_SCORE_WEIGHTS`
- Top-level functions: none
- Classes:
  - `ChampionRecord` (line 15): Persisted champion model record for a prediction horizon.
    - Methods (1): `to_dict`
  - `ModelGovernance` (line 28): Maintains champion model per horizon and promotes challengers if better.
    - Methods (6): `__init__`, `_load`, `_save`, `_score`, `get_champion_version`, `evaluate_and_update`

### `models/iv/__init__.py`

- LOC: 31
- Module intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- Imports (1): `from .models import ArbitrageFreeSVIBuilder, BlackScholes, HestonModel, SVIModel, IVSurface, IVPoint, OptionType, Greeks, HestonParams, SVIParams`
- Top-level functions: none
- Classes: none

### `models/iv/models.py`

- LOC: 937
- Module intent: Implied Volatility Surface Models.
- Imports (5): `from dataclasses import dataclass, field`, `from enum import Enum`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `from scipy import optimize, integrate, interpolate, stats`
- Top-level functions:
  - `generate_synthetic_market_surface(S, r, q)` (line 655): Generate a realistic synthetic market IV surface for demonstration.
- Classes:
  - `OptionType` (line 19): Supported option contract types for pricing and volatility surface models.
  - `Greeks` (line 26): Option Greeks container.
  - `HestonParams` (line 36): Heston model parameters.
    - Methods (1): `validate`
  - `SVIParams` (line 50): Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).
  - `BlackScholes` (line 59): Black-Scholes option pricing and analytics.
    - Methods (4): `price`, `greeks`, `implied_vol`, `iv_surface`
  - `HestonModel` (line 139): Heston (1993) stochastic volatility model.
    - Methods (6): `__init__`, `characteristic_function`, `price`, `implied_vol`, `iv_surface`, `calibrate`
  - `SVIModel` (line 295): SVI (Stochastic Volatility Inspired) implied variance parameterization.
    - Methods (7): `__init__`, `total_variance`, `implied_vol`, `iv_surface`, `smile`, `calibrate`, `check_no_butterfly_arbitrage`
  - `ArbitrageFreeSVIBuilder` (line 397): Arbitrage-aware SVI surface builder.
    - Methods (9): `__init__`, `_svi_total_variance`, `_initial_guess`, `_vega_spread_weights`, `_slice_objective`, `fit_slice`, `enforce_calendar_monotonicity`, `interpolate_total_variance`, `build_surface`
  - `IVPoint` (line 688): Single implied-volatility observation.
  - `IVSurface` (line 698): Store and interpolate an implied-volatility surface.
    - Methods (11): `__init__`, `add_point`, `add_slice`, `add_surface`, `n_points`, `_log_moneyness`, `_build_interpolator`, `get_iv`, `get_smile`, `decompose`, `decompose_surface`

### `models/neural_net.py`

- LOC: 198
- Module intent: Tabular Neural Network — feedforward network for tabular financial data.
- Imports (2): `from typing import List, Optional`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `TabularNet` (line 29): Feedforward network for tabular financial data.
    - Methods (5): `__init__`, `_build_model`, `fit`, `predict`, `feature_importances_`

### `models/online_learning.py`

- LOC: 273
- Module intent: Online learning module for incremental model updates between full retrains.
- Imports (8): `from __future__ import annotations`, `import json`, `import logging`, `from dataclasses import dataclass, field`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `OnlineUpdate` (line 23): Record of a single online update step.
  - `OnlineLearner` (line 34): Incremental model updater between full retrains.
    - Methods (8): `__init__`, `add_sample`, `update`, `adjust_prediction`, `should_retrain`, `get_status`, `_save_state`, `load_state`

### `models/predictor.py`

- LOC: 404
- Module intent: Model Predictor — loads trained ensemble and generates predictions.
- Imports (9): `import json`, `from pathlib import Path`, `from typing import Optional, Dict`, `import joblib`, `import numpy as np`, `import pandas as pd`, `from ..config import MODEL_DIR, REGIME_NAMES`, `from .governance import ModelGovernance`, `from .versioning import ModelRegistry`
- Top-level functions:
  - `_prepare_features(raw, expected, medians)` (line 24): Align, impute, and return features matching expected column order.
- Classes:
  - `EnsemblePredictor` (line 45): Loads a trained regime-conditional ensemble and generates predictions.
    - Methods (6): `__init__`, `_resolve_model_dir`, `_load`, `predict`, `_calibrate_confidence`, `predict_single`

### `models/retrain_trigger.py`

- LOC: 296
- Module intent: ML Retraining Trigger Logic
- Imports (6): `import json`, `import os`, `from datetime import datetime, timedelta`, `from typing import Tuple, List, Dict, Optional`, `import numpy as np`, `from ..config import MODEL_DIR`
- Top-level functions: none
- Classes:
  - `RetrainTrigger` (line 37): Determines when ML model should be retrained.
    - Methods (7): `__init__`, `_load_metadata`, `_save_metadata`, `add_trade_result`, `check`, `record_retraining`, `status`

### `models/trainer.py`

- LOC: 1598
- Module intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Imports (11): `import json`, `import time`, `from dataclasses import dataclass, field`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import joblib`, `import numpy as np`, `import pandas as pd`, `from ..config import MODEL_PARAMS, MAX_FEATURES_SELECTED, MAX_IS_OOS_GAP, CV_FOLDS, HOLDOUT_FRACTION, MODEL_DIR, MIN_REGIME_SAMPLES, REGIME_NAMES, RECENCY_DECAY, REGIME_SOFT_ASSIGNMENT_THRESHOLD, ENSEMBLE_DIVERSIFY, WF_MAX_TRAIN_DATES`, `from .feature_stability import FeatureStabilityTracker`, `from .versioning import ModelVersion, ModelRegistry`
- Top-level functions: none
- Classes:
  - `IdentityScaler` (line 87): No-op scaler that passes data through unchanged.
    - Methods (4): `fit`, `transform`, `fit_transform`, `inverse_transform`
  - `DiverseEnsemble` (line 112): Lightweight ensemble wrapper that combines predictions from multiple
    - Methods (3): `__init__`, `_aggregate_feature_importances`, `predict`
  - `TrainResult` (line 171): Result of training a single model.
  - `EnsembleResult` (line 190): Result of training the full regime-conditional ensemble.
  - `ModelTrainer` (line 200): Trains a regime-conditional gradient boosting ensemble for
    - Methods (19): `__init__`, `_spearmanr`, `_require_sklearn`, `_extract_dates`, `_sort_panel_by_time`, `_temporal_holdout_masks`, `_date_purged_folds`, `_prune_correlated_features`, `_select_features`, `_compute_stable_features`, `train_ensemble`, `_train_single`, `_train_diverse_ensemble`, `_optimize_ensemble_weights`, `_clone_model`, `_make_model`, `_save`, `_fit_calibrator`, `_print_summary`

### `models/versioning.py`

- LOC: 207
- Module intent: Model Versioning — timestamped model directories with registry.
- Imports (7): `import json`, `import shutil`, `from dataclasses import dataclass, field, asdict`, `from datetime import datetime`, `from pathlib import Path`, `from typing import Dict, List, Optional`, `from ..config import MODEL_DIR, MAX_MODEL_VERSIONS`
- Top-level functions: none
- Classes:
  - `ModelVersion` (line 26): Metadata for a single model version.
    - Methods (2): `to_dict`, `from_dict`
  - `ModelRegistry` (line 60): Manages versioned model storage and retrieval.
    - Methods (14): `__init__`, `_load_registry`, `_save_registry`, `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`

### `models/walk_forward.py`

- LOC: 235
- Module intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Imports (4): `from itertools import product`, `from typing import Any, Dict, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_spearmanr(x, y)` (line 24): Spearman rank correlation (scipy optional).
  - `_expanding_walk_forward_folds(dates, n_folds, horizon)` (line 45): Generate expanding-window walk-forward folds using unique dates.
  - `_extract_dates(index)` (line 84): Return row-aligned timestamps from an index (supports panel MultiIndex).
  - `walk_forward_select(features, targets, regimes, param_grid, n_folds, horizon)` (line 93): Select the best model configuration via walk-forward cross-validation.
- Classes: none

## Package `regime`

### `regime/__init__.py`

- LOC: 17
- Module intent: Regime modeling components.
- Imports (4): `from .correlation import CorrelationRegimeDetector`, `from .detector import RegimeDetector, RegimeOutput, detect_regimes_batch`, `from .hmm import GaussianHMM, HMMFitResult`, `from .jump_model import StatisticalJumpModel, JumpModelResult`
- Top-level functions: none
- Classes: none

### `regime/correlation.py`

- LOC: 213
- Module intent: Correlation Regime Detection (NEW 11).
- Imports (4): `from __future__ import annotations`, `from typing import Dict, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `CorrelationRegimeDetector` (line 29): Detect regime changes in pairwise correlation structure.
    - Methods (4): `__init__`, `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`

### `regime/detector.py`

- LOC: 586
- Module intent: Regime detector with two engines:
- Imports (8): `from dataclasses import dataclass`, `from typing import Dict, Optional, Tuple`, `import logging`, `import numpy as np`, `import pandas as pd`, `from ..config import REGIME_MODEL_TYPE, REGIME_HMM_STATES, REGIME_HMM_MAX_ITER, REGIME_HMM_STICKINESS, REGIME_MIN_DURATION, REGIME_HMM_AUTO_SELECT_STATES, REGIME_HMM_MIN_STATES, REGIME_HMM_MAX_STATES, REGIME_JUMP_MODEL_ENABLED, REGIME_JUMP_PENALTY, REGIME_EXPECTED_CHANGES_PER_YEAR, REGIME_ENSEMBLE_ENABLED, REGIME_ENSEMBLE_CONSENSUS_THRESHOLD`, `from .hmm import GaussianHMM, build_hmm_observation_matrix, map_raw_states_to_regimes, select_hmm_states_bic`, `from .jump_model import StatisticalJumpModel`
- Top-level functions:
  - `detect_regimes_batch(features_by_id, detector, verbose)` (line 546): Shared regime detection across multiple PERMNOs.
- Classes:
  - `RegimeOutput` (line 41): Unified regime detection output consumed by modeling, backtesting, and UI layers.
  - `RegimeDetector` (line 51): Classifies market regime at each bar using either rules or HMM.
    - Methods (13): `__init__`, `detect`, `_apply_min_duration`, `_rule_detect`, `_hmm_detect`, `_jump_detect`, `detect_ensemble`, `detect_with_confidence`, `detect_full`, `regime_features`, `get_regime_uncertainty`, `map_raw_states_to_regimes_stable`, `_get_col`

### `regime/hmm.py`

- LOC: 566
- Module intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Imports (4): `from dataclasses import dataclass`, `from typing import Dict, Optional, Tuple`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_logsumexp(a, axis)` (line 13): Internal helper for logsumexp.
  - `select_hmm_states_bic(X, min_states, max_states)` (line 359): Select the optimal number of HMM states using the Bayesian Information Criterion.
  - `build_hmm_observation_matrix(features)` (line 438): Build an expanded observation matrix for regime inference.
  - `map_raw_states_to_regimes(raw_states, features)` (line 517): Map unlabeled HMM states -> semantic regimes used by the system.
- Classes:
  - `HMMFitResult` (line 24): Fitted HMM outputs including decoded states, posteriors, transitions, and log-likelihood.
  - `GaussianHMM` (line 32): Gaussian HMM using EM (Baum-Welch).
    - Methods (9): `__init__`, `_ensure_positive_definite`, `_init_params`, `_log_emission`, `_forward_backward`, `viterbi`, `_smooth_duration`, `fit`, `predict_proba`

### `regime/jump_model.py`

- LOC: 242
- Module intent: Statistical Jump Model for regime detection.
- Imports (5): `from __future__ import annotations`, `import logging`, `from dataclasses import dataclass`, `from typing import Optional`, `import numpy as np`
- Top-level functions: none
- Classes:
  - `JumpModelResult` (line 25): Result from fitting a Statistical Jump Model.
  - `StatisticalJumpModel` (line 36): Statistical Jump Model for regime detection.
    - Methods (7): `__init__`, `fit`, `_kmeans_init`, `_dp_segment`, `_compute_probs`, `compute_jump_penalty_from_data`, `predict`

## Package `risk`

### `risk/__init__.py`

- LOC: 42
- Module intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- Imports (10): `from .position_sizer import PositionSizer`, `from .portfolio_risk import PortfolioRiskManager`, `from .drawdown import DrawdownController`, `from .metrics import RiskMetrics`, `from .stop_loss import StopLossManager`, `from .covariance import CovarianceEstimator, compute_regime_covariance, get_regime_covariance`, `from .factor_portfolio import compute_factor_exposures, compute_residual_returns`, `from .portfolio_optimizer import optimize_portfolio`, `from .attribution import decompose_returns, compute_attribution_report`, `from .stress_test import run_stress_scenarios, run_historical_drawdown_test`
- Top-level functions: none
- Classes: none

### `risk/attribution.py`

- LOC: 266
- Module intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- Imports (4): `from __future__ import annotations`, `from typing import Dict, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_estimate_beta(portfolio_returns, benchmark_returns)` (line 26): OLS beta of portfolio vs benchmark.
  - `_estimate_factor_loadings(portfolio_returns, benchmark_returns, factor_returns)` (line 49): Multivariate OLS regression of excess returns on factor returns.
  - `decompose_returns(portfolio_returns, benchmark_returns, factor_returns)` (line 87): Decompose portfolio returns into market, factor, and alpha components.
  - `compute_attribution_report(portfolio_returns, benchmark_returns, factor_returns, annual_trading_days)` (line 178): Produce an extended attribution summary with risk-adjusted metrics.
- Classes: none

### `risk/covariance.py`

- LOC: 249
- Module intent: Covariance estimation utilities for portfolio risk controls.
- Imports (4): `from dataclasses import dataclass`, `from typing import Dict`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `compute_regime_covariance(returns, regimes, min_obs, shrinkage)` (line 124): Compute separate covariance matrices for each market regime.
  - `get_regime_covariance(regime_covs, current_regime)` (line 211): Return the covariance matrix for *current_regime*.
- Classes:
  - `CovarianceEstimate` (line 12): Covariance estimation output bundle with metadata about the fit method and sample count.
  - `CovarianceEstimator` (line 19): Estimate a robust covariance matrix for asset returns.
    - Methods (4): `__init__`, `estimate`, `portfolio_volatility`, `_estimate_values`

### `risk/drawdown.py`

- LOC: 240
- Module intent: Drawdown Controller — circuit breakers and recovery protocols.
- Imports (6): `from dataclasses import dataclass, field`, `from enum import Enum`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import DRAWDOWN_WARNING_THRESHOLD, DRAWDOWN_CAUTION_THRESHOLD, DRAWDOWN_CRITICAL_THRESHOLD, DRAWDOWN_DAILY_LOSS_LIMIT, DRAWDOWN_WEEKLY_LOSS_LIMIT, DRAWDOWN_RECOVERY_DAYS, DRAWDOWN_SIZE_MULT_WARNING, DRAWDOWN_SIZE_MULT_CAUTION`
- Top-level functions: none
- Classes:
  - `DrawdownState` (line 24): Discrete drawdown-control states used by the drawdown controller.
  - `DrawdownStatus` (line 34): Current drawdown state and action directives.
  - `DrawdownController` (line 49): Multi-tier drawdown protection with circuit breakers.
    - Methods (5): `__init__`, `update`, `_compute_actions`, `reset`, `get_summary`

### `risk/factor_portfolio.py`

- LOC: 220
- Module intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- Imports (3): `from typing import Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `compute_factor_exposures(returns, factor_returns)` (line 18): Estimate factor betas for each asset via OLS regression.
  - `compute_residual_returns(returns, factor_returns, factor_betas)` (line 135): Strip out systematic factor exposure, returning idiosyncratic returns.
- Classes: none

### `risk/metrics.py`

- LOC: 253
- Module intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- Imports (4): `from dataclasses import dataclass, field`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `RiskReport` (line 15): Comprehensive risk metrics report.
  - `RiskMetrics` (line 49): Computes comprehensive risk metrics from trade returns and equity curves.
    - Methods (7): `__init__`, `compute_full_report`, `_drawdown_analytics`, `_drawdown_analytics_array`, `_compute_mae_mfe`, `_empty_report`, `print_report`

### `risk/portfolio_optimizer.py`

- LOC: 276
- Module intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- Imports (5): `import logging`, `from typing import Dict, Optional`, `import numpy as np`, `import pandas as pd`, `from scipy.optimize import minimize`
- Top-level functions:
  - `optimize_portfolio(expected_returns, covariance, current_weights, max_position, max_portfolio_vol, turnover_penalty, risk_aversion, sector_map, max_sector_exposure)` (line 27): Find optimal portfolio weights via mean-variance optimization.
- Classes: none

### `risk/portfolio_risk.py`

- LOC: 329
- Module intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- Imports (6): `from dataclasses import dataclass, field`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import MAX_PORTFOLIO_VOL`, `from .covariance import CovarianceEstimator`
- Top-level functions: none
- Classes:
  - `RiskCheck` (line 51): Result of a portfolio risk check.
  - `PortfolioRiskManager` (line 58): Enforces portfolio-level risk constraints.
    - Methods (8): `__init__`, `_infer_ticker_from_price_df`, `_resolve_sector`, `check_new_position`, `_check_correlations`, `_estimate_portfolio_beta`, `_estimate_portfolio_vol`, `portfolio_summary`

### `risk/position_sizer.py`

- LOC: 564
- Module intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Imports (5): `import math`, `from dataclasses import dataclass, field`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `PositionSize` (line 22): Result of position sizing calculation.
  - `PositionSizer` (line 35): Multi-method position sizer with conservative blending.
    - Methods (11): `__init__`, `size_position`, `_kelly`, `_vol_scaled`, `_atr_based`, `_apply_drawdown_governor`, `update_regime_stats`, `update_kelly_bayesian`, `get_bayesian_kelly`, `size_portfolio_aware`, `size_portfolio`

### `risk/stop_loss.py`

- LOC: 255
- Module intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Imports (7): `from dataclasses import dataclass`, `from enum import Enum`, `from typing import Optional`, `import numpy as np`, `import pandas as pd`, `from ..config import REGIME_STOP_MULTIPLIER`, `from ..config import HARD_STOP_PCT, ATR_STOP_MULTIPLIER, TRAILING_ATR_MULTIPLIER, TRAILING_ACTIVATION_PCT, MAX_HOLDING_DAYS`
- Top-level functions: none
- Classes:
  - `StopReason` (line 28): Enumerated reasons a stop-loss evaluation can trigger an exit.
  - `StopResult` (line 40): Result of stop-loss evaluation.
  - `StopLossManager` (line 51): Multi-strategy stop-loss manager.
    - Methods (4): `__init__`, `evaluate`, `compute_initial_stop`, `compute_risk_per_share`

### `risk/stress_test.py`

- LOC: 363
- Module intent: Stress Testing Module --- scenario analysis and historical drawdown replay.
- Imports (4): `from __future__ import annotations`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_estimate_portfolio_beta(portfolio_weights, returns_history, min_obs)` (line 66): Estimate weighted-average beta of the portfolio vs equal-weight market proxy.
  - `_compute_portfolio_vol(portfolio_weights, returns_history, annual_trading_days)` (line 111): Annualized portfolio volatility from historical covariance.
  - `run_stress_scenarios(portfolio_weights, returns_history, scenarios)` (line 136): Apply stress scenarios to a portfolio and estimate impact.
  - `run_historical_drawdown_test(portfolio_weights, returns_history, n_worst, min_drawdown_pct)` (line 207): Replay the worst historical drawdown episodes on the portfolio.
  - `_find_drawdown_episodes(returns, min_drawdown_pct)` (line 303): Identify non-overlapping drawdown episodes from a return series.
- Classes: none

## Package `scripts`

### `scripts/generate_types.py`

- LOC: 146
- Module intent: Generate TypeScript interfaces from Pydantic schemas.
- Imports (7): `from __future__ import annotations`, `import importlib`, `import sys`, `from datetime import datetime`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional, Set, get_args, get_origin`, `from pydantic import BaseModel`
- Top-level functions:
  - `python_type_to_ts(annotation, seen)` (line 31): Convert a Python type annotation to a TypeScript type string.
  - `model_to_ts(model)` (line 100): Convert a Pydantic model class to a TypeScript interface.
  - `main()` (line 113): No module docstring.
- Classes: none

### `scripts/ibkr_daily_gapfill.py`

- LOC: 417
- Module intent: IBKR Daily Gap-Fill Downloader for quant_engine cache.
- Imports (13): `import argparse`, `import json`, `import os`, `import sys`, `import time`, `from datetime import datetime, timedelta, timezone`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import pandas as pd`, `import numpy as np`, `import asyncio`, `from quant_engine.config import UNIVERSE_FULL` (+1 more)
- Top-level functions:
  - `survey_cache(cache_dir)` (line 64): Survey the cache to find stale and missing tickers.
  - `download_daily_ibkr(ib, ticker, duration)` (line 120): Download daily OHLCV data for a single ticker from IBKR.
  - `merge_and_save(ticker, existing_path, ibkr_df, cache_dir)` (line 182): Merge IBKR data with existing WRDS data and save.
  - `main()` (line 269): No module docstring.
- Classes: none

### `scripts/ibkr_intraday_download.py`

- LOC: 493
- Module intent: IBKR Intraday Data Downloader for quant_engine cache.
- Imports (11): `import argparse`, `import json`, `import sys`, `import time`, `from datetime import datetime, timedelta, timezone`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import pandas as pd`, `import numpy as np`, `import asyncio`, `import importlib.util as _ilu`
- Top-level functions:
  - `survey_intraday(cache_dir, tickers, timeframes)` (line 127): Survey intraday cache for given tickers/timeframes.
  - `download_intraday_chunked(ib, ticker, timeframe, target_days)` (line 172): Download intraday data for a single ticker using IBKR chunked requests.
  - `save_intraday(ticker, timeframe, df, cache_dir, existing_path)` (line 263): Save intraday data to cache. If existing data exists, merge (keep both,
  - `main()` (line 322): No module docstring.
- Classes: none

## Package `tests`

### `tests/__init__.py`

- LOC: 6
- Module intent: Project-level test suite package.
- Top-level functions: none
- Classes: none

### `tests/api/__init__.py`

- LOC: 0
- Module intent: No module docstring.
- Top-level functions: none
- Classes: none

### `tests/api/test_compute_routers.py`

- LOC: 65
- Module intent: Tests for POST compute endpoints — verify job creation.
- Imports (1): `import pytest`
- Top-level functions:
  - `async test_train_creates_job(client)` (line 6): No module docstring.
  - `async test_predict_creates_job(client)` (line 17): No module docstring.
  - `async test_backtest_creates_job(client)` (line 27): No module docstring.
  - `async test_autopilot_creates_job(client)` (line 37): No module docstring.
  - `async test_job_status_queryable(client)` (line 47): No module docstring.
  - `async test_nonexistent_job_404(client)` (line 61): No module docstring.
- Classes: none

### `tests/api/test_envelope.py`

- LOC: 51
- Module intent: Tests for ApiResponse envelope and ResponseMeta.
- Imports (1): `from quant_engine.api.schemas.envelope import ApiResponse, ResponseMeta`
- Top-level functions:
  - `test_success_response()` (line 5): No module docstring.
  - `test_fail_response()` (line 13): No module docstring.
  - `test_from_cached()` (line 20): No module docstring.
  - `test_meta_defaults()` (line 27): No module docstring.
  - `test_meta_custom_fields()` (line 35): No module docstring.
  - `test_serialization_roundtrip()` (line 46): No module docstring.
- Classes: none

### `tests/api/test_integration.py`

- LOC: 72
- Module intent: Integration tests — full app startup, envelope consistency, config patch.
- Imports (1): `import pytest`
- Top-level functions:
  - `async test_all_gets_return_envelope(client)` (line 6): Every GET endpoint should return an ApiResponse envelope.
  - `async test_config_patch_and_read(client)` (line 34): PATCH /api/config should update values and be reflected in GET.
  - `async test_config_patch_invalid_key(client)` (line 58): PATCH with invalid key should return 422.
  - `async test_meta_has_generated_at(client)` (line 67): Response meta should always include generated_at timestamp.
- Classes: none

### `tests/api/test_jobs.py`

- LOC: 135
- Module intent: Tests for the job system: store, runner, lifecycle.
- Imports (5): `import asyncio`, `import pytest`, `from quant_engine.api.jobs.models import JobRecord, JobStatus`, `from quant_engine.api.jobs.store import JobStore`, `from quant_engine.api.jobs.runner import JobRunner`
- Top-level functions:
  - `async store()` (line 12): No module docstring.
  - `async runner(store)` (line 20): No module docstring.
  - `async test_create_and_get_job(store)` (line 25): No module docstring.
  - `async test_list_jobs(store)` (line 37): No module docstring.
  - `async test_update_status(store)` (line 45): No module docstring.
  - `async test_update_progress(store)` (line 54): No module docstring.
  - `async test_cancel_queued_job(store)` (line 63): No module docstring.
  - `async test_cancel_completed_job_fails(store)` (line 72): No module docstring.
  - `async test_get_nonexistent_job(store)` (line 80): No module docstring.
  - `async test_job_runner_succeeds(store, runner)` (line 86): No module docstring.
  - `async test_job_runner_failure(store, runner)` (line 102): No module docstring.
  - `async test_sse_events(store, runner)` (line 115): No module docstring.
- Classes: none

### `tests/api/test_main.py`

- LOC: 48
- Module intent: Tests for app factory and basic middleware.
- Imports (3): `import pytest`, `from quant_engine.api.main import create_app`, `from quant_engine.api.config import ApiSettings`
- Top-level functions:
  - `test_create_app()` (line 7): No module docstring.
  - `test_openapi_schema()` (line 12): No module docstring.
  - `test_routes_registered()` (line 19): No module docstring.
  - `async test_404_wrapped(client)` (line 36): No module docstring.
  - `async test_cors_headers(client)` (line 42): No module docstring.
- Classes: none

### `tests/api/test_read_routers.py`

- LOC: 118
- Module intent: Tests for all GET endpoints — verify ApiResponse envelope.
- Imports (1): `import pytest`
- Top-level functions:
  - `async test_health(client)` (line 6): No module docstring.
  - `async test_dashboard_summary(client)` (line 16): No module docstring.
  - `async test_data_universe(client)` (line 24): No module docstring.
  - `async test_models_versions(client)` (line 33): No module docstring.
  - `async test_signals_latest(client)` (line 41): No module docstring.
  - `async test_backtests_latest(client)` (line 49): No module docstring.
  - `async test_backtests_trades(client)` (line 57): No module docstring.
  - `async test_backtests_equity_curve(client)` (line 65): No module docstring.
  - `async test_autopilot_latest_cycle(client)` (line 73): No module docstring.
  - `async test_autopilot_strategies(client)` (line 81): No module docstring.
  - `async test_autopilot_paper_state(client)` (line 89): No module docstring.
  - `async test_config_get(client)` (line 97): No module docstring.
  - `async test_logs(client)` (line 106): No module docstring.
  - `async test_jobs_list_empty(client)` (line 114): No module docstring.
- Classes: none

### `tests/api/test_services.py`

- LOC: 80
- Module intent: Tests for service wrappers — verify dict outputs.
- Imports (6): `from quant_engine.api.services.data_service import DataService`, `from quant_engine.api.services.autopilot_service import AutopilotService`, `from quant_engine.api.services.backtest_service import BacktestService`, `from quant_engine.api.services.health_service import HealthService`, `from quant_engine.api.services.model_service import ModelService`, `from quant_engine.api.services.results_service import ResultsService`
- Top-level functions:
  - `test_data_service_universe_info()` (line 10): No module docstring.
  - `test_data_service_cached_tickers()` (line 20): No module docstring.
  - `test_backtest_service_no_results()` (line 26): When no results files exist, should return available=False.
  - `test_autopilot_service_latest_cycle()` (line 35): No module docstring.
  - `test_autopilot_service_strategy_registry()` (line 42): No module docstring.
  - `test_autopilot_service_paper_state()` (line 49): No module docstring.
  - `test_health_service_quick()` (line 56): No module docstring.
  - `test_model_service_list_versions()` (line 64): No module docstring.
  - `test_model_service_champion_info()` (line 70): No module docstring.
  - `test_results_service_list()` (line 77): No module docstring.
- Classes: none

### `tests/conftest.py`

- LOC: 240
- Module intent: Shared test fixtures for the quant_engine test suite.
- Imports (6): `from __future__ import annotations`, `import json`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import pytest`
- Top-level functions:
  - `pytest_sessionfinish(session, exitstatus)` (line 12): Spawn a watchdog that force-exits if the process hangs at shutdown.
  - `synthetic_ohlcv_data()` (line 35): 10 synthetic stock series, each with 500 daily bars.
  - `synthetic_trades_csv(tmp_path)` (line 61): Generate a synthetic backtest trades CSV and return its path.
  - `synthetic_model_meta(tmp_model_dir)` (line 93): Write a synthetic model metadata JSON and return the model dir.
  - `tmp_results_dir(tmp_path)` (line 123): Temporary results directory.
  - `tmp_model_dir(tmp_path)` (line 131): Temporary model directory.
  - `tmp_data_cache_dir(tmp_path)` (line 139): Temporary data cache directory.
  - `async app(tmp_path)` (line 195): Create a test FastAPI app with a fresh per-test job store.
  - `async client(app)` (line 232): Async HTTP client bound to the test app.
- Classes:
  - `_InMemoryJobStore` (line 149): Lightweight in-memory job store for tests (avoids aiosqlite threads).
    - Methods (9): `__init__`, `initialize`, `close`, `create_job`, `get_job`, `list_jobs`, `update_status`, `update_progress`, `cancel_job`

### `tests/test_autopilot_predictor_fallback.py`

- LOC: 58
- Module intent: Test module for autopilot predictor fallback behavior and regressions.
- Imports (4): `import unittest`, `from unittest.mock import patch`, `import pandas as pd`, `from quant_engine.autopilot.engine import AutopilotEngine, HeuristicPredictor`
- Top-level functions: none
- Classes:
  - `AutopilotPredictorFallbackTests` (line 13): Test cases covering autopilot predictor fallback behavior and system invariants.
    - Methods (1): `test_ensure_predictor_falls_back_when_model_import_fails`

### `tests/test_backtest_realism.py`

- LOC: 561
- Module intent: Spec 011 — Backtest Execution Realism & Validation Enforcement tests.
- Imports (11): `import inspect`, `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.backtest.advanced_validation import deflated_sharpe_ratio, probability_of_backtest_overfitting`, `from quant_engine.backtest.engine import Backtester, BacktestResult`, `from quant_engine.backtest.execution import ExecutionModel`, `from quant_engine.backtest.optimal_execution import almgren_chriss_trajectory`, `from quant_engine.autopilot.promotion_gate import PromotionGate`, `from quant_engine.autopilot.strategy_discovery import StrategyCandidate`, `from quant_engine.config import ALMGREN_CHRISS_RISK_AVERSION, PROMOTION_MAX_PBO`
- Top-level functions:
  - `_make_ohlcv(n_bars, base_price, daily_volume, seed)` (line 37): Generate synthetic OHLCV data for testing.
  - `_make_predictions(ohlcv, ticker, n_signals, start_bar, spacing)` (line 59): Generate synthetic predictions aligned with OHLCV data.
  - `_candidate()` (line 83): Build a reusable test fixture.
  - `_passing_result(sharpe)` (line 96): Build a passing backtest result.
  - `_all_pass_metrics()` (line 125): Contract metrics that satisfy all promotion gates.
- Classes:
  - `TestEntryTimingConsistency` (line 147): T1: Both simple and risk-managed modes use next-bar Open for entry.
    - Methods (2): `test_simple_mode_entry_at_next_bar_open`, `test_risk_managed_mode_entry_at_next_bar_open`
  - `TestAlmgrenChrissCalibration` (line 204): T2: AC risk_aversion calibrated to institutional levels.
    - Methods (3): `test_config_risk_aversion_not_risk_neutral`, `test_trajectory_default_matches_config`, `test_higher_risk_aversion_frontloads_execution`
  - `TestExitVolumeConstraints` (line 246): T3: Exit simulation respects volume constraints.
    - Methods (3): `test_execution_model_limits_fill_ratio`, `test_force_full_bypasses_volume_constraint`, `test_moderate_order_gets_partial_fill`
  - `TestNegativeSharpeFailsDSR` (line 312): T5: Negative Sharpe strategies always fail DSR.
    - Methods (4): `test_negative_sharpe_rejected`, `test_zero_sharpe_rejected`, `test_positive_sharpe_can_pass`, `test_weak_positive_sharpe_with_many_trials_fails`
  - `TestPBOThreshold` (line 356): T6: PBO threshold tightened to 0.45.
    - Methods (3): `test_config_pbo_threshold`, `test_pbo_above_045_is_overfit`, `test_pbo_max_combinations_increased`
  - `TestValidationRequiredForPromotion` (line 404): T4: Validation required for autopilot promotion.
    - Methods (5): `test_no_validation_rejects_promotion`, `test_negative_sharpe_always_rejected`, `test_mc_not_significant_rejects_promotion`, `test_all_validations_pass_allows_promotion`, `test_pbo_above_045_rejects_promotion`
  - `TestPartialExitMultiBar` (line 486): T7: Residual shares tracked across bars for multi-bar exits.
    - Methods (3): `test_backtester_has_residual_tracking`, `test_volume_constrained_exit_records_fill_ratio`, `test_residual_position_exit_concept`

### `tests/test_cache_metadata_rehydrate.py`

- LOC: 97
- Module intent: Test module for cache metadata rehydrate behavior and regressions.
- Imports (6): `import json`, `import tempfile`, `import unittest`, `from pathlib import Path`, `import pandas as pd`, `from quant_engine.data.local_cache import rehydrate_cache_metadata`
- Top-level functions:
  - `_write_daily_csv(path)` (line 15): Local helper used by the cache metadata rehydrate tests.
- Classes:
  - `CacheMetadataRehydrateTests` (line 31): Test cases covering cache metadata rehydrate behavior and system invariants.
    - Methods (3): `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`

### `tests/test_conceptual_fixes.py`

- LOC: 324
- Module intent: Tests for Spec 008 — Ensemble & Conceptual Fixes.
- Imports (5): `from __future__ import annotations`, `from unittest.mock import patch`, `import numpy as np`, `import pandas as pd`, `import pytest`
- Top-level functions:
  - `_make_synthetic_features(n, seed)` (line 23): Generate synthetic OHLCV features for regime detection.
  - `_make_features_with_regime_flickers(n, seed)` (line 39): Create features designed to produce short regime flickers in rule-based detection.
- Classes:
  - `TestEnsembleNoPhantomVote` (line 97): When REGIME_JUMP_MODEL_ENABLED=False, ensemble must use 2 methods, not 3.
    - Methods (2): `test_two_method_ensemble_does_not_call_jump`, `test_two_method_ensemble_requires_unanimity`
  - `TestEnsembleWithJumpModel` (line 156): When REGIME_JUMP_MODEL_ENABLED=True, ensemble uses 3 independent methods.
    - Methods (2): `test_three_method_ensemble_calls_all_methods`, `test_three_method_probabilities_sum_to_one`
  - `TestRuleDetectMinDuration` (line 211): Rule-based detection must enforce REGIME_MIN_DURATION.
    - Methods (3): `test_no_short_runs_with_min_duration`, `test_min_duration_1_is_noop`, `test_smoothing_preserves_regime_count`
  - `TestConfigEndpointDefaults` (line 270): The /api/config endpoint returns values that match config.py.
    - Methods (4): `test_adjustable_config_includes_backtest_keys`, `test_adjustable_config_values_match_config_py`, `test_config_status_includes_training_section`, `test_config_status_backtest_section_matches`

### `tests/test_covariance_estimator.py`

- LOC: 25
- Module intent: Test module for covariance estimator behavior and regressions.
- Imports (4): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.risk.covariance import CovarianceEstimator`
- Top-level functions: none
- Classes:
  - `CovarianceEstimatorTests` (line 13): Test cases covering covariance estimator behavior and system invariants.
    - Methods (1): `test_single_asset_covariance_is_2d_and_positive`

### `tests/test_data_diagnostics.py`

- LOC: 183
- Module intent: Tests for data loading diagnostics (Spec 005).
- Imports (7): `import json`, `import logging`, `import tempfile`, `import unittest`, `from pathlib import Path`, `from unittest.mock import MagicMock, patch`, `import pandas as pd`
- Top-level functions: none
- Classes:
  - `TestOrchestratorImport` (line 20): T1: DATA_DIR -> DATA_CACHE_DIR fix.
    - Methods (2): `test_orchestrator_imports_data_cache_dir`, `test_orchestrator_uses_data_cache_dir_in_diagnostics`
  - `TestLoadUniverseSkipReasons` (line 47): T2: Skip reasons logged at WARNING even with verbose=False.
    - Methods (2): `test_load_universe_logs_skip_reasons`, `test_skip_reasons_include_reason_text`
  - `TestErrorMessageDiagnostics` (line 91): T3: RuntimeError includes ticker list, WRDS, cache count.
    - Methods (1): `test_error_message_includes_diagnostics`
  - `TestDataStatusService` (line 119): T4: DataService.get_cache_status returns valid ticker info.
    - Methods (4): `test_data_status_returns_summary`, `test_data_status_ticker_entries`, `test_data_status_with_missing_cache_dir`, `test_data_status_freshness_categories`

### `tests/test_delisting_total_return.py`

- LOC: 76
- Module intent: Test module for delisting total return behavior and regressions.
- Imports (4): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.features.pipeline import compute_indicator_features, compute_targets`
- Top-level functions: none
- Classes:
  - `DelistingTotalReturnTests` (line 13): Test cases covering delisting total return behavior and system invariants.
    - Methods (2): `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`

### `tests/test_drawdown_liquidation.py`

- LOC: 138
- Module intent: Test module for drawdown liquidation behavior and regressions.
- Imports (6): `import unittest`, `from types import SimpleNamespace`, `import pandas as pd`, `from quant_engine.backtest.engine import Backtester`, `from quant_engine.risk.drawdown import DrawdownState, DrawdownStatus`, `from quant_engine.risk.stop_loss import StopReason, StopResult`
- Top-level functions: none
- Classes:
  - `_FakePositionSizer` (line 15): Test double used to isolate behavior in this test module.
    - Methods (1): `size_position`
  - `_FakeDrawdownController` (line 21): Test double used to isolate behavior in this test module.
    - Methods (3): `__init__`, `update`, `get_summary`
  - `_FakeStopLossManager` (line 60): Test double used to isolate behavior in this test module.
    - Methods (1): `evaluate`
  - `_FakePortfolioRisk` (line 74): Test double used to isolate behavior in this test module.
    - Methods (1): `check_new_position`
  - `_FakeRiskMetrics` (line 80): Test double used to isolate behavior in this test module.
    - Methods (1): `compute_full_report`
  - `DrawdownLiquidationTests` (line 92): Test cases covering drawdown liquidation behavior and system invariants.
    - Methods (1): `test_critical_drawdown_forces_liquidation`

### `tests/test_execution_dynamic_costs.py`

- LOC: 48
- Module intent: Test module for execution dynamic costs behavior and regressions.
- Imports (2): `import unittest`, `from quant_engine.backtest.execution import ExecutionModel`
- Top-level functions: none
- Classes:
  - `ExecutionDynamicCostTests` (line 10): Test cases covering execution dynamic costs behavior and system invariants.
    - Methods (1): `test_dynamic_costs_increase_under_stress`

### `tests/test_feature_fixes.py`

- LOC: 377
- Module intent: Tests verifying Spec 012 feature engineering fixes:
- Imports (3): `import unittest`, `import numpy as np`, `import pandas as pd`
- Top-level functions:
  - `_make_ohlcv(n, seed)` (line 17): Synthetic OHLCV data for testing.
  - `_make_ohlcv_with_iv(n, seed)` (line 33): Synthetic OHLCV data with option surface columns.
- Classes:
  - `TestTSMomBackwardShift` (line 46): Verify TSMom features are backward-looking (causal).
    - Methods (2): `test_tsmom_uses_backward_shift`, `test_tsmom_values_dont_change_when_future_removed`
  - `TestIVShockNonFuture` (line 112): Verify IV shock features use only backward-looking shifts.
    - Methods (2): `test_iv_shock_no_future_shift`, `test_iv_shock_no_lookahead`
  - `TestFeatureMetadata` (line 163): Verify feature metadata registry coverage.
    - Methods (3): `test_feature_metadata_exists`, `test_all_metadata_has_valid_type`, `test_pipeline_features_have_metadata`
  - `TestProductionModeFilter` (line 223): Verify production_mode filters RESEARCH_ONLY features.
    - Methods (2): `test_production_mode_filters_research`, `test_production_mode_default_false`
  - `TestRollingVWAPIsCausal` (line 272): Verify rolling VWAP uses only past data.
    - Methods (2): `test_rolling_vwap_is_causal`, `test_rolling_vwap_first_rows_nan`
  - `TestMultiHorizonVRP` (line 314): Verify enhanced VRP computation at multiple horizons.
    - Methods (1): `test_vrp_multi_horizon`
  - `TestFullPipelineLookahead` (line 334): End-to-end lookahead check on the complete feature pipeline.
    - Methods (1): `test_automated_lookahead_detection`

### `tests/test_integration.py`

- LOC: 556
- Module intent: End-to-end integration tests for the quant engine pipeline.
- Imports (5): `import sys`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import pytest`
- Top-level functions:
  - `_generate_synthetic_ohlcv(n_stocks, n_days, seed)` (line 23): Generate synthetic OHLCV data for *n_stocks* over *n_days*.
- Classes:
  - `TestFullPipelineSynthetic` (line 70): End-to-end test: data -> features -> regimes -> training -> prediction -> backtest.
    - Methods (8): `synthetic_data`, `pipeline_outputs`, `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`
  - `TestCvGapHardBlock` (line 229): Verify that the CV gap hard block rejects overfit models.
    - Methods (1): `test_cv_gap_hard_block`
  - `TestRegime2Suppression` (line 304): Verify regime 2 gating suppresses trades.
    - Methods (2): `test_regime_2_suppression`, `test_regime_0_not_suppressed`
  - `TestCrossSectionalRanking` (line 412): Verify cross-sectional ranker produces valid output.
    - Methods (4): `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`

### `tests/test_iv_arbitrage_builder.py`

- LOC: 38
- Module intent: Test module for iv arbitrage builder behavior and regressions.
- Imports (3): `import unittest`, `import numpy as np`, `from quant_engine.models.iv.models import ArbitrageFreeSVIBuilder, generate_synthetic_market_surface`
- Top-level functions: none
- Classes:
  - `ArbitrageFreeSVIBuilderTests` (line 12): Test cases covering iv arbitrage builder behavior and system invariants.
    - Methods (1): `test_build_surface_has_valid_shape_and_monotone_total_variance`

### `tests/test_kalshi_asof_features.py`

- LOC: 65
- Module intent: Test module for kalshi asof features behavior and regressions.
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel`
- Top-level functions: none
- Classes:
  - `KalshiAsofFeatureTests` (line 12): Test cases covering kalshi asof features behavior and regression protections.
    - Methods (2): `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`

### `tests/test_kalshi_distribution.py`

- LOC: 114
- Module intent: Test module for kalshi distribution behavior and regressions.
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.kalshi.distribution import DistributionConfig, build_distribution_panel, build_distribution_snapshot`
- Top-level functions: none
- Classes:
  - `KalshiDistributionTests` (line 16): Test cases covering kalshi distribution behavior and regression protections.
    - Methods (3): `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`

### `tests/test_kalshi_hardening.py`

- LOC: 605
- Module intent: Test module for kalshi hardening behavior and regressions.
- Imports (16): `import base64`, `import tempfile`, `import unittest`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.kalshi.client import KalshiClient, KalshiSigner`, `from quant_engine.kalshi.distribution import DistributionConfig, build_distribution_snapshot`, `from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel, build_event_labels`, `from quant_engine.kalshi.mapping_store import EventMarketMappingRecord, EventMarketMappingStore`, `from quant_engine.kalshi.options import add_options_disagreement_features, build_options_reference_panel`, `from quant_engine.kalshi.promotion import EventPromotionConfig, evaluate_event_promotion` (+4 more)
- Top-level functions: none
- Classes:
  - `KalshiHardeningTests` (line 37): Test cases covering kalshi hardening behavior and regression protections.
    - Methods (16): `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof`, `test_mapping_store_asof`, `test_store_ingestion_and_health_tables`, `test_provider_materializes_daily_health_report`, `test_signer_canonical_payload_and_header_fields`

### `tests/test_loader_and_predictor.py`

- LOC: 230
- Module intent: Test module for loader and predictor behavior and regressions.
- Imports (9): `import tempfile`, `import unittest`, `from unittest.mock import patch`, `import pandas as pd`, `from quant_engine.data.loader import load_ohlcv, load_survivorship_universe, load_with_delistings`, `from quant_engine.data.local_cache import load_ohlcv as cache_load`, `from quant_engine.data.local_cache import load_ohlcv_with_meta as cache_load_with_meta`, `from quant_engine.data.local_cache import save_ohlcv as cache_save`, `from quant_engine.models.predictor import EnsemblePredictor`
- Top-level functions: none
- Classes:
  - `_FakeWRDSProvider` (line 22): Test double used to isolate behavior in this test module.
    - Methods (5): `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_option_surface_features`, `resolve_permno`
  - `_UnavailableWRDSProvider` (line 79): Test double representing an unavailable dependency or provider.
    - Methods (1): `available`
  - `LoaderAndPredictorTests` (line 85): Test cases covering loader and predictor behavior and system invariants.
    - Methods (8): `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`

### `tests/test_lookahead_detection.py`

- LOC: 144
- Module intent: Automated lookahead bias detection for the feature pipeline.
- Imports (4): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.features.pipeline import FeaturePipeline`
- Top-level functions:
  - `_make_synthetic_ohlcv(n, seed)` (line 20): Build synthetic OHLCV data for lookahead testing.
- Classes:
  - `TestLookaheadDetection` (line 36): Detect any feature that uses future data.
    - Methods (2): `test_no_feature_uses_future_data`, `test_no_feature_uses_future_data_deep`

### `tests/test_panel_split.py`

- LOC: 55
- Module intent: Test module for panel split behavior and regressions.
- Imports (3): `import unittest`, `import pandas as pd`, `from quant_engine.models.trainer import ModelTrainer`
- Top-level functions: none
- Classes:
  - `PanelSplitTests` (line 12): Test cases covering panel split behavior and system invariants.
    - Methods (2): `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`

### `tests/test_paper_trader_kelly.py`

- LOC: 128
- Module intent: Test module for paper trader kelly behavior and regressions.
- Imports (8): `import json`, `import tempfile`, `import unittest`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `from quant_engine.autopilot.paper_trader import PaperTrader`, `from quant_engine.autopilot.registry import ActiveStrategy`
- Top-level functions:
  - `_mock_price_data()` (line 17): Create mock inputs used by the test cases in this module.
  - `_seed_state(path)` (line 35): Seed deterministic test state used by this module.
  - `_run_cycle(use_kelly)` (line 64): Run the local test helper workflow and return intermediate outputs.
- Classes:
  - `PaperTraderKellyTests` (line 115): Test cases covering paper trader kelly behavior and system invariants.
    - Methods (1): `test_kelly_sizing_changes_position_size_with_bounds`

### `tests/test_position_sizing_overhaul.py`

- LOC: 414
- Module intent: Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul.
- Imports (8): `import math`, `import sys`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import pytest`, `from quant_engine.risk.position_sizer import PositionSizer, PositionSize`, `from quant_engine.risk.drawdown import DrawdownController, DrawdownState`
- Top-level functions: none
- Classes:
  - `TestKellyNegativeEdge` (line 32): T1: Kelly returns 0.0 for negative-edge signals, not min_position.
    - Methods (5): `test_kelly_negative_edge_returns_zero`, `test_kelly_positive_edge_returns_positive`, `test_kelly_invalid_inputs_return_zero`, `test_kelly_small_sample_penalty`, `test_kelly_n_trades_passed_through_size_position`
  - `TestDrawdownGovernor` (line 87): T2: Exponential drawdown governor is more lenient early.
    - Methods (6): `test_convex_at_50pct_drawdown`, `test_aggressive_at_90pct_drawdown`, `test_no_drawdown_full_sizing`, `test_beyond_max_dd_returns_zero`, `test_positive_drawdown_returns_full`, `test_convex_curve_is_smooth`
  - `TestPerRegimeBayesian` (line 137): T3: Bayesian win-rate tracked per-regime with separate priors.
    - Methods (4): `_make_trades`, `test_bull_kelly_greater_than_hv`, `test_global_fallback_when_few_regime_trades`, `test_per_regime_counters_populated`
  - `TestRegimeStatsUpdate` (line 187): T4: Regime stats updated from trade DataFrame with regime column.
    - Methods (3): `test_update_from_trade_df`, `test_too_few_trades_keeps_defaults`, `test_multiple_regimes_updated`
  - `TestConfidenceScalar` (line 232): T5: Confidence scalar range is [0.5, 1.0], never 1.5.
    - Methods (3): `test_max_confidence_scalar_is_one`, `test_zero_confidence_scalar_is_half`, `test_confidence_never_amplifies`
  - `TestPortfolioCorrelationPenalty` (line 268): T6: High-correlation positions get reduced allocation.
    - Methods (4): `returns_data`, `test_high_corr_smaller_than_uncorr`, `test_no_positions_returns_base`, `test_negative_corr_no_penalty`
  - `TestDrawdownControllerIntegration` (line 331): T7: DrawdownController blocks entries and forces liquidation.
    - Methods (5): `test_normal_state_allows_entries`, `test_large_loss_triggers_caution`, `test_critical_drawdown_forces_liquidation`, `test_paper_trader_has_drawdown_controller`, `test_recovery_allows_gradual_reentry`
  - `TestSizePositionIntegration` (line 387): End-to-end tests for the updated size_position method.
    - Methods (2): `test_negative_edge_signal_still_gets_sized_via_blend`, `test_drawdown_reduces_kelly_in_blend`

### `tests/test_promotion_contract.py`

- LOC: 105
- Module intent: Test module for promotion contract behavior and regressions.
- Imports (5): `import unittest`, `import pandas as pd`, `from quant_engine.autopilot.promotion_gate import PromotionGate`, `from quant_engine.autopilot.strategy_discovery import StrategyCandidate`, `from quant_engine.backtest.engine import BacktestResult`
- Top-level functions:
  - `_candidate()` (line 14): Build a reusable test fixture object for promotion-contract assertions.
  - `_result()` (line 27): Build a reusable test fixture object for promotion-contract assertions.
- Classes:
  - `PromotionContractTests` (line 56): Test cases covering promotion contract behavior and system invariants.
    - Methods (2): `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`

### `tests/test_provider_registry.py`

- LOC: 29
- Module intent: Test module for provider registry behavior and regressions.
- Imports (3): `import unittest`, `from quant_engine.data.provider_registry import get_provider, list_providers`, `from quant_engine.kalshi.client import KalshiClient`
- Top-level functions: none
- Classes:
  - `ProviderRegistryTests` (line 11): Test cases covering provider registry behavior and system invariants.
    - Methods (3): `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`

### `tests/test_research_factors.py`

- LOC: 133
- Module intent: Test module for research factors behavior and regressions.
- Imports (5): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.features.pipeline import FeaturePipeline`, `from quant_engine.features.research_factors import ResearchFactorConfig, compute_cross_asset_research_factors, compute_single_asset_research_factors`
- Top-level functions:
  - `_make_ohlcv(seed, periods, drift)` (line 18): Build synthetic test data for the scenarios in this module.
- Classes:
  - `ResearchFactorTests` (line 40): Test cases covering research factors behavior and system invariants.
    - Methods (4): `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`

### `tests/test_survivorship_pit.py`

- LOC: 74
- Module intent: Test module for survivorship pit behavior and regressions.
- Imports (5): `import tempfile`, `import unittest`, `from pathlib import Path`, `import pandas as pd`, `from quant_engine.data.survivorship import filter_panel_by_point_in_time_universe, hydrate_universe_history_from_snapshots`
- Top-level functions: none
- Classes:
  - `SurvivorshipPointInTimeTests` (line 17): Test cases covering survivorship pit behavior and system invariants.
    - Methods (1): `test_filter_panel_by_point_in_time_universe`

### `tests/test_training_pipeline_fixes.py`

- LOC: 567
- Module intent: Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance.
- Imports (6): `import sys`, `from collections import Counter`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import pytest`
- Top-level functions:
  - `_make_feature_matrix(n_rows, n_features, seed)` (line 26): Create a synthetic feature matrix with a MultiIndex (permno, date).
  - `_make_targets(X, signal_features, seed)` (line 40): Create synthetic targets with signal from specified features.
  - `_make_regimes(X, seed)` (line 51): Create synthetic regime labels (0-3) aligned with features.
- Classes:
  - `TestPerFoldFeatureSelection` (line 62): Verify that feature selection runs independently per CV fold.
    - Methods (3): `test_feature_selection_per_fold`, `test_stable_features_selected_most_folds`, `test_compute_stable_features_fallback`
  - `TestCalibrationValidationSplit` (line 185): Verify calibration uses a separate split and computes ECE.
    - Methods (5): `test_calibration_uses_separate_split`, `test_ece_computed`, `test_ece_perfect_calibration`, `test_ece_worst_calibration`, `test_reliability_curve`
  - `TestRegimeMinSamples` (line 282): Verify regime models require minimum sample count.
    - Methods (3): `test_regime_model_min_samples`, `test_regime_model_skipped_for_low_samples`, `test_regime_fallback_to_global`
  - `TestCorrelationPruning` (line 365): Verify feature correlation pruning at 0.80 threshold.
    - Methods (3): `test_correlation_threshold_080`, `test_old_threshold_would_keep_correlated`, `test_default_threshold_is_080`
  - `TestGovernanceScoring` (line 443): Verify governance scoring includes validation metrics.
    - Methods (4): `test_governance_score_includes_validation`, `test_governance_score_backward_compatible`, `test_governance_promotion_with_validation`, `test_dsr_penalty_reduces_score`

### `tests/test_validation_and_risk_extensions.py`

- LOC: 107
- Module intent: Test module for validation and risk extensions behavior and regressions.
- Imports (5): `import unittest`, `import numpy as np`, `import pandas as pd`, `from quant_engine.backtest.validation import combinatorial_purged_cv, superior_predictive_ability, strategy_signal_returns`, `from quant_engine.risk.portfolio_risk import PortfolioRiskManager`
- Top-level functions:
  - `_make_ohlcv(close)` (line 18): Build synthetic test data for the scenarios in this module.
- Classes:
  - `ValidationAndRiskExtensionTests` (line 32): Test cases covering validation and risk extensions behavior and system invariants.
    - Methods (3): `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`

### `tests/test_zero_errors.py`

- LOC: 120
- Module intent: Integration test: common operations must produce zero ERROR-level log entries.
- Imports (6): `from __future__ import annotations`, `import logging`, `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import pytest`
- Top-level functions:
  - `error_capture()` (line 29): Attach an error-capturing handler to the root logger for the test.
  - `test_config_validation_no_errors(error_capture)` (line 38): validate_config() must not trigger ERROR-level logs.
  - `test_regime_detection_no_errors(error_capture)` (line 49): Regime detection on synthetic data must not produce ERROR-level logs.
  - `test_health_service_no_errors(error_capture)` (line 75): HealthService instantiation must not produce ERROR-level logs.
  - `test_data_loading_graceful_degradation(error_capture)` (line 87): Loading a nonexistent ticker must fail gracefully (no ERROR logs, no crash).
  - `test_data_loading_single_ticker_if_cache_exists(error_capture)` (line 98): If cache exists, loading a single ticker must not produce ERROR-level logs.
- Classes:
  - `_ErrorCapture` (line 17): Handler that records any ERROR or CRITICAL log records.
    - Methods (2): `__init__`, `emit`

## Package `utils`

### `utils/__init__.py`

- LOC: 5
- Module intent: Utility helpers package namespace.
- Top-level functions: none
- Classes: none

### `utils/logging.py`

- LOC: 439
- Module intent: Structured logging for the quant engine.
- Imports (9): `from __future__ import annotations`, `import json`, `import logging`, `import sys`, `import urllib.request`, `import urllib.error`, `from datetime import datetime, timezone`, `from pathlib import Path`, `from typing import Any, Dict, List, Optional`
- Top-level functions:
  - `get_logger(name, level)` (line 46): Get a structured logger for the quant engine.
- Classes:
  - `StructuredFormatter` (line 22): JSON formatter for machine-parseable log output.
    - Methods (1): `format`
  - `AlertHistory` (line 77): Persistent alert history with optional webhook notifications.
    - Methods (7): `__init__`, `_load`, `_save`, `record`, `record_batch`, `query`, `_notify_webhook`
  - `MetricsEmitter` (line 322): Emit key metrics on every cycle and check alert thresholds.
    - Methods (3): `__init__`, `emit_cycle_metrics`, `check_alerts`
