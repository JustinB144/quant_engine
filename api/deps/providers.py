"""Singleton dependency providers for FastAPI ``Depends()``."""
from __future__ import annotations

from functools import lru_cache

from ..config import ApiSettings, RuntimeConfig


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache
def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


# Lazy singletons — initialised at first call rather than import time
# so the event loop is already running when async resources are needed.

_job_store = None
_job_runner = None
_cache = None


def get_job_store():
    """Return the singleton ``JobStore``."""
    global _job_store
    if _job_store is None:
        from ..jobs.store import JobStore

        _job_store = JobStore(get_settings().job_db_path)
    return _job_store


def get_job_runner():
    """Return the singleton ``JobRunner``."""
    global _job_runner
    if _job_runner is None:
        from ..jobs.runner import JobRunner

        _job_runner = JobRunner(get_job_store())
    return _job_runner


def get_cache():
    """Return the singleton ``CacheManager``."""
    global _cache
    if _cache is None:
        from ..cache.manager import CacheManager

        _cache = CacheManager()
    return _cache
