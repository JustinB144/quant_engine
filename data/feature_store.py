"""
Point-in-time feature store for backtest acceleration.

Stores precomputed features with temporal metadata to prevent
look-ahead bias and enable fast feature retrieval during backtesting.

Storage layout::

    <store_dir>/
        <permno>/
            <feature_version>/
                features_<computed_at>.parquet
                features_<computed_at>.meta.json

The JSON sidecar records ``computed_at``, ``feature_version``, row count,
and column names so that ``load_features`` can enforce point-in-time
constraints without reading the parquet files first.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..config import ROOT_DIR

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = ROOT_DIR / "data" / "feature_store"

_DATE_FMT = "%Y-%m-%d"


def _atomic_replace(target: Path, write_fn: Callable[[str], None]) -> None:
    """Write to *target* atomically via a temp file and ``os.replace``.

    On success the temp file is atomically renamed to *target* (POSIX
    guarantees of ``os.replace``).  On failure the temp file is cleaned
    up and the exception re-raised so that *target* is never left in a
    partially-written state.
    """
    dir_name = str(target.parent)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        os.close(fd)
        write_fn(tmp_path)
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class FeatureStore:
    """Point-in-time feature store for backtest acceleration.

    Stores precomputed features with temporal metadata to prevent
    look-ahead bias and enable fast feature retrieval.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, store_dir: Optional[Path] = None):
        """Initialize FeatureStore."""
        self.store_dir = Path(store_dir) if store_dir is not None else _DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _version_dir(self, permno: str, feature_version: str) -> Path:
        """Internal helper for version dir."""
        return self.store_dir / permno / feature_version

    @staticmethod
    def _ts_tag(computed_at: str) -> str:
        """Normalise *computed_at* to ``YYYY-MM-DD`` for file names."""
        # Accept both ``YYYY-MM-DD`` and ISO-8601 with time component.
        return computed_at[:10]

    def _parquet_path(self, permno: str, feature_version: str, computed_at: str) -> Path:
        """Internal helper for parquet path."""
        tag = self._ts_tag(computed_at)
        return self._version_dir(permno, feature_version) / f"features_{tag}.parquet"

    def _meta_path(self, permno: str, feature_version: str, computed_at: str) -> Path:
        """Internal helper for meta path."""
        tag = self._ts_tag(computed_at)
        return self._version_dir(permno, feature_version) / f"features_{tag}.meta.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_features(
        self,
        permno: str,
        features: pd.DataFrame,
        computed_at: str,
        feature_version: str = "v1",
    ) -> Path:
        """Persist a feature DataFrame with point-in-time metadata.

        Parameters
        ----------
        permno:
            Security identifier (e.g. ``"AAPL"`` or a PERMNO string).
        features:
            DataFrame of precomputed features.  The index should be a
            ``DatetimeIndex`` (dates for which the features are valid).
        computed_at:
            Date string (``YYYY-MM-DD``) indicating *when* these features
            were computed.  Used for point-in-time gating in
            ``load_features``.
        feature_version:
            Versioning tag (default ``"v1"``).  Allows storing multiple
            generations of a feature pipeline side-by-side.

        Returns
        -------
        Path
            The path to the saved parquet file.
        """
        vdir = self._version_dir(permno, feature_version)
        vdir.mkdir(parents=True, exist_ok=True)

        pq_path = self._parquet_path(permno, feature_version, computed_at)
        meta_path = self._meta_path(permno, feature_version, computed_at)

        # Write parquet atomically
        _atomic_replace(pq_path, lambda p: features.to_parquet(p, engine="pyarrow"))

        # Write JSON sidecar atomically
        meta = {
            "permno": permno,
            "computed_at": computed_at,
            "feature_version": feature_version,
            "row_count": len(features),
            "columns": list(features.columns),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        }

        def _dump_json(tmp: str) -> None:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        _atomic_replace(meta_path, _dump_json)

        logger.info(
            "FeatureStore: saved %d rows for %s (version=%s, computed_at=%s)",
            len(features),
            permno,
            feature_version,
            computed_at,
        )
        return pq_path

    def load_features(
        self,
        permno: str,
        as_of: Optional[str] = None,
        feature_version: str = "v1",
    ) -> Optional[pd.DataFrame]:
        """Load the most recent feature set for *permno*, respecting
        point-in-time constraints.

        Parameters
        ----------
        permno:
            Security identifier.
        as_of:
            If provided (``YYYY-MM-DD``), only snapshots whose
            ``computed_at`` date is **on or before** this date are
            considered.  This prevents future information from leaking
            into a backtest.
        feature_version:
            Feature version tag to load (default ``"v1"``).

        Returns
        -------
        pd.DataFrame or None
            The feature DataFrame, or ``None`` if no qualifying
            snapshot exists.
        """
        vdir = self._version_dir(permno, feature_version)
        if not vdir.exists():
            return None

        # Collect all metadata files and pick the latest valid one.
        meta_files = sorted(vdir.glob("*.meta.json"))
        if not meta_files:
            return None

        best_meta: Optional[Dict] = None
        best_date: Optional[str] = None

        for mf in meta_files:
            try:
                meta = json.loads(mf.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("FeatureStore: skipping corrupt metadata %s", mf)
                continue

            computed = meta.get("computed_at", "")[:10]

            # Point-in-time gate: skip snapshots computed *after* as_of.
            if as_of is not None:
                as_of_date = as_of[:10]
                if computed > as_of_date:
                    continue

            if best_date is None or computed > best_date:
                best_date = computed
                best_meta = meta

        if best_meta is None:
            return None

        pq_path = self._parquet_path(permno, feature_version, best_meta["computed_at"])
        if not pq_path.exists():
            logger.warning("FeatureStore: metadata exists but parquet missing: %s", pq_path)
            return None

        df = pd.read_parquet(pq_path)
        logger.info(
            "FeatureStore: loaded %d rows for %s (version=%s, computed_at=%s)",
            len(df),
            permno,
            feature_version,
            best_meta["computed_at"],
        )
        return df

    def list_available(self, permno: Optional[str] = None) -> List[Dict[str, str]]:
        """List stored feature snapshots.

        Parameters
        ----------
        permno:
            If provided, restrict listing to this security.  Otherwise
            list all securities.

        Returns
        -------
        list of dict
            Each dict contains ``permno``, ``feature_version``,
            ``computed_at``, and ``row_count``.
        """
        results: List[Dict[str, str]] = []

        if permno is not None:
            permno_dirs = [self.store_dir / permno]
        else:
            permno_dirs = [
                d for d in sorted(self.store_dir.iterdir()) if d.is_dir()
            ]

        for pdir in permno_dirs:
            if not pdir.exists():
                continue
            pname = pdir.name
            for vdir in sorted(pdir.iterdir()):
                if not vdir.is_dir():
                    continue
                for mf in sorted(vdir.glob("*.meta.json")):
                    try:
                        meta = json.loads(mf.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue
                    results.append({
                        "permno": pname,
                        "feature_version": meta.get("feature_version", vdir.name),
                        "computed_at": meta.get("computed_at", ""),
                        "row_count": str(meta.get("row_count", "?")),
                    })

        return results

    def invalidate(self, permno: str, feature_version: Optional[str] = None) -> int:
        """Remove cached features for a security.

        Parameters
        ----------
        permno:
            Security identifier.
        feature_version:
            If provided, only invalidate this version.  Otherwise
            invalidate **all** versions for the security.

        Returns
        -------
        int
            Number of parquet files deleted.
        """
        pdir = self.store_dir / permno
        if not pdir.exists():
            return 0

        if feature_version is not None:
            dirs_to_clean = [pdir / feature_version]
        else:
            dirs_to_clean = [d for d in pdir.iterdir() if d.is_dir()]

        deleted = 0
        for vdir in dirs_to_clean:
            if not vdir.exists():
                continue
            for f in list(vdir.iterdir()):
                f.unlink()
                if f.suffix == ".parquet":
                    deleted += 1
            # Remove empty version directory.
            try:
                vdir.rmdir()
            except OSError:
                pass

        # Remove empty permno directory.
        try:
            pdir.rmdir()
        except OSError:
            pass

        logger.info(
            "FeatureStore: invalidated %d parquet file(s) for %s (version=%s)",
            deleted,
            permno,
            feature_version or "ALL",
        )
        return deleted
