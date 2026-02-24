"""
Feature versioning system.

Tracks feature pipeline versions so models and their feature definitions
stay aligned. When the feature pipeline changes (new features, renamed
features, changed parameters), the version hash changes, preventing
accidental model-feature mismatches.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureVersion:
    """Immutable snapshot of a feature pipeline configuration."""

    feature_names: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        self.feature_names = sorted(self.feature_names)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the feature definition."""
        data = {
            "features": self.feature_names,
            "params": self.parameters,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "version_hash": self.compute_hash(),
            "parameters": self.parameters,
            "created_at": self.created_at,
        }

    def diff(self, other: FeatureVersion) -> Dict[str, Any]:
        """Compare this version to another, returning added/removed features."""
        self_set = set(self.feature_names)
        other_set = set(other.feature_names)
        return {
            "added": sorted(other_set - self_set),
            "removed": sorted(self_set - other_set),
            "unchanged": sorted(self_set & other_set),
            "param_changes": {
                k: {"old": self.parameters.get(k), "new": other.parameters.get(k)}
                for k in set(self.parameters) | set(other.parameters)
                if self.parameters.get(k) != other.parameters.get(k)
            },
        }

    def is_compatible(self, other: FeatureVersion) -> bool:
        """Check if two versions have identical feature sets (ignoring params)."""
        return set(self.feature_names) == set(other.feature_names)


class FeatureRegistry:
    """Registry tracking feature versions over time with JSON persistence."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/feature_versions.json")
        self._versions: List[Dict[str, Any]] = []
        self._load()

    def register(self, version: FeatureVersion) -> str:
        """Register a new feature version. Returns the version hash."""
        version_hash = version.compute_hash()

        # Check if this exact version already exists
        for v in self._versions:
            if v.get("version_hash") == version_hash:
                return version_hash

        entry = version.to_dict()
        entry["version_index"] = len(self._versions)
        self._versions.append(entry)
        self._save()
        logger.info(
            "Registered feature version %s (%d features)",
            version_hash,
            version.n_features,
        )
        return version_hash

    def get_version(self, version_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a version by its hash."""
        for v in self._versions:
            if v.get("version_hash") == version_hash:
                return v
        return None

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recently registered version."""
        if not self._versions:
            return None
        return self._versions[-1]

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered versions (most recent first)."""
        return list(reversed(self._versions))

    def check_compatibility(
        self, model_version_hash: str, current_features: List[str]
    ) -> Dict[str, Any]:
        """Check if a model's feature version is compatible with current features.

        Returns a dict with compatibility status and any mismatches.
        """
        model_version = self.get_version(model_version_hash)
        if model_version is None:
            return {
                "compatible": False,
                "reason": f"Unknown feature version: {model_version_hash}",
                "missing": [],
                "extra": [],
            }

        model_feats = set(model_version["feature_names"])
        current_feats = set(current_features)

        missing = sorted(model_feats - current_feats)  # model expects but not available
        extra = sorted(current_feats - model_feats)  # available but model doesn't use

        return {
            "compatible": len(missing) == 0,
            "reason": "" if not missing else f"Missing {len(missing)} features model expects",
            "missing": missing,
            "extra": extra,
            "model_n_features": len(model_feats),
            "current_n_features": len(current_feats),
        }

    def _load(self) -> None:
        try:
            if self.storage_path.exists():
                with open(self.storage_path) as f:
                    self._versions = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load feature registry: %s", e)
            self._versions = []

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self._versions, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save feature registry: %s", e)
