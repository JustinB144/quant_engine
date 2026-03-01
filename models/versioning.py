"""
Model Versioning — timestamped model directories with registry.

Layout:
    trained_models/
    ├── registry.json              # {"latest": "20260220_143022", "versions": [...]}
    ├── 20260220_143022/           # timestamped version
    │   ├── ensemble_10d_*.pkl
    │   ├── ensemble_10d_meta.json
    │   └── version_info.json      # training metadata
    ├── 20260115_091500/           # previous version
    │   └── ...
    └── ensemble_10d_*.pkl         # legacy flat files (backward compat)
"""
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import MODEL_DIR, MAX_MODEL_VERSIONS

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a single model version."""
    version_id: str                      # e.g. "20260220_143022"
    training_date: str                   # ISO format
    horizon: int                         # prediction horizon (days)
    universe_size: int                   # number of PERMNO series trained on
    n_samples: int = 0                   # total training samples
    n_features: int = 0                  # number of selected features
    # Metrics
    oos_spearman: float = 0.0            # OOS Spearman correlation
    cv_gap: float = 0.0                  # IS - OOS gap
    holdout_r2: float = 0.0              # holdout R²
    holdout_spearman: float = 0.0        # holdout Spearman
    # Survivorship
    survivorship_mode: bool = False      # trained on survivorship-free data?
    universe_as_of: Optional[str] = None # point-in-time universe date
    delisted_included: int = 0           # number of delisted stocks included
    # Misc
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize ModelVersion to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        # Handle missing fields gracefully
        """Build ModelVersion from a serialized dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class ModelRegistry:
    """
    Manages versioned model storage and retrieval.

    Registry file: trained_models/registry.json
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize ModelRegistry."""
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.registry_path = self.model_dir / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or initialize the registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"latest": None, "versions": []}

    def _save_registry(self):
        """Persist registry to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    @property
    def latest_version_id(self) -> Optional[str]:
        """Get the latest version ID, or None if no versions exist."""
        return self._registry.get("latest")

    def get_latest(self) -> Optional[ModelVersion]:
        """Get the latest model version metadata."""
        vid = self.latest_version_id
        if vid is None:
            return None
        return self.get_version(vid)

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get metadata for a specific version."""
        for v in self._registry["versions"]:
            if v["version_id"] == version_id:
                return ModelVersion.from_dict(v)
        return None

    def get_version_dir(self, version_id: str) -> Path:
        """Get the directory path for a model version."""
        return self.model_dir / version_id

    def get_latest_dir(self) -> Optional[Path]:
        """Get the directory path for the latest version."""
        vid = self.latest_version_id
        if vid is None:
            return None
        return self.get_version_dir(vid)

    def list_versions(self) -> List[ModelVersion]:
        """List all versions, newest first."""
        versions = [ModelVersion.from_dict(v) for v in self._registry["versions"]]
        return sorted(versions, key=lambda v: v.version_id, reverse=True)

    def create_version_dir(self, version_id: Optional[str] = None) -> tuple:
        """
        Create a new timestamped version directory.

        Returns:
            (version_id, version_dir_path)
        """
        if version_id is None:
            version_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        version_dir = self.model_dir / version_id
        if version_dir.exists():
            # Extremely unlikely with microsecond precision, but handle it
            original_id = version_id
            version_id = f"{version_id}_{uuid.uuid4().hex[:6]}"
            version_dir = self.model_dir / version_id
            logger.warning(
                "Version directory collision for %s; using fallback %s",
                original_id, version_id,
            )
        version_dir.mkdir(parents=True, exist_ok=False)
        return version_id, version_dir

    def register_version(self, version: ModelVersion):
        """
        Register a newly trained model version and set it as latest.
        """
        # Remove any existing entry with same ID
        self._registry["versions"] = [
            v for v in self._registry["versions"]
            if v["version_id"] != version.version_id
        ]
        self._registry["versions"].append(version.to_dict())
        self._registry["latest"] = version.version_id

        # Save version_info.json inside the version directory
        version_dir = self.get_version_dir(version.version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        info_path = version_dir / "version_info.json"
        with open(info_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        self._save_registry()

    def rollback(self, version_id: str) -> bool:
        """
        Set a previous version as the active latest.

        Returns True if successful.
        """
        for v in self._registry["versions"]:
            if v["version_id"] == version_id:
                version_dir = self.get_version_dir(version_id)
                if version_dir.exists():
                    self._registry["latest"] = version_id
                    self._save_registry()
                    return True
        return False

    def prune_old(self, keep_n: Optional[int] = None):
        """
        Remove old versions beyond keep_n most recent.

        Never removes the current latest version.  Guarantees exactly
        ``keep_n`` entries (or fewer if fewer exist), with the latest
        always included.
        """
        if keep_n is None:
            keep_n = MAX_MODEL_VERSIONS

        versions = sorted(
            self._registry["versions"],
            key=lambda v: v["version_id"],
            reverse=True,
        )

        latest = self._registry.get("latest")
        to_keep = []
        to_remove = []

        # Always keep the latest version first
        latest_entry = None
        for v in versions:
            if v["version_id"] == latest:
                latest_entry = v
                break

        # Fill remaining slots from most recent
        for v in versions:
            if v["version_id"] == latest:
                continue  # Already pinned
            if len(to_keep) < keep_n - (1 if latest_entry else 0):
                to_keep.append(v)
            else:
                to_remove.append(v)

        if latest_entry:
            to_keep.insert(0, latest_entry)

        # Delete old version directories
        for v in to_remove:
            version_dir = self.get_version_dir(v["version_id"])
            if version_dir.exists():
                shutil.rmtree(version_dir)

        self._registry["versions"] = to_keep
        self._save_registry()

    def has_versions(self) -> bool:
        """Check if any versions are registered."""
        return len(self._registry["versions"]) > 0
