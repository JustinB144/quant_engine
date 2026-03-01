"""
Atomic JSON write utility for crash-safe state persistence.

Uses tempfile.mkstemp + os.replace to ensure that a crash during
serialization never corrupts the target file.  Before each write,
the current file is copied to a .bak backup.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def atomic_json_write(path: Path, payload: Dict[str, Any], **kwargs: Any) -> None:
    """Write *payload* to *path* atomically via temp-file + rename.

    1. Copy the current file (if it exists) to ``path.bak`` as a backup.
    2. Serialize *payload* to a temporary file in the same directory.
    3. ``os.replace`` the temp file onto *path* (atomic on POSIX).

    If serialization fails, the temp file is removed and the original
    file remains untouched.

    Parameters
    ----------
    path : Path
        Target JSON file path.
    payload : dict
        Data to serialize.
    **kwargs
        Extra keyword arguments forwarded to ``json.dump`` (e.g.
        ``indent``, ``default``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Back up current file before overwriting
    if path.exists():
        backup = path.with_suffix(".bak")
        try:
            shutil.copy2(str(path), str(backup))
        except OSError as exc:
            logger.warning("Failed to create backup %s: %s", backup, exc)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f, **kwargs)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_json_load(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    """Load JSON from *path* with corrupt-file recovery.

    If the primary file cannot be decoded, attempts to load from
    ``path.bak``.  Returns *default* if neither file is usable.

    Parameters
    ----------
    path : Path
        Primary JSON file path.
    default : dict
        Fallback value when no usable file is found.

    Returns
    -------
    dict
        Parsed JSON payload.
    """
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Corrupt state file %s: %s — attempting backup", path, exc)
            backup = path.with_suffix(".bak")
            if backup.exists():
                try:
                    with open(backup, "r") as f:
                        data = json.load(f)
                    logger.info("Recovered state from backup %s", backup)
                    return data
                except (json.JSONDecodeError, ValueError) as bak_exc:
                    logger.error("Backup also corrupt %s: %s", backup, bak_exc)
            return default
    return default
