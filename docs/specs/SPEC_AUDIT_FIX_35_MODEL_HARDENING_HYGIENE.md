# SPEC_AUDIT_FIX_35: Model Hardening & Hygiene

**Priority:** MEDIUM — Defense-in-depth, edge-condition safety, and API consistency improvements.
**Scope:** `models/predictor.py`, `models/trainer.py`, `models/versioning.py`, `models/__init__.py`
**Estimated effort:** 3–4 hours
**Depends on:** SPEC_34 (contracts must be stable before hardening)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 6 surfaced several hardening and hygiene items that do not cause incorrect results today but represent latent risks or code quality gaps. These include unsigned model artifact deserialization, missing scaler attribute stubs that could break future introspection, version directory collision at second resolution, a prune_old off-by-one, and inconsistent `__init__.py` public surface.

### Cross-Audit Reconciliation

| Finding | Auditor 1 | Auditor 2 | Disposition |
|---------|-----------|-----------|-------------|
| Joblib deserialization risk | F-02 (HIGH) | — | **NEW** — accepted risk with documentation + optional checksum |
| IdentityScaler attribute stubs | F-03 (MEDIUM) | — | **NEW** — future-proofing |
| Version directory collision | — | F-08 (MEDIUM) | **NEW** — second-resolution race |
| Prune off-by-one | — | F-08 (MEDIUM) | **NEW** — can retain keep_n+1 |
| `__init__.py` missing exports | F-04 (MEDIUM) | — | **NEW** — style/consistency |

---

## Tasks

### T1: Document and Mitigate Joblib Deserialization Risk

**Problem:** `predictor.py:127-128,145,171-176` uses `joblib.load()` on `.pkl` files. Joblib deserialization is equivalent to `pickle.load()` and can execute arbitrary Python code. If the `trained_models/` directory is compromised, loading a malicious artifact executes that code with the process's full permissions. No integrity checks, checksums, or signatures exist.

**Files:** `models/predictor.py`, `models/trainer.py`

**Implementation:**
1. **Immediate (documentation + warning):** Add a prominent docstring/comment in `predictor._load()`:
   ```python
   # SECURITY NOTE: joblib.load() is equivalent to pickle.load() and can
   # execute arbitrary code during deserialization. Model artifacts MUST
   # only be sourced from trusted, operator-controlled directories.
   # See: https://joblib.readthedocs.io/en/latest/persistence.html
   ```
2. **Defense-in-depth (checksum verification):** In `trainer._save()`, compute and store SHA-256 checksums alongside each artifact:
   ```python
   import hashlib

   def _compute_checksum(filepath: Path) -> str:
       sha = hashlib.sha256()
       with open(filepath, "rb") as f:
           for chunk in iter(lambda: f.read(8192), b""):
               sha.update(chunk)
       return sha.hexdigest()

   # After saving each .pkl file:
   checksum = _compute_checksum(model_path)
   checksums[model_path.name] = checksum
   # Write checksums to meta.json under "artifact_checksums" key
   ```
3. In `predictor._load()`, verify checksums before loading:
   ```python
   expected = self.meta.get("artifact_checksums", {})
   if expected:
       actual = _compute_checksum(model_path)
       if actual != expected.get(model_path.name):
           raise ValueError(
               f"Checksum mismatch for {model_path.name}: "
               f"expected {expected.get(model_path.name)}, got {actual}. "
               f"Model artifact may be corrupted or tampered with."
           )
   ```
4. Make checksum verification opt-in via `VERIFY_MODEL_CHECKSUMS = True` config constant (default True for new installations, False for backward compat).

**Acceptance:** Model artifacts have SHA-256 checksums in meta.json. Predictor verifies checksums when enabled. A tampered file raises ValueError. Security note is documented.

---

### T2: Add Attribute Stubs to IdentityScaler

**Problem:** `trainer.py:90-113` defines `IdentityScaler` as a no-op scaler for `DiverseEnsemble`. It implements `.fit()`, `.transform()`, `.fit_transform()`, and `.inverse_transform()` but lacks `mean_`, `scale_`, `var_`, and `n_features_in_` attributes that `sklearn.preprocessing.StandardScaler` exposes. Any downstream code that inspects scaler internals (e.g., for logging, diagnostics, or serialization inspection) will get `AttributeError`.

**File:** `models/trainer.py`

**Implementation:**
1. Add stub attributes in `fit()`:
   ```python
   class IdentityScaler:
       """No-op scaler for DiverseEnsemble (handles its own per-constituent scaling)."""

       def fit(self, X, y=None):
           X = np.asarray(X)
           n = X.shape[1] if X.ndim > 1 else 1
           self.n_features_in_ = n
           self.mean_ = np.zeros(n)
           self.scale_ = np.ones(n)
           self.var_ = np.ones(n)
           self.n_samples_seen_ = X.shape[0]
           return self

       def transform(self, X):
           return np.asarray(X)

       def fit_transform(self, X, y=None):
           self.fit(X, y)
           return self.transform(X)

       def inverse_transform(self, X):
           return np.asarray(X)
   ```
2. Values are identity-consistent: `mean_=0`, `scale_=1`, `var_=1`.

**Acceptance:** `IdentityScaler().fit(X).mean_` returns a zero array. `IdentityScaler().fit(X).scale_` returns a ones array. No `AttributeError` on any StandardScaler-compatible attribute.

---

### T3: Fix Version Directory Second-Resolution Collision

**Problem:** `versioning.py:128-129` generates version IDs with second-resolution timestamps (`%Y%m%d_%H%M%S`). With `exist_ok=True` at line 132, two model versions created within the same second silently share a directory, causing the second training run to overwrite the first's artifacts without warning.

**File:** `models/versioning.py`

**Implementation:**
1. Add microsecond precision to prevent collision:
   ```python
   if version_id is None:
       version_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
   ```
2. Remove `exist_ok=True` and handle collision explicitly:
   ```python
   version_dir = self.model_dir / version_id
   if version_dir.exists():
       # Extremely unlikely with microsecond precision, but handle it
       version_id = f"{version_id}_{uuid.uuid4().hex[:6]}"
       version_dir = self.model_dir / version_id
   version_dir.mkdir(parents=True, exist_ok=False)
   ```
3. Log a warning if collision fallback is triggered.

**Acceptance:** Two `create_version_dir()` calls within the same second produce different directories. `exist_ok=True` is removed.

---

### T4: Fix prune_old Off-By-One in Version Retention

**Problem:** `versioning.py:190-191` iterates sorted versions and keeps entries where `v["version_id"] == latest OR len(to_keep) < keep_n`. If the latest version is not among the first `keep_n` entries in the sorted list, the final roster can contain `keep_n + 1` entries (the `keep_n` most recent plus the latest pinned separately).

**File:** `models/versioning.py`

**Implementation:**
1. Separate the latest-pinning logic from the count logic:
   ```python
   def prune_old(self, keep_n: Optional[int] = None):
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
   ```
2. This guarantees exactly `keep_n` entries (or fewer if fewer exist), with the latest always included.

**Acceptance:** After `prune_old(keep_n=5)`, the registry has at most 5 entries. The latest version is always retained. No off-by-one.

---

### T5: Add Missing High-Use Exports to `models/__init__.py`

**Problem:** `models/__init__.py` exports 9 symbols but omits the 5 most-consumed classes: `ModelTrainer`, `EnsemblePredictor`, `ModelRegistry`, `ModelVersion`, `RetrainTrigger`. All 28 external consumers import these via full submodule paths (`from models.trainer import ModelTrainer`), so this works. But the package's `__init__.py` public surface is inconsistent with its actual public API.

**File:** `models/__init__.py`

**Implementation:**
1. Add the missing exports:
   ```python
   from .trainer import ModelTrainer
   from .predictor import EnsemblePredictor
   from .versioning import ModelRegistry, ModelVersion
   from .retrain_trigger import RetrainTrigger
   ```
2. Update `__all__` if it exists to include all exported names.
3. This is backward-compatible — existing full-path imports continue to work.

**Acceptance:** `from models import ModelTrainer, EnsemblePredictor, ModelRegistry, ModelVersion, RetrainTrigger` works. Existing imports via submodule paths are unaffected.

---

## Verification

- [ ] Run `pytest tests/ -k "model or predictor or trainer or versioning"` — all pass
- [ ] Verify checksum mismatch raises ValueError
- [ ] Verify `IdentityScaler().fit(X).mean_` returns zeros
- [ ] Verify two rapid `create_version_dir()` calls produce different directories
- [ ] Verify `prune_old(keep_n=3)` with 10 versions retains exactly 3
- [ ] Verify `from models import ModelTrainer` works
