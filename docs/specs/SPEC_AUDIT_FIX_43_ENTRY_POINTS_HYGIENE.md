# SPEC_AUDIT_FIX_43: Entry Points — Reproducibility, CLI Consistency & Tooling Hygiene

**Priority:** MEDIUM — Reproducibility manifests overwritten and incomplete; CLI defaults inconsistent with config; type generation incomplete; audit tooling paths broken.
**Scope:** `utils/reproducibility.py`, `run_backtest.py`, `run_predict.py`, `run_retrain.py`, `run_autopilot.py`, `run_server.py`, `run_kalshi_event_pipeline.py`, `run_wrds_daily_refresh.py`, `scripts/generate_types.py`, `scripts/hotspot_scoring.py`, `scripts/ibkr_daily_gapfill.py`
**Estimated effort:** 3–4 hours
**Depends on:** SPEC_42 (high-priority entry-points fixes)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 11 (Entry Points & Scripts) identified medium and low priority findings covering reproducibility manifest fragility, CLI argument inconsistencies, incomplete type generation, broken audit tooling paths, and fragile import patterns. These issues don't cause runtime failures but degrade reproducibility, maintainability, and developer experience.

### Cross-Audit Traceability

| Finding | Auditor 1 (Codex) | Auditor 2 (Claude Opus) | Disposition |
|---------|-------------------|------------------------|-------------|
| Manifests overwritten + lack provenance | F-04 (P2) | — | **NEW → T1** |
| generate_types.py hardcoded to 2 modules | F-05 (P2) | — | **NEW → T2** |
| Audit tooling paths out of sync | F-08 (P2) | — | **NEW → T3** |
| run_server.py import pattern inconsistency | — | F3 (P2) | **NEW → T4** |
| --years hardcoded instead of LOOKBACK_YEARS | F-09 (P3) | F12 (P3) | **NEW → T5** |
| feature-mode missing "minimal" in predict/retrain | — | F7/F8 (P3) | **NEW → T5** |
| run_predict.py hardcodes years=2 | F-10 (P3) | — | **NEW → T5** |
| Missing repro manifests in kalshi/wrds | — | F9/F10 (P3) | **NEW → T6** |
| ibkr_daily_gapfill private function imports | — | F6 (P2) | **NEW → T7** |
| run_retrain.py calls private trigger._save_metadata() | — | F11 (P3) | **NEW → T7** |
| importlib.util bypass in alpaca/ibkr scripts | — | F4/F5 (P2) | **Deferred** |

### Deferred Items

| ID | File | Description | Rationale for Deferral |
|----|------|-------------|----------------------|
| Aud2 F4/F5 | alpaca_intraday_download.py, ibkr_intraday_download.py | importlib.util namespace bypass | Intentional standalone script design; scripts run outside the quant_engine namespace. Auditor 2 agrees: "acceptable for standalone scripts but should be documented as intentional." |

---

## Tasks

### T1: Make Reproducibility Manifests Unique and Include Runtime Provenance

**Problem:** `utils/reproducibility.py:102` uses a constant default filename `"run_manifest.json"`. All 5 entry points that call `write_run_manifest()` use the default, so every run overwrites the same file. Additionally, manifests only capture CLI arguments — they don't include resolved model version IDs, dataset checksums, or config snapshots.

**Files:** `utils/reproducibility.py`, `run_backtest.py`, `run_train.py`, `run_predict.py`, `run_retrain.py`, `run_autopilot.py`

**Implementation:**
1. Change the default filename to include a timestamp:
   ```python
   from datetime import datetime

   def write_run_manifest(
       manifest: Dict[str, Any],
       output_dir: Path,
       filename: Optional[str] = None,
   ) -> Path:
       if filename is None:
           ts = datetime.now().strftime("%Y%m%d_%H%M%S")
           filename = f"run_manifest_{ts}.json"
       output_dir.mkdir(parents=True, exist_ok=True)
       path = output_dir / filename
       with open(path, "w") as f:
           json.dump(manifest, f, indent=2, default=str)
       # Also write/overwrite a "latest" symlink for convenience
       latest = output_dir / "run_manifest_latest.json"
       try:
           if latest.is_symlink() or latest.exists():
               latest.unlink()
           latest.symlink_to(path.name)
       except OSError:
           pass  # Symlinks may not be supported on all platforms
       return path
   ```

2. Enhance the manifest builder to capture runtime provenance:
   ```python
   def build_run_manifest(args, script_name: str, **extra) -> Dict[str, Any]:
       manifest = {
           "script": script_name,
           "timestamp": datetime.now().isoformat(),
           "cli_args": vars(args),
           "python_version": sys.version,
           "platform": platform.platform(),
       }
       # Add resolved runtime provenance if available
       try:
           from quant_engine.config import RESULTS_DIR
           model_version_path = Path(RESULTS_DIR) / "model_version.txt"
           if model_version_path.exists():
               manifest["model_version"] = model_version_path.read_text().strip()
       except Exception:
           pass
       manifest.update(extra)
       return manifest
   ```

3. Update each entry point to pass `script_name` and any relevant resolved parameters:
   ```python
   # In run_backtest.py:
   manifest = build_run_manifest(
       args, script_name="run_backtest",
       effective_entry_threshold=effective_entry,
       effective_confidence=effective_conf,
       result_total_trades=result.total_trades,
   )
   ```

**Acceptance:** Each run creates a uniquely timestamped manifest file. Manifest includes resolved config values, not just CLI args. A `run_manifest_latest.json` symlink points to the most recent run.

---

### T2: Auto-Discover All Schema Modules in generate_types.py

**Problem:** `scripts/generate_types.py:25-28` hardcodes only 2 schema modules (`api.schemas.envelope`, `api.schemas.compute`), but `api/schemas/` contains many more Pydantic model files (autopilot.py, backtests.py, dashboard.py, data_explorer.py, model_lab.py, signals.py, system_health.py). New schemas are silently excluded from TypeScript generation.

**File:** `scripts/generate_types.py`

**Implementation:**
1. Replace the hardcoded list with dynamic discovery:
   ```python
   import importlib
   import pkgutil

   def discover_schema_modules() -> list:
       """Auto-discover all modules in api.schemas package."""
       import api.schemas as schemas_pkg
       modules = []
       for importer, modname, ispkg in pkgutil.iter_modules(schemas_pkg.__path__):
           if modname.startswith("_"):
               continue  # Skip __init__.py etc.
           full_name = f"api.schemas.{modname}"
           modules.append(full_name)
       return sorted(modules)

   SCHEMA_MODULES = discover_schema_modules()
   ```
2. Fall back to the hardcoded list if dynamic discovery fails (e.g., import error):
   ```python
   try:
       SCHEMA_MODULES = discover_schema_modules()
   except Exception:
       logger.warning("Auto-discovery failed; using fallback schema list")
       SCHEMA_MODULES = [
           "api.schemas.envelope",
           "api.schemas.compute",
       ]
   ```
3. Log which modules were discovered so discrepancies are visible.

**Acceptance:** Running `generate_types.py` discovers and processes ALL schema modules in `api/schemas/`. Adding a new schema file to that directory automatically includes it in type generation.

---

### T3: Fix Audit Tooling Path References

**Problem:** `scripts/hotspot_scoring.py:81` reads `docs/audit/DEPENDENCY_EDGES.json`, but the actual file lives at `docs/audit/data/DEPENDENCY_EDGES.json`. The script fails with `FileNotFoundError` on execution. Similarly, `extract_dependencies.py:569-583` and `generate_interface_contracts.py:2549` write to `docs/audit/` root when the canonical output location is `docs/audit/data/`.

**Files:** `scripts/hotspot_scoring.py`, `scripts/extract_dependencies.py`, `scripts/generate_interface_contracts.py`

**Implementation:**
1. Fix the read path in `hotspot_scoring.py`:
   ```python
   def load_edges():
       edges_path = ROOT / "docs" / "audit" / "data" / "DEPENDENCY_EDGES.json"
       if not edges_path.exists():
           # Fallback to legacy path for backward compatibility
           edges_path = ROOT / "docs" / "audit" / "DEPENDENCY_EDGES.json"
       with open(edges_path) as f:
           data = json.load(f)
       return data["edges"]
   ```
2. Verify `extract_dependencies.py` and `generate_interface_contracts.py` write paths match the canonical `docs/audit/data/` location. Update if they write to the wrong directory.
3. Add a comment documenting the canonical data layout:
   ```python
   # Canonical audit data layout:
   # docs/audit/data/DEPENDENCY_EDGES.json  — Job 2 output
   # docs/audit/data/INTERFACE_CONTRACTS.yaml — Job 3 output
   ```

**Acceptance:** `hotspot_scoring.py` executes without `FileNotFoundError`. All audit tooling scripts read from and write to consistent paths under `docs/audit/data/`.

---

### T4: Standardize run_server.py Import Pattern

**Problem:** `run_server.py:6-7` uses bare imports (`from api.config import ApiSettings`) without the `sys.path.insert(0, ...)` pattern used by all other 8 production entry points. Other entry points use:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from quant_engine.config import ...
```
This means `run_server.py` only works when the current working directory IS the project root — it fails when invoked from any other directory (e.g., during deployment).

**File:** `run_server.py`

**Implementation:**
1. Add the standard sys.path setup and use the `quant_engine.*` namespace:
   ```python
   import sys
   from pathlib import Path

   # Ensure quant_engine is importable regardless of CWD
   sys.path.insert(0, str(Path(__file__).resolve().parent))

   def main() -> None:
       # ...
       from quant_engine.api.config import ApiSettings
       from quant_engine.api.main import create_app
       # ...
   ```
2. Keep the imports lazy (inside `main()`) to maintain the existing lazy-import pattern.

**Acceptance:** `run_server.py` works when invoked from any directory, not just the project root. Uses the same `quant_engine.*` namespace as all other entry points.

---

### T5: Fix CLI Argument Defaults and Choices Inconsistencies

**Problem:** Three related CLI inconsistencies across entry points:

1. `run_backtest.py:64` and `run_retrain.py:85` hardcode `--years` default to `15` instead of using the `LOOKBACK_YEARS` config constant. `run_train.py` correctly uses `LOOKBACK_YEARS`. If the config constant changes, backtest/retrain will use the old value.

2. `run_predict.py` and `run_retrain.py` define `--feature-mode` choices as `["core", "full"]`, missing `"minimal"`. `run_backtest.py` and `run_train.py` correctly include all three: `["minimal", "core", "full"]`.

3. `run_predict.py:97` hardcodes `years=2` in the `load_universe()` call with no corresponding CLI argument, making it impossible to override.

**Files:** `run_backtest.py`, `run_predict.py`, `run_retrain.py`

**Implementation:**
1. Replace hardcoded `--years` defaults with the config constant:
   ```python
   # In run_backtest.py and run_retrain.py:
   from quant_engine.config import LOOKBACK_YEARS

   parser.add_argument("--years", type=int, default=LOOKBACK_YEARS,
                        help=f"Lookback years (default: {LOOKBACK_YEARS})")
   ```

2. Add `"minimal"` to feature-mode choices in predict and retrain:
   ```python
   # In run_predict.py and run_retrain.py:
   parser.add_argument("--feature-mode", choices=["minimal", "core", "full"], default="core")
   ```

3. Surface the years parameter in run_predict.py:
   ```python
   # In run_predict.py:
   parser.add_argument("--years", type=int, default=2,
                        help="Data lookback years for prediction (default: 2)")
   # Then use: load_universe(... years=args.years ...)
   ```

**Acceptance:** All entry points use `LOOKBACK_YEARS` for `--years` default. All entry points that accept `--feature-mode` include `"minimal"` as a choice. `run_predict.py --years 5` correctly loads 5 years of data.

---

### T6: Add Reproducibility Manifests to Kalshi and WRDS Entry Points

**Problem:** `run_kalshi_event_pipeline.py` and `run_wrds_daily_refresh.py` do not import or use the reproducibility module. All other production entry points generate manifests.

**Files:** `run_kalshi_event_pipeline.py`, `run_wrds_daily_refresh.py`

**Implementation:**
1. Add manifest generation to both scripts, following the pattern used by `run_backtest.py`:
   ```python
   # At the end of main(), after pipeline completes:
   try:
       from quant_engine.utils.reproducibility import build_run_manifest, write_run_manifest
       manifest = build_run_manifest(args, script_name="run_kalshi_event_pipeline")
       write_run_manifest(manifest, output_dir=RESULTS_DIR)
   except Exception as e:
       logger.warning("Could not write reproducibility manifest: %s", e)
   ```
2. For `run_wrds_daily_refresh.py`, include the refresh date range and ticker count in the manifest:
   ```python
   manifest = build_run_manifest(
       args, script_name="run_wrds_daily_refresh",
       tickers_refreshed=len(refreshed_tickers),
       refresh_date=datetime.now().isoformat(),
   )
   ```

**Acceptance:** Running `run_kalshi_event_pipeline.py` and `run_wrds_daily_refresh.py` both produce reproducibility manifest files.

---

### T7: Expose Public API for Functions Used by External Scripts

**Problem:** Two private API violations:
1. `scripts/ibkr_daily_gapfill.py` imports `_normalize_ohlcv_columns` and `_write_cache_meta` from `data/local_cache.py` — these are private functions (prefixed with `_`).
2. `run_retrain.py` calls `trigger._save_metadata()` — a private method on `RetrainTrigger`.

Both are coupling to private internals that may change without notice.

**Files:** `data/local_cache.py`, `models/retrain_trigger.py`

**Implementation:**
1. In `data/local_cache.py`, rename the private functions to public (drop the underscore) or create public wrappers:
   ```python
   # Option A: Rename to public
   def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
       ...  # existing _normalize_ohlcv_columns logic

   def write_cache_meta(path, *, ticker, df, source, meta=None):
       ...  # existing _write_cache_meta logic

   # Keep aliases for backward compatibility:
   _normalize_ohlcv_columns = normalize_ohlcv_columns
   _write_cache_meta = write_cache_meta
   ```
2. Export them in `data/__init__.py` if appropriate.
3. In `models/retrain_trigger.py`, add a public `save_metadata()` method:
   ```python
   def save_metadata(self, *args, **kwargs):
       """Public API for persisting retrain metadata."""
       return self._save_metadata(*args, **kwargs)
   ```
4. Update `run_retrain.py` and `ibkr_daily_gapfill.py` to use the public APIs.

**Acceptance:** `ibkr_daily_gapfill.py` imports public functions (no underscore prefix). `run_retrain.py` calls `trigger.save_metadata()` (public). No private API access from external scripts.

---

## Verification

- [ ] Run all entry points — no import errors or FileNotFoundError
- [ ] Verify manifest files have unique timestamps and runtime provenance
- [ ] Run `generate_types.py` — verify all `api/schemas/*.py` modules are discovered
- [ ] Run `hotspot_scoring.py` — verify no FileNotFoundError
- [ ] Verify `run_server.py` works from non-project-root directory
- [ ] Verify `--years` defaults match `LOOKBACK_YEARS` config
- [ ] Verify `--feature-mode minimal` works in predict and retrain
