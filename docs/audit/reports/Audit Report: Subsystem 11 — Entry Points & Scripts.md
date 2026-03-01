Executive Summary
I reviewed all 17 in-scope files line-by-line and verified each T1–T6 check from the spec.

In-scope files reviewed: 17/17
Current line coverage: 9813/9813 (baseline in map/spec is 8883; current files have grown)
Findings:
P0: 0
P1: 2
P2: 6
P3: 2
Highest-risk issues are:

Shared backtest summary contract drift (11 emitted fields vs 17-field contract record).
Stale result artifact risk when a backtest produces zero trades.
Findings (Severity-Ranked)
F-01 — Backtest summary contract drift (11 emitted fields vs 17-field contract) [P1]
Evidence:
run_backtest.py (line 427)-439 writes 11 fields.
api/orchestrator.py (line 349)-360 mirrors same 11 fields.
INTERFACE_CONTRACTS.yaml (line 1064)-1089 declares 17 fields (winning_trades, losing_trades, avg_win, avg_loss, total_return, avg_holding_days, etc.).
evaluation/engine.py has no summary-reader path (only output write at evaluation/engine.py (line 551)).

Impacted consumers:
backtest_service.py, results_service.py, audit contract artifacts.

F-02 — Zero-trade runs can leave stale summary/trade artifacts [P1]
Evidence:
run_backtest.py (line 396)-442: file writes happen only under if result.total_trades > 0.
No else-branch clears or rewrites summary/trades for zero-trade output.
Same pattern in API backtest path: api/orchestrator.py (line 332)-363.

Impacted consumers:
backtest_service.py, results_service.py, dashboard/API users.

F-03 — SPA validation uses a different confidence threshold than executed backtest trades [P2]
Evidence:
run_backtest.py (line 338) sets effective_conf = ... else 0.0 for SPA path.
run_backtest.py (line 366) applies min_confidence=effective_conf to strategy_signal_returns.
Backtester default threshold is config-driven (backtest/engine.py (line 183)).

Impacted consumers:
Backtest statistical diagnostics and any user decisions based on SPA output.

F-04 — Reproducibility manifests are overwritten and omit key resolved runtime provenance [P2]
Evidence:
write_run_manifest default filename is constant run_manifest.json (reproducibility.py (line 102)).
Run scripts call it without unique filenames: run_train.py (line 215), run_backtest.py (line 454), run_predict.py (line 227), run_retrain.py (line 312), run_autopilot.py (line 114).
Manifests are built from CLI snapshots only (e.g., run_backtest.py (line 105)-108), not resolved model version IDs/checksum datasets.

Impacted consumers:
Reproducibility/replay workflows and auditability.

F-05 — generate_types.py does not introspect all schema modules as claimed [P2]
Evidence:
Docstring claims all api/schemas models (generate_types.py (line 6)-8).
Hardcoded module list only includes two modules (generate_types.py (line 25)-28).
api/schemas contains many additional Pydantic model files (e.g., autopilot.py, backtests.py, dashboard.py, etc.).

Impacted consumers:
Frontend generated type coverage and schema drift detection.

F-06 — Alpaca intraday utility writes to non-canonical cache root used by default loaders [P2]
Evidence:
Script targets DATA_CACHE_ALPACA_DIR (alpaca_intraday_download.py (line 59), alpaca_intraday_download.py (line 1010)).
Default intraday loader reads DATA_CACHE_DIR (local_cache.py (line 391)).
Config confirms separate roots (config.py (line 40)-41).

Impacted consumers:
Feature/backtest flows that rely on default intraday cache lookup.

F-07 — Merge logic can drop required OHLCV columns if existing file schema is degraded [P2]
Evidence:
alpaca_intraday_download.py (line 699)-703,
ibkr_intraday_download.py (line 482)-487,
ibkr_daily_gapfill.py (line 220)-225 merge on column intersection without asserting all required OHLCV columns remain.

Impacted consumers:
Downstream cache readers expecting full OHLCV schema.

F-08 — Audit tooling paths are out of sync with current docs/audit/data layout [P2]
Evidence:
hotspot_scoring.py (line 81)-84 reads docs/audit/DEPENDENCY_EDGES.json and currently fails (FileNotFoundError in execution).
extract_dependencies.py (line 569)-583 writes to docs/audit/ root.
generate_interface_contracts.py (line 2549) writes to docs/audit/ root.
Canonical pipeline outputs are documented under docs/audit/data (INPUT_CONTEXT.md (line 12)-18).

Impacted consumers:
Audit automation jobs and artifact consistency.

F-09 — years=15 defaults are duplicated as literals instead of sourced from config constant [P3]
Evidence:
run_backtest.py (line 64), run_retrain.py (line 85), run_autopilot.py (line 36) vs config.py (line 168) (LOOKBACK_YEARS).

Impacted consumers:
CLI consistency and future config drift risk.

F-10 — run_predict.py hardcodes years=2 without CLI/config surface [P3]
Evidence:
run_predict.py (line 97) calls load_universe(... years=2 ...) with no corresponding CLI arg.

Impacted consumers:
Predict script behavior transparency and reproducibility expectations.

T1 — Immutable Audit Baseline
jq -r '.subsystems.entry_points_scripts.files[]' ... | xargs wc -l result:

9813 total lines reviewed (vs map baseline 8883)
Line Ledger
File	start_line	end_line	reviewer	status	category
run_train.py	1	227	Codex	reviewed	production entry point
run_backtest.py	1	464	Codex	reviewed	production entry point
run_predict.py	1	237	Codex	reviewed	production entry point
run_autopilot.py	1	129	Codex	reviewed	production entry point
run_retrain.py	1	322	Codex	reviewed	production entry point
run_server.py	1	94	Codex	reviewed	production entry point
run_kalshi_event_pipeline.py	1	227	Codex	reviewed	production entry point
run_wrds_daily_refresh.py	1	915	Codex	reviewed	production entry point
run_rehydrate_cache_metadata.py	1	101	Codex	reviewed	production entry point
scripts/alpaca_intraday_download.py	1	1618	Codex	reviewed	data utility
scripts/ibkr_intraday_download.py	1	1022	Codex	reviewed	data utility
scripts/ibkr_daily_gapfill.py	1	417	Codex	reviewed	data utility
scripts/compare_regime_models.py	1	323	Codex	reviewed	data utility
scripts/generate_types.py	1	146	Codex	reviewed	data utility
scripts/extract_dependencies.py	1	600	Codex	reviewed	audit tooling
scripts/generate_interface_contracts.py	1	2570	Codex	reviewed	audit tooling
scripts/hotspot_scoring.py	1	401	Codex	reviewed	audit tooling
T2 — Shared Artifact Schema Verification
Status: FAIL (contract drift found)

Writer checked: run_backtest.py
Consumer cross-checks:
api/services/backtest_service.py
api/services/results_service.py
evaluation/engine.py
Key result:

Current emitter writes 11-field payload.
Interface contract file declares 17-field payload.
evaluation/engine.py no longer reads this summary artifact directly.
Advanced-validation conditional import at run_backtest.py (line 382) does not alter summary schema.
T3 — Module Constructor Wiring Pass
Status: PASS (no constructor/signature mismatches found in entry-point wiring)

Validated calls in:

run_train.py
run_backtest.py
run_predict.py
run_retrain.py
run_autopilot.py
Cross-checked against:

features/pipeline.py
regime/detector.py
models/predictor.py
models/trainer.py
backtest/engine.py
autopilot/engine.py
api/orchestrator.py
Note:

PositionSizer.size_position() (21 params) is not directly called by these entry scripts; it is consumed inside backtest/autopilot internals.
T4 — CLI Defaults and Reproducibility Pass
Status: PARTIAL FAIL

Passes:

Argparse wiring is coherent and scripts run expected flow shapes.
Kalshi entry point correctly wires distribution/events/pipeline/promotion/walkforward modules.
Issues:

Hardcoded default duplication (years=15) vs config constant (F-09).
run_predict fixed 2-year load window not surfaced in CLI/config (F-10).
Repro manifest persistence/provenance gaps (F-04).
T5 — Data Utility and Audit Tooling Review
Status: PARTIAL FAIL

Passes:

Data utilities generally produce parquet + metadata sidecars with OHLCV handling.
compare_regime_models.py does not mutate trained model artifacts.
Tooling scripts are non-production runtime code paths.
Issues:

Non-canonical cache root for Alpaca utility (F-06).
Merge-column intersection can persist degraded schema (F-07).
Audit tooling path drift/breakage in current layout (F-08).
Type generation utility does not cover all schema modules (F-05).
T6 — Final Risk Closure
Unresolved High Risks (must-fix before “clean” closure)
F-01 (P1): backtest summary contract drift.
F-02 (P1): stale summary/trade artifacts on zero-trade runs.
Recommended follow-up tests
Contract test: validate emitted backtest_*d_summary.json against one canonical schema.
Zero-trade regression: ensure summary/trades files are rewritten/cleared deterministically.
SPA parity test: assert SPA input signal filter equals executed backtest thresholds.
Manifest retention test: verify per-run unique manifest files and resolved model version capture.
Typegen coverage test: ensure all api/schemas/*.py BaseModel classes are emitted.
Cache schema guard tests: assert saved parquet always contains full OHLCV set.
Acceptance Criteria Check
100% line coverage of all 17 files: PASS (9813/9813 current lines reviewed).
Summary schema verified vs 3 downstream consumers: FAIL (contract drift and stale consumer metadata).
Constructor calls match upstream signatures: PASS.
CLI defaults consistent with config constants: PARTIAL FAIL (hardcoded drift risks).
Repro manifests capture relevant runtime parameters: PARTIAL FAIL.
No unresolved P0/P1 findings remain: FAIL (2 unresolved P1 findings).
Tooling scripts are non-production/no runtime side effects: PASS (but tooling-path correctness issues remain).
Verification Notes
Executed the exact spec verification commands for T1–T5.
Also executed python3 scripts/hotspot_scoring.py to validate tooling-path health; it fails due missing docs/audit/DEPENDENCY_EDGES.json in current repo layout.
User-provided context path DEPENDENCY_MAT does not exist; canonical file is DEPENDENCY_MATRIX.md.
You are implementing fixes to a Python quantitative trading engine called quant_engine, your job is to go the the entirety of specs/SPEC_AUDIT_FIX_09_EVALUATION_ENGINE.md and follow the spec exactly. RULES: Read the spec FULLY before writing any code. Do NOT guess file contents. Read every file the spec references before modifying it. Do NOT add features not described in the spec. Do NOT refactor code that the spec does not mention. Preserve ALL existing functionality that the spec does not explicitly change. Follow existing code conventions (logging patterns, import style, docstring format). When the spec says 'verify' or 'check', actually read the source code and confirm. If the spec references another spec (e.g., 'coordinate with SPEC_15'), note the dependency but do NOT implement that other spec's changes. Run tests after implementation: pytest tests/ -k '<subsystem>' -v. Write a brief summary of every change you made, file by file. CODING CONVENTIONS: Use logging module (not print statements) for all output.Use absolute imports from project root. Use type hints on all new function signatures. Use numpy/pandas idioms (vectorized ops over loops where possible). Follow the atomic write patterns(tempfile.mkstemp + os.replace) for any file persistence. Config constants go in config.py; structured config in config_structured.py. All new features must have an entry in FEATURE_METADATA with causality type
