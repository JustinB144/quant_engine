Subsystem 10 Audit Report (API & Frontend)
Spec followed: SPEC_AUDIT_10_API_FRONTEND.md
Overall status: FAIL (pre-change)

Findings (ordered by severity)
[P1] Router envelope contract is violated on 2 endpoints
Evidence: risk.py (line 20), risk.py (line 34), diagnostics.py (line 20), diagnostics.py (line 41), frontend envelope expectation at client.ts (line 34) and client.ts (line 36).
Impact: /api/risk/factor-exposures and /api/diagnostics return raw dicts, not ApiResponse. Any consumer using the shared frontend client will treat these as failed requests.

[P1] Frontend/backed job status contract drift breaks job lifecycle UX
Evidence: backend statuses/fields at models.py (line 11), models.py (line 29), store.py (line 119); frontend expects different contract at jobs.ts (line 1), jobs.ts (line 11), useJobs.ts (line 21), useJobProgress.ts (line 21), JobMonitor.tsx (line 15).
Impact: backend uses queued/succeeded/progress_message; frontend expects pending/completed/message. Polling/SSE activity and completion handling are inconsistent.

[P1] SSE event wiring mismatch prevents frontend from receiving job events
Evidence: backend emits named SSE events at jobs.py (line 57); frontend only listens to onmessage at useSSE.ts (line 34).
Impact: named events (progress, completed, failed, etc.) are not handled by onmessage; progress stream is effectively dropped.

[P1] Compute request contracts drifted between frontend and API schemas
Evidence: frontend request types at compute.ts (line 1), usage at BacktestPage.tsx (line 27), TrainingTab.tsx (line 49), SignalControls.tsx (line 16), backend schema at compute.py (line 9).
Impact: fields like survivorship_filter, holding_period, max_positions, entry_threshold, position_size, top_n, dry_run are not part of backend models and are silently ignored.

[P2] Cancelled queued jobs can leave SSE subscribers waiting indefinitely
Evidence: cancel path at runner.py (line 83) and store.py (line 124), stream loop at runner.py (line 93).
Impact: for non-running jobs cancelled via store update, no explicit emitted terminal event from runner; stream can block after initial status snapshot.

[P2] Some artifact readers do not handle corrupt files gracefully
Evidence: unguarded json.load/read_csv in backtest_service.py (line 24), backtest_service.py (line 51), results_service.py (line 24), results_service.py (line 42).
Impact: malformed artifacts raise and bubble to HTTP 500 instead of returning a graceful unavailable/error payload.

[P3] Data refresh invalidation hook exists but is not wired
Evidence: definition at invalidation.py (line 21), no callsites found in API/frontend/run scripts.
Impact: refresh-triggered cache eviction relies only on TTL expiration, not event-driven invalidation.

Task/Spec check results (T1–T7)
T1 Ledger + baseline: PASS
59/59 files covered, 10,188/10,188 lines reviewed, 15 router modules mounted.
T2 Orchestrator/service contract pass: PASS with risks noted
Cross-subsystem symbols resolved; lazy imports verified; no P0/P1 in api/orchestrator.py or api/services/health_service.py.
T3 Job lifecycle pass: FAIL
Canonical backend statuses exist, but frontend contract mismatch + SSE mismatch + cancel-stream edge case.
T4 Router envelope consistency pass: FAIL
45 envelope endpoints, 1 SSE endpoint, 2 raw-dict endpoints (risk, diagnostics).
T5 Cache/invalidation pass: PARTIAL FAIL
Train/backtest/config invalidation wired; data-refresh invalidation not wired; corruption handling incomplete in some artifact readers.
T6 Circular coupling/boundary pass: PASS
All 6 autopilot -> api edges and reverse api/jobs/autopilot_job.py -> autopilot/engine.py are lazy/conditional and runtime-safe.
T7 Findings synthesis/release gate: FAIL
Open P1 findings remain.
Boundary verification (required contracts)
api_to_config_36: PASS (82 imports present; lazy/conditional cross-subsystem usage confirmed).
api_to_data_37: PASS (load_universe, load_survivorship_universe, get_skip_reasons, get_data_provenance, WRDSProvider present).
api_to_models_40: PASS (ModelGovernance, ModelTrainer, ModelRegistry, EnsemblePredictor, FeatureStabilityTracker, IV symbols present).
api_to_autopilot_43: PASS (AutopilotEngine exists; lazy import in autopilot_job.py (line 12)).
api_to_kalshi_44: PASS (EventTimeStore exists; API service handles Kalshi exceptions).
autopilot_to_api_circular_5: PASS (6 edges verified; guarded by lazy/conditional imports).
Mandatory contract-drift section (job status mismatch)
Backend canonical statuses: queued/running/succeeded/failed/cancelled at models.py (line 11).
Frontend assumes pending/running/completed/failed/cancelled at jobs.ts (line 1).
Backend uses progress_message; frontend reads message.
This drift directly affects polling cadence, “active” detection, and completion callbacks.
Critical-file must-not gate (from spec)
api/services/health_service.py: No P0/P1 found
api/orchestrator.py: No P0/P1 found
api/jobs/autopilot_job.py: No P0/P1 found
api/schemas/envelope.py: No P0/P1 found
Line ledger (full coverage)
file,start_line,end_line,status
api/__init__.py,1,1,reviewed
api/main.py,1,148,reviewed
api/config.py,1,75,reviewed
api/errors.py,1,77,reviewed
api/ab_testing.py,1,794,reviewed
api/orchestrator.py,1,372,reviewed
api/deps/__init__.py,1,16,reviewed
api/deps/providers.py,1,54,reviewed
api/cache/__init__.py,1,4,reviewed
api/cache/manager.py,1,62,reviewed
api/cache/invalidation.py,1,32,reviewed
api/jobs/__init__.py,1,6,reviewed
api/jobs/store.py,1,145,reviewed
api/jobs/runner.py,1,115,reviewed
api/jobs/models.py,1,32,reviewed
api/jobs/autopilot_job.py,1,31,reviewed
api/jobs/backtest_job.py,1,29,reviewed
api/jobs/predict_job.py,1,28,reviewed
api/jobs/train_job.py,1,30,reviewed
api/routers/__init__.py,1,42,reviewed
api/routers/autopilot.py,1,83,reviewed
api/routers/backtests.py,1,89,reviewed
api/routers/benchmark.py,1,128,reviewed
api/routers/config_mgmt.py,1,146,reviewed
api/routers/dashboard.py,1,170,reviewed
api/routers/data_explorer.py,1,490,reviewed
api/routers/diagnostics.py,1,65,reviewed
api/routers/iv_surface.py,1,71,reviewed
api/routers/jobs.py,1,72,reviewed
api/routers/logs.py,1,38,reviewed
api/routers/model_lab.py,1,90,reviewed
api/routers/regime.py,1,133,reviewed
api/routers/risk.py,1,53,reviewed
api/routers/signals.py,1,55,reviewed
api/routers/system_health.py,1,121,reviewed
api/schemas/__init__.py,1,4,reviewed
api/schemas/autopilot.py,1,30,reviewed
api/schemas/backtests.py,1,56,reviewed
api/schemas/compute.py,1,56,reviewed
api/schemas/dashboard.py,1,38,reviewed
api/schemas/data_explorer.py,1,37,reviewed
api/schemas/envelope.py,1,54,reviewed
api/schemas/model_lab.py,1,43,reviewed
api/schemas/signals.py,1,26,reviewed
api/schemas/system_health.py,1,47,reviewed
api/services/__init__.py,1,20,reviewed
api/services/autopilot_service.py,1,62,reviewed
api/services/backtest_service.py,1,139,reviewed
api/services/data_helpers.py,1,1058,reviewed
api/services/data_service.py,1,208,reviewed
api/services/diagnostics.py,1,288,reviewed
api/services/health_alerts.py,1,324,reviewed
api/services/health_confidence.py,1,314,reviewed
api/services/health_risk_feedback.py,1,275,reviewed
api/services/health_service.py,1,2929,reviewed
api/services/kalshi_service.py,1,50,reviewed
api/services/model_service.py,1,98,reviewed
api/services/regime_service.py,1,50,reviewed
api/services/results_service.py,1,85,reviewed