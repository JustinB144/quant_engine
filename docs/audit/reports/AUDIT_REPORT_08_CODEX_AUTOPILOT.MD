Severity-Ranked Findings

P0 — Contract drift in statistical validation path causes runtime failures in candidate evaluation
Evidence: engine.py (line 881) calls run_statistical_tests(...) without required trade_returns, while provider requires it at validation.py (line 393); engine only catches ValueError/RuntimeError at engine.py (line 887), so TypeError is uncaught. Engine also references non-existent fields: overall_pass engine.py (line 886), is_significant engine.py (line 902), mean_test_corr engine.py (line 947), rejects_null engine.py (line 915), but provider dataclasses expose passes and mean_oos_corr at validation.py (line 163), validation.py (line 179), validation.py (line 174), validation.py (line 193).
Repro executed (venv): TypeError run_statistical_tests() missing 1 required positional argument: 'trade_returns'.
Impact: autopilot cycle can fail during promotion contract checks.

P1 — API job runner is incompatible with current AutopilotEngine constructor
Evidence: API job uses AutopilotEngine(years=..., full_universe=...) at autopilot_job.py (line 17), but engine requires tickers and has no full_universe parameter at engine.py (line 150).
Repro executed (venv): TypeError ... unexpected keyword argument 'full_universe'.
Impact: API-triggered autopilot job path is broken.

P1 — Drawdown controller state is mutated during sizing probes
Evidence: _position_size_pct calls self._dd_controller.update(0.0) at paper_trader.py (line 709); normal cycle already updates drawdown once at paper_trader.py (line 762). update() mutates _total_days, equity history, and state at drawdown.py (line 102).
Impact: recovery/state progression can advance artificially within entry loops, distorting risk controls.

P2 — State persistence is non-atomic and corruption handling is missing
Evidence: direct JSON writes in paper_trader.py (line 249) and registry.py (line 48); direct json.load without decode recovery in paper_trader.py (line 235) and registry.py (line 42).
Impact: partial writes can break restart/recovery paths.

P2 — paper_state.json producer/consumer schema mismatch for utilization health check
Evidence: consumer expects equity and per-position value at health_service.py (line 1463) and health_service.py (line 1476). Producer position schema lacks value at paper_trader.py (line 1171), and persisted state does not include top-level equity at paper_trader.py (line 245).
Impact: capital utilization can be materially under-reported.

T1–T7 Check Closure

T1 Ledger/schema baseline: Completed. Artifacts and consumers mapped.
T2 Orchestration correctness: Failed (P0 contract drift in statistical tests path).
T3 Paper-trading parity/risk: Partial pass; execution kwargs parity is correct (paper_trader.py (line 113) vs backtest/engine.py (line 210)), but P1/P2 issues above remain.
T4 API coupling/degradation safety: Partial pass. All 6 autopilot→api imports are lazy and try-guarded (paper_trader.py (line 173), paper_trader.py (line 189), paper_trader.py (line 211), paper_trader.py (line 532), engine.py (line 1868), engine.py (line 1911)); reverse edge job runner is broken (P1).
T5 Meta-labeling/allocation: Passed. Fail-open behavior and safe defaults observed in engine.py (line 1058), engine.py (line 1074), strategy_allocator.py (line 143).
T6 Boundary contracts: Failed where noted below.
T7 Findings publication/closure: Completed (this report), closure gate is blocked by unresolved P0/P1.
Explicit Boundary Dispositions

autopilot_to_multi_4: FAIL due validation API/field drift in autopilot engine statistical checks (P0).
autopilot_to_api_circular_5: PARTIAL PASS. Forward circular imports are correctly lazy/guarded; operational path still fails via reverse edge contract drift (api_to_autopilot_43, P1).
Shared Artifact Schema Baseline

results/autopilot/latest_cycle.json: observed keys horizon, years, feature_mode, strict_oos, survivorship_mode, n_candidates, n_passed, n_promoted, n_active, top_decisions, paper_report; consumed by autopilot_service.py (line 14) and IC fallback extraction at health_service.py (line 2459).
results/autopilot/strategy_registry.json: observed keys active, history; consumed by autopilot_service.py (line 30).
results/autopilot/paper_state.json: observed keys cash, realized_pnl, positions, trades, last_update; consumed by autopilot_service.py (line 48) and health checks at health_service.py (line 1350), health_service.py (line 1437). Schema mismatch noted in Finding 5.
Acceptance Criteria Status (Spec)

Full line coverage across all 8 autopilot files: PASS (4480/4480).
Promotion, paper-trading, artifact contracts validated: PASS with findings.
API coupling degradation paths verified: PASS with findings.
No unresolved P0/P1 in hotspot files: FAIL.
Gate
Subsystem 8 is not ready for change-free acceptance. Unresolved P0/P1 issues remain.

If you want, I can produce a remediation patch plan ordered by blast radius (P0 first) without applying code changes.
If you want, I can then implement fixes in a separate pass with targeted tests for each finding.