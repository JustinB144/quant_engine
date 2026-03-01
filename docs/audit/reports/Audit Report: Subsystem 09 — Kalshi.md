Findings (Severity-Ranked)
F-01 — api/services/kalshi_service.py is incompatible with EventTimeStore API and schema [P1 HIGH]
Evidence: api/services/kalshi_service.py (line 27) calls store.query_df("kalshi_markets", limit=200) and api/services/kalshi_service.py (line 45) calls store.query_df(..., where=...), but kalshi/storage.py (line 637) only supports (sql, params=None).
Evidence: api/services/kalshi_service.py (line 45) filters event_id on kalshi_contracts, but kalshi_contracts schema does not include event_id (kalshi/storage.py (line 193)).
Reproduction: with enabled=True, both methods return runtime errors (unexpected keyword argument 'limit'/'where').
Impact: Boundary api_to_kalshi_44 is broken when Kalshi is enabled; API consumer contract is not currently valid.
F-02 — Post-repair constraint metric is incorrect in distribution reconstruction [P2 MEDIUM]
Evidence: Threshold monotonic repair is applied (kalshi/distribution.py (line 694), kalshi/distribution.py (line 700)), but violated_constraints_post is set to pre-violation count whenever any isotonic adjustment occurs (kalshi/distribution.py (line 803)).
Impact: Diagnostics misreport repaired curves as still violating constraints, which can mislead downstream monitoring and gating.
F-03 — Hard quality-gate contract is defined but not enforced in snapshot output [P2 MEDIUM]
Evidence: passes_hard_gates() states failing gates should produce NaN outputs (kalshi/quality.py (line 171)), but it is not called by distribution builder; snapshot still returns numeric moments with only quality_low=1 (kalshi/distribution.py (line 755), kalshi/distribution.py (line 772)).
Impact: Consumers that ignore quality_low can ingest low-integrity features despite stated hard-gate semantics.
F-04 — Event-type purge config exists but is not wired into default Kalshi walk-forward path [P2 MEDIUM]
Evidence: Config exposes placeholders config.py (line 306), config.py (line 307), but default walk-forward config has empty purge_window_by_event (kalshi/walkforward.py (line 25)).
Evidence: Pipeline runner uses default EventWalkForwardConfig() without injecting event-type purge map (run_kalshi_event_pipeline.py (line 182)).
Impact: Intended event-type-aware purge behavior is not active unless callers manually pass it.
F-05 — current_version() can return a non-latest mapping version [P3 LOW]
Evidence: kalshi/mapping_store.py (line 58) takes iloc[-1] from as-of query results, but as-of query ordering is by event_id, market_id, mapping_version DESC (kalshi/storage.py (line 633)), not globally by recency/version.
Impact: Utility can report an arbitrary terminal row’s version in multi-event datasets.
Open Questions / Assumptions
data/kalshi.duckdb is currently absent in this workspace, so live artifact schema validation was done from DDL/source contracts rather than on-disk DB introspection.
KALSHI_ENABLED defaults to False (config.py (line 68)); API contract breakage is latent until enabled.
Summary
Reviewed all 25 Kalshi files listed in audit scope, 6096/6096 lines.
Verified all required spec commands and boundary checks.
Ran Kalshi test suite: PYTHONPATH=.:.. ./.venv/bin/pytest -q kalshi/tests -rs → 45 passed, 3 skipped (all skipped were RSA signature tests due missing cryptography dependency).
Scope Coverage
Production files reviewed: 16 files, 5208 lines.
Test files reviewed: 9 files, 888 lines.
Total reviewed: 25 files, 6096 lines.
Contract Verdicts
Leakage / as-of correctness: PASS
Backward merge_asof joins and explicit leakage guard (asof_ts < release_ts) are implemented and tested.
Storage schema compatibility (18-table contract): PARTIAL / FAIL at API boundary
Storage DDL is coherent and includes all 18 tables, but API consumer KalshiService does not match storage query API.
Distribution + quote-quality contract: PARTIAL
Core reconstruction and stale policy exist; post-repair reporting and hard-gate enforcement are inconsistent with documented contract semantics.
Promotion / walk-forward integration: PASS with caveat
PromotionDecision, PromotionGate.evaluate_event_strategy, StrategyCandidate, BacktestResult, deflated_sharpe_ratio, monte_carlo_validation, and compute_option_surface_factors interfaces are wired correctly.
Event-type purge config is not wired into default runner path.
Acceptance Criteria Check
100% line coverage across 25 Kalshi files: PASS
As-of/leakage contracts validated with tests: PASS
Storage/distribution schema stability confirmed: PARTIAL (API boundary break + quality-gate contract mismatch)
Promotion/walk-forward interfaces validated: PASS
Remediation Plan (No Code Changes Applied Yet)
Fix api/services/kalshi_service.py to use SQL strings + params, align filters with actual schema (market_id), and remove string-interpolated filtering.
Correct violated_constraints_post semantics to reflect post-repair state.
Enforce hard-gate behavior in distribution output path (NaN-out invalid snapshots or centrally apply passes_hard_gates()).
Wire KALSHI_PURGE_WINDOW_BY_EVENT / KALSHI_DEFAULT_PURGE_WINDOW into default Kalshi walk-forward config path.
Make current_version() deterministic (global max by version timestamp / mapping version).