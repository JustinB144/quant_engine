# Kalshi Storage Schema Reference

This document is a source-derived reference of the event-time storage schema defined in `kalshi/storage.py` (`EventTimeStore.init_schema`).

## Tables

### `event_market_map_versions`

Purpose: Versioned event-to-market mapping with effective windows.

- Columns: 8
- Constraints captured: 1
- Indexes captured: 1

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | NOT NULL |
| `event_type` | `TEXT` |  |
| `market_id` | `TEXT` | NOT NULL |
| `effective_start_ts` | `TEXT` | NOT NULL |
| `effective_end_ts` | `TEXT` |  |
| `mapping_version` | `TEXT` | NOT NULL |
| `source` | `TEXT` |  |
| `inserted_at` | `TEXT` | DEFAULT CURRENT_TIMESTAMP |

Constraints:
- `PRIMARY KEY (event_id, market_id, effective_start_ts, mapping_version)`

Indexes:
- `idx_map_window` on (`effective_start_ts`, `effective_end_ts`)

### `event_outcomes`

Purpose: Legacy unified event outcome table (compatibility).

- Columns: 6
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | NOT NULL |
| `realized_value` | `REAL` |  |
| `release_ts` | `TEXT` |  |
| `asof_ts` | `TEXT` | NOT NULL |
| `source` | `TEXT` |  |
| `learned_at_ts` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (event_id, asof_ts)`

### `event_outcomes_first_print`

Purpose: First-print event outcomes (point-in-time).

- Columns: 6
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | NOT NULL |
| `realized_value` | `REAL` |  |
| `release_ts` | `TEXT` |  |
| `asof_ts` | `TEXT` | NOT NULL |
| `source` | `TEXT` |  |
| `learned_at_ts` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (event_id, asof_ts)`

### `event_outcomes_revised`

Purpose: Revised event outcomes (point-in-time).

- Columns: 6
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | NOT NULL |
| `realized_value` | `REAL` |  |
| `release_ts` | `TEXT` |  |
| `asof_ts` | `TEXT` | NOT NULL |
| `source` | `TEXT` |  |
| `learned_at_ts` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (event_id, asof_ts)`

### `kalshi_contract_specs`

Purpose: Historical contract spec snapshots for reproducibility.

- Columns: 10
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `contract_id` | `TEXT` | NOT NULL |
| `spec_version_ts` | `TEXT` | NOT NULL |
| `market_id` | `TEXT` |  |
| `bin_low` | `REAL` |  |
| `bin_high` | `REAL` |  |
| `threshold_value` | `REAL` |  |
| `payout_structure` | `TEXT` |  |
| `direction` | `TEXT` |  |
| `status` | `TEXT` |  |
| `raw_contract_json` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (contract_id, spec_version_ts)`

### `kalshi_contracts`

Purpose: Latest contract catalog state and bin/threshold metadata.

- Columns: 13
- Constraints captured: 0
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `contract_id` | `TEXT` | PRIMARY KEY |
| `market_id` | `TEXT` | NOT NULL |
| `bin_low` | `REAL` |  |
| `bin_high` | `REAL` |  |
| `threshold_value` | `REAL` |  |
| `payout_structure` | `TEXT` |  |
| `direction` | `TEXT` |  |
| `tick_size` | `REAL` |  |
| `fee_bps` | `REAL` |  |
| `status` | `TEXT` |  |
| `spec_version_ts` | `TEXT` |  |
| `raw_contract_json` | `TEXT` |  |
| `inserted_at` | `TEXT` | DEFAULT CURRENT_TIMESTAMP |

### `kalshi_coverage_diagnostics`

Purpose: Coverage/quality diagnostics for distribution materialization.

- Columns: 8
- Constraints captured: 2
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | NOT NULL |
| `asof_date` | `TEXT` | NOT NULL |
| `expected_contracts` | `INTEGER` |  |
| `observed_contracts` | `INTEGER` |  |
| `missing_fraction` | `REAL` |  |
| `average_spread` | `REAL` |  |
| `median_quote_age_seconds` | `REAL` |  |
| `quality_score` | `REAL` |  |

Constraints:
- `constraint_violations REAL`
- `PRIMARY KEY (market_id, asof_date)`

### `kalshi_daily_health_report`

Purpose: Daily aggregate health report across coverage + ingestion metrics.

- Columns: 10
- Constraints captured: 0
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `asof_date` | `TEXT` | PRIMARY KEY |
| `markets_synced` | `INTEGER` |  |
| `contracts_synced` | `INTEGER` |  |
| `quotes_synced` | `INTEGER` |  |
| `avg_missing_fraction` | `REAL` |  |
| `avg_quality_score` | `REAL` |  |
| `total_constraint_violations` | `REAL` |  |
| `p95_quote_age_seconds` | `REAL` |  |
| `ingestion_errors` | `INTEGER` |  |
| `updated_at` | `TEXT` |  |

### `kalshi_data_provenance`

Purpose: Source/provenance records for ingestions by date and endpoint.

- Columns: 7
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | NOT NULL |
| `asof_date` | `TEXT` | NOT NULL |
| `source_env` | `TEXT` |  |
| `endpoint` | `TEXT` | NOT NULL |
| `ingest_ts` | `TEXT` |  |
| `records_pulled` | `INTEGER` |  |
| `notes` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (market_id, asof_date, endpoint)`

### `kalshi_distributions`

Purpose: Computed market-level probability distribution snapshots + quality metadata.

- Columns: 34
- Constraints captured: 2
- Indexes captured: 1

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | NOT NULL |
| `ts` | `TEXT` | NOT NULL |
| `spec_version_ts` | `TEXT` |  |
| `mean` | `REAL` |  |
| `var` | `REAL` |  |
| `skew` | `REAL` |  |
| `entropy` | `REAL` |  |
| `quality_score` | `REAL` |  |
| `coverage_ratio` | `REAL` |  |
| `median_spread` | `REAL` |  |
| `median_quote_age_seconds` | `REAL` |  |
| `volume_oi_proxy` | `REAL` |  |
| `tail_p_1` | `REAL` |  |
| `tail_p_2` | `REAL` |  |
| `tail_p_3` | `REAL` |  |
| `tail_threshold_1` | `REAL` |  |
| `tail_threshold_2` | `REAL` |  |
| `tail_threshold_3` | `REAL` |  |
| `tail_left_missing` | `INTEGER` |  |
| `tail_right_missing` | `INTEGER` |  |
| `mass_missing_estimate` | `REAL` |  |
| `moment_truncated` | `INTEGER` |  |
| `monotonic_violations_pre` | `INTEGER` |  |
| `monotonic_violation_magnitude` | `REAL` |  |
| `renorm_delta` | `REAL` |  |
| `isotonic_l1` | `REAL` |  |
| `isotonic_l2` | `REAL` |  |
| `distance_kl_1h` | `REAL` |  |
| `distance_js_1h` | `REAL` |  |
| `distance_wasserstein_1h` | `REAL` |  |
| `distance_kl_1d` | `REAL` |  |
| `distance_js_1d` | `REAL` |  |
| `distance_wasserstein_1d` | `REAL` |  |
| `quality_low` | `INTEGER` |  |

Constraints:
- `constraint_violation_score REAL`
- `PRIMARY KEY (market_id, ts)`

Indexes:
- `idx_distributions_market_ts` on (`market_id`, `ts`)

### `kalshi_fees`

Purpose: Effective fee history by market/contract/time.

- Columns: 4
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` |  |
| `contract_id` | `TEXT` |  |
| `effective_ts` | `TEXT` |  |
| `fee_bps` | `REAL` |  |

Constraints:
- `PRIMARY KEY (market_id, contract_id, effective_ts)`

### `kalshi_ingestion_checkpoints`

Purpose: Idempotent ingestion checkpoints by market/date/endpoint.

- Columns: 5
- Constraints captured: 2
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | NOT NULL |
| `asof_date` | `TEXT` | NOT NULL |
| `endpoint` | `TEXT` | NOT NULL |
| `last_ingest_ts` | `TEXT` | NOT NULL |
| `records_ingested` | `INTEGER` |  |

Constraints:
- `checksum TEXT`
- `PRIMARY KEY (market_id, asof_date, endpoint)`

### `kalshi_ingestion_logs`

Purpose: Endpoint-level ingestion logs and quality counters.

- Columns: 11
- Constraints captured: 1
- Indexes captured: 1

| Column | Type | Extra |
|---|---|---|
| `endpoint` | `TEXT` | NOT NULL |
| `asof_date` | `TEXT` | NOT NULL |
| `source_env` | `TEXT` |  |
| `ingest_ts` | `TEXT` | NOT NULL |
| `records_pulled` | `INTEGER` |  |
| `missing_markets` | `INTEGER` |  |
| `missing_contracts` | `INTEGER` |  |
| `missing_bins` | `INTEGER` |  |
| `p95_quote_age_seconds` | `REAL` |  |
| `error_count` | `INTEGER` |  |
| `notes` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (endpoint, ingest_ts)`

Indexes:
- `idx_ingestion_logs_date` on (`asof_date`)

### `kalshi_market_specs`

Purpose: Historical market spec/rules snapshots for reproducibility.

- Columns: 6
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | NOT NULL |
| `spec_version_ts` | `TEXT` | NOT NULL |
| `rules_text` | `TEXT` |  |
| `rules_hash` | `TEXT` |  |
| `raw_market_json` | `TEXT` |  |
| `source` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (market_id, spec_version_ts)`

### `kalshi_markets`

Purpose: Latest market catalog state (current spec snapshot per market).

- Columns: 13
- Constraints captured: 0
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `market_id` | `TEXT` | PRIMARY KEY |
| `event_id` | `TEXT` |  |
| `event_type` | `TEXT` |  |
| `title` | `TEXT` |  |
| `rules_text` | `TEXT` |  |
| `rules_hash` | `TEXT` |  |
| `open_ts` | `TEXT` |  |
| `close_ts` | `TEXT` |  |
| `settle_ts` | `TEXT` |  |
| `status` | `TEXT` |  |
| `spec_version_ts` | `TEXT` |  |
| `raw_market_json` | `TEXT` |  |
| `inserted_at` | `TEXT` | DEFAULT CURRENT_TIMESTAMP |

### `kalshi_quotes`

Purpose: Quote time series by contract and timestamp.

- Columns: 9
- Constraints captured: 1
- Indexes captured: 2

| Column | Type | Extra |
|---|---|---|
| `contract_id` | `TEXT` | NOT NULL |
| `ts` | `TEXT` | NOT NULL |
| `bid` | `REAL` |  |
| `ask` | `REAL` |  |
| `mid` | `REAL` |  |
| `last` | `REAL` |  |
| `volume` | `REAL` |  |
| `oi` | `REAL` |  |
| `market_status` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (contract_id, ts)`

Indexes:
- `idx_quotes_contract_ts` on (`contract_id`, `ts`)
- `idx_quotes_ts` on (`ts`)

### `macro_events`

Purpose: Authoritative latest macro event calendar rows.

- Columns: 8
- Constraints captured: 0
- Indexes captured: 1

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | PRIMARY KEY |
| `event_type` | `TEXT` | NOT NULL |
| `release_ts` | `TEXT` | NOT NULL |
| `timezone` | `TEXT` |  |
| `source` | `TEXT` |  |
| `revision_rules` | `TEXT` |  |
| `known_at_ts` | `TEXT` |  |
| `version_ts` | `TEXT` |  |

Indexes:
- `idx_events_release` on (`release_ts`)

### `macro_events_versioned`

Purpose: Versioned macro event snapshots for point-in-time event schedule auditing.

- Columns: 9
- Constraints captured: 1
- Indexes captured: 0

| Column | Type | Extra |
|---|---|---|
| `event_id` | `TEXT` | NOT NULL |
| `version_ts` | `TEXT` | NOT NULL |
| `event_type` | `TEXT` |  |
| `release_ts` | `TEXT` |  |
| `timezone` | `TEXT` |  |
| `source` | `TEXT` |  |
| `known_at_ts` | `TEXT` |  |
| `revision_policy` | `TEXT` |  |
| `raw_event_json` | `TEXT` |  |

Constraints:
- `PRIMARY KEY (event_id, version_ts)`
