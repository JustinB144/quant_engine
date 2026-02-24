# Quant Engine Glossary

## PERMNO

CRSP permanent security identifier used as the canonical identity for core equity workflows and point-in-time joins.

## PIT (Point-in-Time)

Data-handling principle: only use information known at the decision timestamp, not revised or future information.

## As-of Join

Backward-looking time join that attaches the latest record at or before the target timestamp; used to prevent leakage.

## Leakage / Look-Ahead Bias

Use of future information in features, labels, or validation that inflates backtest/model performance.

## Regime

Market state classification used across modeling and risk (configured in `config.py` via `REGIME_NAMES`).

## HMM

Hidden Markov Model; in this repo, a Gaussian HMM implementation used for regime inference (`regime/hmm.py`).

## Regime Confidence / Probabilities

Confidence and posterior probabilities attached to regime assignments; consumed by prediction, suppression, and UI diagnostics.

## Walk-Forward

Sequential train/test evaluation preserving time order; used in validation and autopilot robustness checks.

## Purge / Embargo

Validation safeguards that remove temporally adjacent samples/events to reduce leakage through overlap/dependence.

## DSR (Deflated Sharpe Ratio)

Multiple-testing-aware Sharpe significance metric used in advanced validation and promotion gating.

## PBO (Probability of Backtest Overfitting)

Metric estimating how likely a strategy’s apparent performance is due to overfitting.

## Promotion Gate

Hard quality/risk/robustness checks (`autopilot/promotion_gate.py`) required before a strategy is promoted/paper-traded.

## Autopilot

Strategy lifecycle subsystem that discovers variants, validates them, applies promotion rules, and runs paper trading.

## Paper Trading

Stateful simulated execution of promoted strategies (persisted in `results/autopilot/paper_state.json`).

## Champion Model

Preferred production model version resolved by model governance/versioning logic.

## ApiResponse Envelope

Standard API response wrapper (`api/schemas/envelope.py`) containing `ok`, `data`, `error`, and `meta` provenance fields.

## Job Store / Job Runner

API background job persistence and execution system (`api/jobs/store.py`, `api/jobs/runner.py`) with SSE progress streaming.

## Frontend UI (React/Vite)

Active web interface implemented in `frontend/`, backed by FastAPI endpoints under `/api`.

## Kalshi Distribution Snapshot

Reconstructed market-level probability distribution from Kalshi contract quotes at a given as-of timestamp.

## Threshold Direction (`ge` / `le`)

Kalshi contract threshold semantics describing whether a contract represents `P(X >= t)` or `P(X <= t)`.

## Isotonic Repair

Monotonic projection used to repair inconsistent threshold probability curves before feature extraction in Kalshi distribution building.
