# Quant Engine Glossary

## PERMNO

CRSP permanent security identifier used as the canonical identity in core equity workflows.

## PIT (Point-in-Time)

A data handling principle meaning the system should only use information known at the time, not revised/future data.

## As-of Join

A backward time join that attaches the most recent record at or before a target timestamp; used to prevent leakage.

## Leakage / Look-Ahead Bias

Using future information in features, labels, or validation, producing unrealistically strong results.

## Regime

A market state classification (e.g., bull trend, bear trend, mean-reverting, high volatility) used as context in modeling and risk.

## HMM

Hidden Markov Model; here, a Gaussian HMM used to infer latent market states probabilistically.

## Regime Confidence

A confidence/posterior signal attached to regime assignments, used for gating/blending/sizing decisions.

## DSR (Deflated Sharpe Ratio)

A multiple-testing-aware Sharpe significance adjustment used in advanced validation and promotion checks.

## PBO (Probability of Backtest Overfitting)

A validation metric estimating the probability that a strategy is overfit to historical data.

## Walk-Forward

A sequential train/test evaluation design that preserves time ordering and reduces overfitting risk.

## Purge / Embargo

Validation safeguards that remove temporally adjacent samples to reduce leakage through overlap or dependence.

## OptionMetrics

Options data source/features used for volatility surface and options factor enrichment in the feature pipeline.

## Kalshi Distribution Snapshot

A reconstructed market-level probability distribution from Kalshi contract quotes at a given as-of time.

## Threshold Direction (`ge`/`le`)

Contract semantics indicating whether a threshold contract represents P(X >= t) or P(X <= t).

## Isotonic Repair

Monotonic projection used to repair inconsistent threshold probability curves before extracting distribution features.

## Coverage Ratio

Fraction of expected contracts observed in a snapshot; part of Kalshi quality scoring.

## Autopilot

Strategy lifecycle layer that discovers, validates, promotes, and paper-trades execution variants around a predictor.

## Promotion Gate

Hard quality/risk/robustness rules that determine whether a strategy candidate is deployable.

## Paper Trading

Stateful simulated live trading used to monitor promoted strategies without real capital.

## Champion Model

The currently preferred production model version resolved by versioning/governance logic.

## Dash UI

The active web interface (`dash_ui`) used for system observability and workflow tooling.
