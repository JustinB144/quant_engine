# Quant Engine: Complete Improvement Roadmap

**Date:** February 21, 2026
**Basis:** Three full system audits + codebase analysis of 119 files (~21,000 lines)

---

## TIER 1 — CRITICAL (Do First)

These are blocking issues. Nothing else matters until these are resolved.

### 1.1 Run a Fresh Training + Backtest Cycle

All current metrics (Sharpe -0.38, 0/36 promoted, DSR p-value 1.0) are from before any guardrails existed. The system now has CV gap hard blocks, holdout R² rejection, regime 2 gating, excess return targets, HAR features, full covariance HMM, and Viterbi decoding — none of which are reflected in the stored results.

The CV gap hard block alone would reject every current model (global gap 0.54 vs threshold 0.15). You need to retrain from scratch to see what the system actually produces with all the new infrastructure in place.

**What to watch for after retraining:**
- Do any models survive the CV gap and holdout R² gates?
- If not, you need to relax thresholds slightly or improve the feature set before models can pass
- Does regime 2 gating actually improve aggregate Sharpe, or does it just shift losses to other regimes?
- Do the excess return targets reduce CV gap (they should, since you've removed market-direction noise)?

### 1.2 Validate Regime 2 Gating Isn't Masking a Deeper Problem

The backtester suppresses entries when regime == 2 and confidence > 0.5. This should flip Sharpe from -0.38 to roughly +0.10 based on the old numbers, but it's a bandaid. The real question is why the model is systematically wrong in mean-reverting regimes. After retraining with the new infrastructure, check whether the new model actually has edge in regimes 1 and 3, or whether the old positive numbers were also noise.

---

## TIER 2 — WRDS ALTERNATIVE DATA INTEGRATION

You already have WRDS connected with working SQL, sanitized queries, and deterministic secid linking. Some of these datasets are already partially in `wrds_provider.py` but not wired into `alternative.py`. Others need new query methods.

### 2.1 Wire IBES Earnings Surprise into Alternative Data Pipeline

**Current state:** `wrds_provider.py` already has a method querying `ibes.statsum_epsus` (lines 883-947) that pulls actual, meanest, medest, stdev, numest, surprise_pct, beat, and dispersion. But `alternative.py`'s `get_earnings_surprise()` returns None — it's not connected to the WRDS method.

**What needs to happen:**
- `alternative.py` should call the WRDS provider's IBES method instead of returning None
- Map the IBES output columns to the expected schema: `report_date`, `eps_estimate`, `eps_actual`, `surprise_pct`
- Ensure point-in-time semantics: use `rdq` (report date of quarterly earnings) so backtests only see surprises after announcement
- Add derived features: `surprise_zscore` (surprise_pct / dispersion), `beat_streak` (consecutive beats), `revision_momentum` (change in meanest over trailing 90 days)

**Expected alpha contribution:** Earnings surprise is one of the most well-documented alpha factors. Post-earnings announcement drift (PEAD) persists 60-90 days. The 10-day prediction horizon aligns well with capturing the early portion of this drift.

### 2.2 Wire OptionMetrics into Alternative Data Pipeline

**Current state:** `wrds_provider.py` already queries OptionMetrics (lines 576-794) for IV surface data: iv_atm_30, iv_atm_60, iv_atm_90, iv_put_25d, iv_call_25d, term_slope, skew, vrp. But `alternative.py`'s `get_options_flow()` returns None.

**What needs to happen:**
- `alternative.py` should call the WRDS provider's OptionMetrics method
- Add volume-based features not currently extracted: `put_call_ratio` (put volume / call volume), `total_open_interest`, `unusual_volume_flag` (volume > 2× 20-day average)
- These require querying `optionm.opprcd` for volume and open interest fields in addition to the IV fields already pulled
- The IV surface features already flow through `options_factors.py` — the gap is the flow/sentiment features (volume, OI, put-call ratio)

**Expected alpha contribution:** Options flow captures informed trading. Unusual call buying before earnings, elevated put-call ratios, and OI buildups are empirically predictive.

### 2.3 Add Compustat Short Interest

**Current state:** Not implemented in `wrds_provider.py`. The alternative.py stub mentions ORTEX, S3 Partners, and FINRA but not Compustat.

**What needs to happen:**
- Add a new method to `wrds_provider.py` querying `comp.sec_shortint` or the equivalent Compustat short interest table
- Pull: `settlement_date`, `short_interest` (shares short), `days_to_cover` (short interest / avg daily volume), `short_pct_float` (short interest / float)
- Wire into `alternative.py`'s `get_short_interest()` method
- Derive features: `short_interest_ratio` (short / float), `short_interest_change` (month-over-month delta), `days_to_cover` (liquidity-adjusted short pressure), `squeeze_risk` (high short interest + declining borrow availability)

**Expected alpha contribution:** Short interest is a crowded but persistent factor. The value is less in the level and more in the change — rising short interest precedes underperformance, rapid covering precedes squeezes. Days-to-cover captures the liquidity dimension that raw share counts miss.

**Note:** WRDS may have short interest through Compustat or through a separate subscription (like the Nasdaq Short Interest file). Check your WRDS subscription entitlements for which table is available.

### 2.4 Add Thomson Reuters / SEC Insider Transactions

**Current state:** `wrds_provider.py` already queries `tfn.s34` for institutional ownership / 13F filings (lines 953-1001). But insider transactions (Form 4 filings — officer/director buys and sells) are in a different table.

**What needs to happen:**
- Add a method querying `tfn.insiderdata` or the equivalent Thomson Reuters insider filing table on WRDS
- If Thomson Reuters insider data isn't in your subscription, use SEC EDGAR Form 4 XML parsing as fallback (but this is significantly more work)
- Pull: `filing_date`, `insider_name`, `title`, `transaction_type` (purchase/sale/exercise), `shares`, `price`, `value`
- Wire into `alternative.py`'s `get_insider_transactions()` method
- Derive features: `net_insider_buying` (buy $ - sell $ over trailing 90 days), `insider_buy_count` (number of distinct insiders buying), `cluster_buy_signal` (3+ insiders buying within 30 days)

**Expected alpha contribution:** Insider buying is more informative than selling (executives sell for many non-signal reasons; they buy for one reason). Cluster buying by multiple insiders is the strongest signal. This factor is low-frequency (updates monthly at best) but has strong long-horizon predictive power that complements the 10-day horizon.

### 2.5 Add 13F Institutional Ownership Changes

**Current state:** `wrds_provider.py` already pulls institutional ownership from `tfn.s34` (lines 953-1001): total_shares_held, num_institutions, pct_institutional. But this is only used as a static snapshot — no change features are derived.

**What needs to happen:**
- Compute quarter-over-quarter changes in institutional ownership
- Derive features: `inst_ownership_change` (delta in pct_institutional), `num_institutions_change`, `new_positions` (institutions initiating positions), `closed_positions` (institutions exiting entirely)
- These don't require new WRDS queries — just temporal differencing of the existing data

**Expected alpha contribution:** Institutional herding (many institutions building positions simultaneously) precedes momentum. Institutional exodus (many closing) precedes reversals. The change is more informative than the level.

### 2.6 Integrate TAQmsec Intraday Features

**Current state:** `wrds_provider.py` has TAQmsec millisecond tick data access (lines 1013-1142) that aggregates to OHLCV bars at various frequencies (1m, 5m, 15m, 30m, 45m). But this data isn't flowing into the feature pipeline.

**What needs to happen:**
- Add intraday features to `features/pipeline.py` or `features/research_factors.py`:
  - `intraday_vol_ratio`: first-hour volatility / last-hour volatility (captures informed trading)
  - `vwap_deviation`: close price vs VWAP (captures institutional execution pressure)
  - `amihud_illiquidity`: |return| / dollar volume (captures liquidity risk)
  - `kyle_lambda`: price impact coefficient from intraday regressions (captures adverse selection)
  - `realized_vol_5m`: 5-minute realized volatility (more accurate than daily RV)
  - `microstructure_noise`: difference between 5m RV and 1m RV (captures market friction)
- These features are computationally expensive — use the feature store (PIT) to cache them
- TAQmsec coverage is 2022-01-04 to ~2024-12-31 per your subscription

**Expected alpha contribution:** Intraday microstructure features capture information that daily bars completely miss. Kyle's lambda and VWAP deviation are particularly valuable for predicting short-horizon returns.

---

## TIER 3 — FEATURE ENGINEERING IMPROVEMENTS

### 3.1 Implement Dynamic Time Warping (DTW) for Lead-Lag Detection

**Current state:** `research_factors.py` has network momentum using simple lagged correlation (lines 379-493). The docstring references DTW but it's not implemented.

**What needs to happen:**
- Implement DTW distance computation between pairs of asset return series
- Use DTW alignment paths to detect non-linear lead-lag relationships (where asset A's pattern appears in asset B with variable delay)
- Replace or augment the simple correlation-based weight matrix with DTW-derived weights
- Consider using `tslearn` or `dtaidistance` Python libraries for efficient DTW computation
- Add features: `dtw_leader_score`, `dtw_follower_score`, `dtw_avg_lag` (average warping delay)

**Why it matters:** Simple lagged correlation only captures linear, fixed-delay relationships. DTW captures cases where the delay varies (e.g., sector rotation that accelerates during crises). The uploaded papers on network momentum (2501.07135v1, 2308.11294v1) specifically advocate for DTW over correlation.

**Effort:** Medium. The infrastructure (network momentum framework, centrality features) already exists — this is swapping the distance metric.

### 3.2 Implement Path Signature Features

**Current state:** The uploaded paper 1603.03788v2 covers path signatures for financial time series. `research_factors.py` has a Lévy area computation but no higher-order signatures or multi-scale signature features.

**What needs to happen:**
- Compute truncated path signatures (order 2-3) of (price, volume) paths over rolling windows
- Use the `iisignature` or `signatory` Python library
- Multi-scale signatures: compute over 5d, 20d, 60d windows to capture different pattern scales
- The signature captures the "shape" of the price path in a mathematically rigorous way — it's a universal nonlinear feature extractor for sequential data

**Why it matters:** Path signatures are provably universal features for continuous paths. They capture patterns like "down-then-up" or "volatile-then-calm" that momentum and volatility features miss. They're also naturally invariant to time reparametrization.

**Effort:** Medium-High. Requires understanding the math, but libraries handle the computation.

### 3.3 Implement Full OFI (Order Flow Imbalance) Calibration

**Current state:** The OFI proxy in `research_factors.py` uses `sign(close_change) × volume` as a crude approximation. The uploaded Cont et al. paper (1011.6402v3) defines OFI using depth changes at best bid/ask.

**What needs to happen:**
- If TAQmsec quote data is accessible, compute true OFI: `ΔQ_bid - ΔQ_ask` (change in bid depth minus change in ask depth)
- If only OHLCV data is available, improve the proxy: use (close - open) / (high - low) × volume (captures intrabar dynamics better than sign alone)
- Calibrate the impact coefficient: regress 5-minute returns on OFI to estimate Kyle's lambda per stock
- Add depth-adjusted features: `ofi_normalized` (OFI / average depth), `ofi_persistence` (autocorrelation of OFI), `ofi_momentum` (cumulative OFI over 5 bars)

**Effort:** Depends on data access. With TAQmsec: Medium. Without: Low (just improve the proxy).

### 3.4 Add Sentiment / Macro Indicator Features

**Current state:** No macroeconomic indicator features besides what comes through the Kalshi module.

**What needs to happen:**
- FRED API integration for: VIX, term spread (10Y-2Y), credit spread (BAA-AAA), unemployment claims, ISM manufacturing, consumer confidence
- These are freely available and don't require WRDS
- Compute both levels and changes (momentum of macro indicators)
- Cross with regime: macro features interact differently depending on the HMM state

**Effort:** Low. FRED has a free API with excellent Python support (`fredapi` package).

---

## TIER 4 — MODEL & PORTFOLIO IMPROVEMENTS

### 4.1 Add XGBoost to the Ensemble

**Current state:** `ENSEMBLE_DIVERSIFY = True` trains GBR + ElasticNet + RandomForest. XGBoost is not included.

**What needs to happen:**
- Add XGBoost as a fourth model in the ensemble (if `xgboost` is installed)
- XGBoost typically outperforms sklearn's GBR due to regularization (L1/L2 on leaf weights), native handling of missing values, and histogram-based splitting
- Consider LightGBM as an alternative — it's faster for large feature sets

**Effort:** Low. The DiverseEnsemble infrastructure already handles arbitrary numbers of models.

### 4.2 Add a Neural Network Model (Optional/Experimental)

**Current state:** All models are tree-based or linear. No neural network.

**What needs to happen:**
- Add a small feedforward network (2-3 hidden layers, dropout, batch norm) as an additional ensemble member
- Or add a temporal model (LSTM/GRU) that takes the last 20 days of features as a sequence
- Use PyTorch with early stopping on validation loss
- This is experimental — neural networks are harder to train on tabular financial data and may not add value

**Effort:** High. Only pursue after tree-based models show signal.

### 4.3 Predictor-Level Regime 2 Suppression

**Current state:** Regime 2 gating happens in the backtester (engine.py:444-449) but the predictor still generates predictions for regime 2 normally.

**What needs to happen:**
- In `predictor.py`, when regime == 2, either: (a) zero out the prediction, (b) set confidence to 0, or (c) add a `regime_suppressed` flag to the output
- This is a cleanliness issue — the gating already works downstream — but it means live trading would also need to implement the gate separately
- Better to have the predictor itself signal "no trade" so the gate is centralized

**Effort:** Low. A few lines in predictor.py.

### 4.4 Implement Walk-Forward Model Selection

**Current state:** The system trains a single model configuration and evaluates it. No systematic comparison of hyperparameters across walk-forward folds.

**What needs to happen:**
- For each walk-forward fold, train N model configurations (varying max_depth, learning_rate, n_estimators, feature subsets)
- Select the configuration with best average OOS performance across folds
- Use the Deflated Sharpe Ratio to penalize for multiple testing
- The trial counting infrastructure in `kalshi/walkforward.py` already exists — adapt it for the equity model

**Effort:** Medium. The infrastructure exists; this is orchestration.

### 4.5 Integrate Cross-Sectional Ranking into the Live Pipeline

**Current state:** `models/cross_sectional.py` exists and generates cs_rank, cs_zscore, and long_short_signal. But the autopilot engine doesn't appear to use it — it runs time-series predictions and ranks by predicted return.

**What needs to happen:**
- After the predictor generates time-series forecasts, pass them through `CrossSectionalRanker` to get market-neutral signals
- Use cs_zscore (not raw predicted return) as the ranking criterion for portfolio construction
- This removes market beta from the signal — you're trading relative value, not directional exposure

**Effort:** Low-Medium. The ranker is built; it needs to be wired into the autopilot engine.

### 4.6 Integrate Portfolio Optimizer into Position Sizing

**Current state:** `risk/portfolio_optimizer.py` has mean-variance optimization with turnover penalty. But the autopilot/paper trader appears to use simpler rank-and-fill allocation.

**What needs to happen:**
- After ranking stocks, use the optimizer to determine weights
- Feed it: expected returns (from predictor), covariance matrix (from regime-conditional covariance), current weights (from existing positions), turnover penalty (from config)
- The optimizer handles concentration limits and volatility constraints

**Effort:** Medium. The optimizer is built; integration requires plumbing between autopilot and risk modules.

---

## TIER 5 — INFRASTRUCTURE & OBSERVABILITY

### 5.1 Complete the UI Dashboard

**Current state:** `ui/` has 11 files (app.py + 8 page files) with shell structure. The page structure exists but content is minimal.

**What needs to happen:**
- Fill in dashboard pages: portfolio overview, regime state, model performance, feature importance, trade log, risk metrics
- Add real-time regime indicator (current HMM state + probability distribution)
- Add model health monitoring (current CV gap, holdout R², IC drift, retrain trigger status)
- Visualization of the performance attribution decomposition (market vs factor vs alpha)

### 5.2 Add Integration Tests

**Current state:** 18 test files (~1,500+ lines) covering individual components. No end-to-end integration test that runs the full pipeline: data load → feature computation → regime detection → model prediction → portfolio construction → backtest.

**What needs to happen:**
- Write an integration test with synthetic data (100 stocks, 500 days) that exercises the full pipeline
- Verify that PIT semantics are maintained end-to-end
- Verify that the CV gap hard block actually prevents bad models from being saved
- Verify that regime 2 gating actually suppresses trades
- Verify that the cross-sectional ranker produces market-neutral signals

### 5.3 Add Logging and Monitoring Infrastructure

**Current state:** Logging is ad-hoc (print statements, self._log methods). No structured logging, no metrics collection, no alerting.

**What needs to happen:**
- Structured logging with Python's logging module (JSON format for machine parsing)
- Key metrics emitted on every cycle: model age, IC, rolling Sharpe, regime distribution, portfolio turnover, execution costs
- Alert thresholds: IC below 2%, Sharpe below 0, drawdown exceeding tier thresholds, regime 2 duration exceeding 30 days

### 5.4 Add Reproducibility Verification

**Current state:** `reproducibility.py` creates run manifests with git commit, config snapshot, and dataset checksums. But there's no verification that a given manifest can reproduce results.

**What needs to happen:**
- Add a `verify_manifest()` function that, given a manifest, checks that the current environment matches (same git commit, same config, same data checksums)
- Add a `replay()` function that re-runs a historical cycle and compares results to stored outputs
- This is essential for audit trails and regulatory compliance

---

## TIER 6 — RESEARCH PAPER IMPLEMENTATIONS

These are from the 10 uploaded papers. Each represents a significant research implementation effort.

### 6.1 HARX Volatility Spillovers (Paper: 2601.03146v4)

**Current state:** HAR features (daily/weekly/monthly RV) are implemented. But the HARX extension — cross-market volatility spillovers (how one market's volatility predicts another's) — is not.

**What needs to happen:**
- Implement HARX: HAR model augmented with exogenous regressors from other markets
- Compute vol spillover indices: how much of stock X's volatility is explained by the broad market, its sector, and its lead-lag network neighbors
- The network momentum infrastructure (lead-lag weights) can be reused for the spillover weights

### 6.2 Markov Limit Order Book (Paper: 1104.4596v1)

**Current state:** Not implemented. This requires tick-level order book data.

**What needs to happen:**
- Model the limit order book as a continuous-time Markov chain
- Extract features: queue position dynamics, fill probability, expected time-to-fill
- Duration features: time between quote updates as a signal for information arrival
- This requires TAQmsec data (which you have) but at a granularity level that may be computationally expensive

### 6.3 Time Series Momentum Enhancements (Paper: ssrn-2089463)

**Current state:** Basic momentum features exist. But the paper's key insight — volatility-scaled momentum (dividing momentum by trailing volatility to normalize across regimes) — may not be implemented.

**What needs to happen:**
- Verify whether momentum features are volatility-scaled
- Add reversal horizon features: identify the horizon at which momentum turns into mean-reversion for each stock
- Add cross-sectional momentum (relative momentum): stock's momentum minus sector/market momentum

### 6.4 Waves & Mean Flows (Book: 9781107669666)

**Current state:** Not implemented. This is a physics-inspired conceptual framework about flow dynamics.

**What needs to happen:**
- Apply the wave equation analogy: model price as a superposition of "waves" (momentum) and "mean flows" (drift)
- Decompose returns into oscillatory (wave) and secular (flow) components using spectral analysis
- Use wave-flow decomposition to improve regime detection: trending regimes have strong flow, mean-reverting regimes have strong waves

**Note:** This is the most conceptual of the papers and the hardest to operationalize. Lower priority.

---

## PRIORITY SUMMARY

| Priority | Item | Effort | Expected Impact |
|----------|------|--------|-----------------|
| **Do Now** | Fresh train/backtest cycle | Low | Validates everything |
| **Do Now** | Validate regime 2 gating | Low | Confirms the gate works |
| **Week 1** | Wire IBES earnings surprise (2.1) | Low | High — proven alpha factor, data already in WRDS |
| **Week 1** | Wire OptionMetrics flow features (2.2) | Low | Medium — volume/OI features complement IV |
| **Week 1** | Add 13F ownership changes (2.5) | Low | Medium — data already pulled, just needs differencing |
| **Week 2** | Add Compustat short interest (2.3) | Medium | Medium — new WRDS query needed |
| **Week 2** | Add insider transactions (2.4) | Medium | Medium — depends on WRDS subscription |
| **Week 2** | Integrate cross-sectional ranking (4.5) | Low | High — moves to market-neutral signals |
| **Week 2** | Integrate portfolio optimizer (4.6) | Medium | Medium — better allocation |
| **Week 3** | TAQmsec intraday features (2.6) | Medium | High — microstructure features are orthogonal |
| **Week 3** | DTW momentum (3.1) | Medium | Low-Medium — correlation already works |
| **Week 3** | Path signatures (3.2) | Medium-High | Medium — universal features |
| **Week 3** | Add XGBoost to ensemble (4.1) | Low | Low-Medium |
| **Month 2** | FRED macro indicators (3.4) | Low | Medium |
| **Month 2** | Full OFI calibration (3.3) | Medium | Medium |
| **Month 2** | Walk-forward model selection (4.4) | Medium | Medium |
| **Month 2** | Integration tests (5.2) | Medium | Risk reduction |
| **Month 2** | HARX vol spillovers (6.1) | High | Medium |
| **Month 2** | UI dashboard (5.1) | Medium | Observability |
| **Month 3** | Predictor regime 2 flag (4.3) | Low | Cleanliness |
| **Month 3** | Logging infrastructure (5.3) | Medium | Observability |
| **Month 3** | Reproducibility verification (5.4) | Medium | Compliance |
| **Month 3** | Vol-scaled momentum (6.3) | Medium | Medium |
| **Month 3** | Markov LOB features (6.2) | High | Uncertain — needs tick data |
| **Month 3+** | Neural network model (4.2) | High | Uncertain |
| **Month 3+** | Wave-flow decomposition (6.4) | High | Uncertain — conceptual |
