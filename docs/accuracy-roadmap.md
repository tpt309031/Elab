# Accuracy Improvement Roadmap

This document records the current out-of-sample baseline and the research plan for improving the Calendar and Full Hybrid forecast lanes. All promotion decisions must be based on immutable, purged walk-forward predictions. In-sample accuracy is diagnostic only.

## Current Baseline

The July 13, 2026 rebuild contains 924 out-of-sample observations per lane.

| Lane | Exact grade | Weighted grade | Directional accuracy | Expectancy |
| --- | ---: | ---: | ---: | ---: |
| Calendar | 22.31% | 35.64% | 50.38% | -0.0778% per call |
| Full Hybrid | 24.33% | 37.77% | 53.01% | -0.0137% per call |

The 70% directional target is not reached. The system must show this honestly and must not optimize the displayed call mix to make the result look stronger.

## Main Weaknesses

1. **Target and score mismatch.** Training currently separates UP, SIDEWAY, and DOWN near the +/-1% boundary, while the official score only awards a fully correct UP or DOWN at +/-3%. The model is therefore not trained for the reward it receives.
2. **Forced quota distortion.** The historical rate inside +/-1% is about 41%, or roughly 12 days in a 30-day month. A hard maximum of eight SIDEWAY calls forces some statistically valid SIDEWAY forecasts into UP or DOWN.
3. **Concept drift.** Results weakened in 2026 relative to 2024-2025. Fixed historical weights react too slowly to a changing market regime.
4. **Small daily sample.** Approximately 3,000 daily rows are insufficient for a large LSTM or Transformer without severe overfitting risk.
5. **Potential timestamp leakage.** Private index values need an `available_at` timestamp. A value must only be used if it was actually available before the forecast cutoff.
6. **Pattern selection bias.** Searching many patterns and ranking them on the same observations inflates apparent confidence. Pattern evidence needs purged out-of-sample scoring, shrinkage, and a minimum sample.
7. **Unstable live ranking.** A model should not be promoted from one or two live outcomes. Daily evidence can update its score, but promotion needs a minimum live sample and should normally happen monthly.
8. **Source inconsistency.** Exchange fallback keeps the pipeline alive, but different exchange closes can slightly change labels near thresholds. Provider and price discrepancy must be stored with every evaluation.

## Recommended Forecast Target

Predict a calibrated return distribution using seven mutually exclusive bins that align with the official scoring boundaries:

1. `return <= -3%`
2. `-3% < return < -1%`
3. `-1% <= return <= -0.1%`
4. `-0.1% < return < 0.1%`
5. `0.1% <= return <= 1%`
6. `1% < return < 3%`
7. `return >= 3%`

Convert the seven probabilities into expected grading utility:

- `UP = P(bin 7) + 0.5 * P(bins 5 or 6)`
- `DOWN = P(bin 1) + 0.5 * P(bins 2 or 3)`
- `SIDEWAY = P(bins 3, 4, or 5)`

The daily forecast is the action with the highest calibrated expected utility. Keep a probability forecast for every day. A separate trade decision may remain flat when expected net return after fees and slippage is not positive.

## Evaluation Protocol

1. Use expanding and rolling walk-forward tests, never shuffled splits.
2. Purge overlapping targets and apply an embargo at least as long as the forecast horizon.
3. Fit preprocessing, feature selection, model, calibration, and ensemble weights only inside each training fold.
4. Preserve every issued forecast in the immutable ledger before the outcome exists.
5. Grade daily calls only after the UTC candle closes, with exactly three outcomes: Correct, Partial, or Wrong.
6. Evaluate delayed pivot and large-move patterns in a separate event-horizon report. Do not use delayed matching to change the daily grade.
7. Report exact score, weighted score, balanced accuracy, MCC, class precision/recall, Brier score, log loss, ECE, expectancy, profit factor, Sharpe, drawdown, turnover, and coverage.
8. Break metrics down by year, volatility regime, trend regime, weekday, confidence bucket, and market-data provider.
9. Add block-bootstrap confidence intervals. Promote a model only when its lower confidence bound and expectancy beat the champion after costs.

## Model Slate

### Production candidates

- Multinomial and ordinal Logistic Regression as transparent baselines.
- CatBoost, LightGBM, and XGBoost for the seven-bin distribution.
- Quantile LightGBM or CatBoost for return quantiles and expected magnitude.
- A two-stage hurdle model: first magnitude/volatility, then conditional direction.
- Hidden Markov or change-point regime models for dynamic model weights.
- A constrained out-of-fold stacking model for the final probability distribution.

### Research challengers

- Temporal Convolutional Network on 1h and 4h sequences.
- PatchTST, Temporal Fusion Transformer, or iTransformer only after adding enough intraday history.
- Gradient-boosted discrete hazard model for pivot-high and pivot-low risk.
- Bayesian model averaging or online Hedge weighting with decay for controlled adaptation.

Deep sequence models remain challengers until their purged out-of-sample confidence interval and net expectancy beat the tabular champion.

## Feature Priorities

### P0: correctness and provenance

- `issued_at`, `available_at`, forecast cutoff, provider, and candle-close timestamps.
- Cross-exchange close discrepancy and stale-source health flags.
- Missingness indicators and feature lineage for every private index value.

### P1: market structure

- 1h/4h realized volatility, intraday trend, range, wick, volume profile, and weekend effects.
- Funding rate, open interest, basis, liquidations, and perpetual volume.
- Options implied volatility and skew where history is available.
- ETF flow, DXY, rates, and selected on-chain variables with strict publication lags.

### P2: index and Astro interactions

- Multi-horizon slope, acceleration, shock, crossing age, phase duration, and divergence duration.
- Regime-conditioned rolling correlation and lagged cross-correlation.
- Astro event strength, distance-to-event, interaction with volatility regime, and interaction with private-index extrema.
- Similarity features calculated only against the training window.

## Automatic Ranking

1. Update model and pattern evidence after each closed UTC day.
2. Apply Bayesian shrinkage to small samples and rank by a conservative lower confidence bound, calibration, and net expectancy.
3. Require at least 20-30 live graded forecasts before live evidence can independently promote a model.
4. Re-rank daily, but promote or replace production models on a monthly schedule unless a safety rule triggers immediate demotion.
5. Keep retired forecasts immutable and show why each model was promoted, demoted, or placed on standby.
6. Use drift alarms such as PSI plus Page-Hinkley or ADWIN to shorten the training window or increase recent-regime weight.

## Product Additions

- Data health panel: latest closed candle, expected candle, provider, last successful evaluation, retry state, and stale alert.
- Seven-bin probability chart plus expected move and expected grading utility.
- Champion/challenger table with confidence intervals, calibration, expectancy, sample size, and promotion reason.
- Confusion matrix and precision/recall by class, year, and regime.
- Reliability diagram, confidence-risk curve, and prediction-distribution history.
- Drift dashboard for features, class frequency, calibration, and live performance.
- Historical analog cards with the source window and subsequent return chart.
- Separate Pivot/Event Lab for +/-3-day event research, distinct from daily forecast grading.
- Pipeline failure notification and a visible audit trail for every evaluation and model-state change.

## Acceptance Gates

A release is acceptable only when:

- the artifact contains the most recent fully closed UTC candle;
- all elapsed official forecasts are graded exactly once;
- source freshness and forecast immutability tests pass;
- no feature uses information unavailable at the forecast cutoff;
- production metrics are computed from out-of-sample predictions only;
- the promoted model has positive net expectancy and a documented confidence interval;
- production remains blocked rather than publishing stale market data.
