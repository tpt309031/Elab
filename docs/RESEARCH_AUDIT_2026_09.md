# Research integrity and market activity upgrade

## Scope and findings

- Weekly GitHub jobs on Aug 31 and Sep 7 failed publication: Full Hybrid had
  5 NO CALL days against cap 4. Recomputed next-session proposals overwrote
  immutable quota reservations. Locked dates must win the capacity merge.
  Evidence: GitHub Actions run 34107658779, job 101696289633.

- Monthly quota sorted the entire month by confidence. This allowed future test
  signals to determine earlier decisions. Use chronological capacity consumption
  (SIDEWAY <=8, NO CALL <=4) and test prefix invariance. This is a user-imposed
  decision constraint, not an accuracy improvement; forced direction is disclosed.
- Live estimators were refitted on calibration data but retained the original
  calibrator. Keep the estimator/calibrator pair unchanged after validation.
- Overlapping policy histories counted dates twice. Preserve the first OOS record.
- Sharpe omitted abstentions and drawdown omitted initial losses. Use the full
  chronological equity path with initial capital included in the running peak.
- An execution champion's next-day signal was reused for all future dates. Only
  its actual next session can pass the gate; long-range outlooks remain FLAT.
- Technical features were row-shifted on a frame ending at the last market bar,
  dropping the live forecast tail. Shift timestamps and outer-join labels so
  the next publishable session retains the latest real technical inputs.
- Health previously checked candle freshness but missed overdue OOS retraining
  and ungraded forecasts. Add these checks and surface failures on the dashboard.
- Legacy Event Lab matches pool pivot/wick/magnitude events with a +/-3-day
  retrospective window. They measure temporal association, not alert precision.
  Preserve these records but label them explicitly. New magnitude evaluation is
  forward-only, has fixed horizons and counts false alarms and missed events.

## Implementation checklist

- [x] Causal quota, unique policy observations, estimator/calibrator parity.
- [x] Correct equity metrics and date-specific execution gate.
- [x] Separate fixed-venue Binance BTC/USDT flow source with validation and units.
- [x] New magnitude experiment: 1D >4%, 3D >8%, 5D >12% absolute return.
- [x] Monthly purged fits, calibration, held-out model selection, base-rate control.
- [x] Preserve and score immutable event probabilities and all candidate models.
- [x] Local JSON checkpoints keyed by code, protocol and training inputs; bounded
  CPU workers to avoid oversubscription. A failed later stage can reuse a completed
  lane; partial lanes are recalculated. Cache files are not committed.
- [x] Pre-publication verification and atomic per-file artifact replacement.
- [ ] Regression tests and complete 2024-present refit.
- [ ] English, responsive activity UI, filtered metrics and freshness handling.
- [ ] Production build, deployment verification and measured result report.

## Protocol

Volume is venue-specific, never aggregate market volume: BTC base volume, USDT
quote turnover, trades and taker-buy flow. Features include relative volume,
log-volume z-score, trend, taker imbalance and average trade notional. Missing
days remain missing. Market/flow features are available two sessions before the
target; precomputed Index/Astro features retain their availability assumptions.

The magnitude task uses open on target day to close on target+horizon-1. Complete
UTC candles are required. High-low range remains a separate diagnostic. Daily
direction grades retain the user's exact thresholds and never gain credit from
event lag. Event forecasts are graded after the horizon closes and 03:00 UTC;
the existing 03:20 UTC job performs grading, ranking and publication.

Three feature sets (Index + Astro, Hybrid, Hybrid + Volume) each compare Logistic,
Histogram GB and base rate. Monthly selection minimizes validation Brier loss;
the outer test is never used to pick that month's model. Report precision with
Wilson lower bound, recall, false alarms, misses, average precision, Brier skill,
sample size and dates. Candidate scores are shown alongside selected-model scores.
Overlapping 3D/5D outcomes are dependent; their sample count is not an independent
sample size and model comparisons are exploratory, not significance claims.

## Limitations and next gates

Historical private indices/Astro without timestamped source versions cannot prove
point-in-time availability. OOS research is conditional on prepublication, not a
verified prospective track record. Keep official forecasts separate and immutable.
Monthly quotas may reduce accuracy and cannot justify trading. Candidate selection
over many models introduces selection bias; monitor prospective results and demand
positive after-cost expectancy lower bound before execution. Do not promise 70%.
Deep challengers remain on the existing weekly process; adding model complexity is
not evidence of improvement. Promote only after independent evaluation.

Sources: [Binance public data fields](https://github.com/binance/binance-public-data),
[scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
