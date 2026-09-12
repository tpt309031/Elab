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
- [x] Regression tests and complete 2024-present refit.
- [x] English, responsive activity UI, filtered metrics and freshness handling.
- [x] Production build, lint, mobile overflow check and measured result report.
- Deployment status and the verified production commit are recorded in the GitHub
  release after publication, rather than inferred from a successful local build.

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

The existing grader gives no partial credit below the specified 0.1% directional
minimum. Exact accuracy, directional sign accuracy and mean weighted grade
(Correct=1, Partial=0.5, Wrong=0) are different metrics and must be labeled separately.
Hybrid already contains legacy OHLCV/intraday volume; Hybrid + Volume measures
the incremental effect of validated fixed-venue turnover and taker flow.

Three feature sets (Index + Astro, Hybrid, Hybrid + Volume) each compare Logistic,
Histogram GB and base rate. Monthly selection minimizes validation Brier loss;
the outer test is never used to pick that month's model. Report precision with
Wilson lower bound, recall, false alarms, misses, average precision, Brier skill,
sample size and dates. Candidate scores are shown alongside selected-model scores.
Overlapping 3D/5D outcomes are dependent; their sample count is not an independent
sample size and model comparisons are exploratory, not significance claims.

## Measured release results

Evaluation covers 984 UTC sessions, 2024-01-01 through 2026-09-10. Both lanes
have 852 calls and 132 abstentions under the chronological 8/4 monthly caps.
These are simulated walk-forward results, not the official live ledger.

| Ensemble | Exact grade | Directional accuracy | Net expectancy / call | Profit factor | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Index + Astro (Calendar) | 17.72% | 48.12% | -0.0584% | 0.911 | -58.13% |
| Full Hybrid | 18.43% | 48.94% | -0.1192% | 0.830 | -81.68% |

Neither ensemble meets the 70% target or the positive-expectancy gate. The best
individual directional accuracy is 51.93%, not the ensemble result. Prior metrics
used a different, noncausal quota protocol and are not a valid improvement baseline.
After-cost expectancy lower bounds remain negative; execution stays FLAT.

The new event audit has 8,856 predictions across three feature sets and three
horizons. Complete labels number 984 / 982 / 980 per feature set for 1D / 3D / 5D.
Hybrid + Volume selected-model Brier skill versus the pre-test base rate is
+1.82% / -1.20% / -0.28%, respectively. At the preregistered alert threshold,
selected models have no correct large-move alerts in this test period. Thus the
new volume features show a small 1D probability improvement, not a validated
alerting advantage. Keep these models experimental; do not lower the threshold
after seeing test outcomes and report that as independent validation.

There are 9 newly published immutable magnitude forecasts for 2026-09-12,
separate from backtests. Repeating the daily pipeline on the same closed candle
does not duplicate them. Live grades begin only after each complete horizon and
the 03:00 UTC evaluation cutoff. The validated daily ledger has 128 records,
124 evaluated and 4 pending, with no overdue grades at release preparation.

Validation: 49 regression tests passed; 6 optional deep-learning tests skipped
locally. Production build, TypeScript and lint are checked separately. Existing
weekly deep-model research is retained and was not retrained in this release.

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
