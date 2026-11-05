# ETF integration verification

## Completed implementation

- Strict publication-time macro ingestion, append-only vintages and atomic refresh.
- Two-session availability alignment and revision-safe rolling features.
- Flow specialists, common-candidate input integration and independent live gates.
- Paired walk-forward experiment command that never writes the learning ledger.
- Additive dashboard/API contract, dark flow chart and coefficient diagnostics.
- Regression coverage for late publications, revisions, weekends and unknown times.
- Public history adapter with validated coverage, content hashes, preserved
  changed snapshots, attribution and a display-only dashboard fallback.

## Data gate

706 real ETF observations from 2024-01-11 through 2026-11-03 were fetched from
TFTC's public, CC BY 4.0 archive. They support the daily/cumulative chart. Its
original publication times are unknown, so `available_at` remains null in that
archive and it cannot enter training. The strict `macro.csv` remains header-only.
Source comparison and licensing are in `data/external/README.md`.

The trusted publication-time API adapter is implemented but not configured.
No trading-date label or capture time is silently presented as an actual release
timestamp. Flow-dependent candidates remain inactive. The paired experiment
reports `blocked_missing_publication_time_history`; a full no-ETF refit is only
a regression baseline, not evidence of ETF uplift. No subscription was purchased.

## Validation record

- Python suite: **74 passed, 6 skipped**. The skips are optional PyTorch tests;
  this run did not train new deep models. The installed tabular candidates ran.
- TypeScript check, ESLint, production Next.js build and `git diff --check`: pass.
- Full monthly walk-forward: 35 folds per lane, 1,036 OOS target days from
  2024-01-01 through 2026-11-01. Lead/purge/calibration/fees/quotas unchanged.
- A subsequent fast refresh fetched closed BTC candles through 2026-11-03,
  evaluated four forecasts, and produced 234 official records (232 evaluated,
  two pending). All **232 pre-existing forecast digests** match HEAD exactly.
- Artifact verifier passes; public JSON schema remains version 5 with optional
  ETF fields. Health API reports zero overdue daily/event grades and explicitly
  reports the missing verified-publication feed.
- Browser: real public archive renders; Daily/Cumulative toggle and historical
  year filter tested. At 390px mobile width, document scroll width is also 390px.
  Fixed a stale tooltip position causing overflow after desktop-to-mobile resize.
  Dark tooltip, compact numeric labels, source attribution and exclusion warning
  verified. No browser runtime errors observed.

## Baseline, not ETF uplift

The following are the ensemble OOS rows saved in `data/reports/etf_ablation.json`.
These results do **not** meet the requested >70% directional-accuracy objective
and have negative expectancy. Execution remains subject to existing positive
expectancy gates; adding a feed or model name is not evidence of improvement.

| Metric | Index + Astro | Full Hybrid |
| --- | ---: | ---: |
| Exact accuracy | 17.91% | 19.02% |
| Weighted accuracy | 31.98% | 33.15% |
| Directional accuracy | 48.28% | 49.72% |
| Expectancy after configured costs | -0.0552% | -0.1109% |
| Brier | 0.64485 | 0.63819 |
| Coverage | 86.78% | 86.78% |
| Sharpe | -0.491 | -0.970 |
| Profit factor | 0.915 | 0.839 |
| Maximum drawdown | -58.13% | -81.68% |

## Remaining gate

No verified historical release-vintage dataset was found among the reviewed
public sources. A provider that supplies such evidence can plug into
`refresh_etf_flow()` without code changes. Until then, neither the public archive
nor synthetic fixtures may justify an ETF-driven promotion, attribution or OOS
improvement claim. Changes have been verified locally; deployment is separate.
