# Optional Point-in-Time Data Feeds

The core pipeline works when every file in this directory is absent. Missing feeds are reported as unavailable and are never replaced by zeroes. Add a file only when its historical publication timestamps are trustworthy.

Non-macro CSVs use the legacy contract:

- `date`: UTC forecast date for which the value may be used.
- `available_at`: UTC timestamp at which the value became known, for example `2026-07-12T23:00:00Z`.

If `available_at` is later than `date 00:00:00 UTC`, the value is excluded from that date's forecast features. If `available_at` is omitted, the file is treated as prepublished but receives an imputation warning in Data Health.

Supported files and columns:

## `derivatives.csv`

- `funding_rate`
- `open_interest`
- `basis`
- `liquidations_long`
- `liquidations_short`

## `options.csv`

- `options_iv`
- `options_skew_25d`

## `macro.csv`

This feed uses a stricter observation/publication contract, not the legacy
forecast-date alignment above. The committed file is intentionally header-only:
no verified publication-time ETF history or API credentials have been supplied.
Synthetic data exists only in tests, never in the live ledger or claimed OOS results.

Required schema: `date,available_at,etf_net_flow`.
Optional columns: `etf_aum,dxy,us10y`.

- `date`: US ETF trading/observation date, `YYYY-MM-DD`.
- `available_at`: the actual public release timestamp, with an explicit timezone;
  normalized to UTC without rounding away subsecond precision. Missing/naive
  timestamps are rejected, not inferred from a closing bell or trading date.
- `etf_net_flow`: aggregate US spot Bitcoin ETF net creations/redemptions in USD,
  positive for inflow. Convert a source quoted in US$ millions by multiplying
  by 1,000,000 upstream. Do not mix Hong Kong ETFs, futures ETFs or BTC units.
- `etf_aum`: optional total assets in USD, known at this publication time;
  strictly positive. Flow/AUM is unavailable if AUM is missing.
- Every correction is a new `(date, available_at)` vintage. A same-key conflicting
  value is rejected. Send complete ETF snapshots for revisions, not partial
  replacement fields. An AUM estimate published later must not be attached to an
  earlier timestamp.

### Point-in-time alignment

`FORECAST_LEAD_DAYS=2` remains unchanged. For forecast target T, only observations
dated through T-2 with `available_at <= T-1 00:00 UTC` are eligible. For example,
a Friday observation first public Saturday afternoon cannot enter Sunday's
forecast, but can enter Monday's. Later corrections never change earlier feature
vintages. Rows with publication times beyond the current run are excluded.

The last eligible flow may carry across weekends for up to seven calendar days
of observation age at the target. It is not a new daily flow or a zero-flow day.
Rolling 3/7/14/30 sums use that many distinct **reported trading observations**,
requiring the full window. Z30 uses up to 30 reports, at least 10; zero variance
produces missing z-score, not an extreme. No missing observations are zero-filled.
Features expire rather than carrying an old ETF observation indefinitely into
the future calendar. Price markers show the eligible target date, not a
retrospectively backdated observation date.

### Source and ingestion

Reference sources were reviewed during 2026-11-02 to 2026-11-05:
[Farside](https://farside.co.uk/bitcoin-etf-flow-all-data/) publishes the daily
aggregate in US$ millions; [CoinGlass](https://docs.coinglass.com/reference/etf-flows-history)
documents a historical USD flow endpoint. A daily timestamp alone is **not**
evidence of the first public release time or vintage. Neither source is silently
treated as a publication-stamped historical feed here. Use a licensed or archived
provider that supplies actual release timestamps and revision history; retain
the provider's publication evidence outside the CSV for audit.

### Public history without invented publication times

`python -m research.etf_sources` refreshes `etf_history.json` from
[TFTC's public JSON](https://www.tftc.io/bitcoin-etf-flows/data.json).
The Commoner Bitcoin ETF Tracker by TFTC supplies history from 2024-01-11,
updates at 02:00/10:00 UTC, and permits reuse under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Its aggregates originate from SoSoValue and earlier fund details from Farside.
See [the provider methodology](https://www.tftc.io/bitcoin-etf-flows/methodology).

This archive is **display-only**. The provider's `dateModified` is a synthetic
22:00 trading-date label, not proof of publication time. `available_at` stays
null, `training_eligible=false`, and neither the archive nor its capture times
can enter `build_feature_frame`, model ranking or the ETF ablation. ELAB removes
the provider's zero/no-fund holiday placeholders and recalculates cumulative
totals. Source attribution and this transformation are displayed on the chart.

Each changed dataset gets an immutable, timestamped snapshot under
`etf_snapshots/`, with a content hash. Identical data retains its original capture
time; `checked_at` records a later successful fetch. This is evidence of what was
observed, **not** a claim about original publication time. Regressed coverage,
invalid units or invalid values leave the prior validated cache untouched.
The daily workflow refreshes and preserves the archive alongside its snapshots.

Other candidates reviewed:

| Source | Useful coverage/access | PIT limitation |
| --- | --- | --- |
| [Farside](https://farside.co.uk/) | Daily USD-million flow table, updated as issuers report | No per-row release-vintage archive established |
| [CoinGlass](https://docs.coinglass.com/reference/etf-flows-history) | Keyed aggregate and fund-level USD history API | Documented timestamp is the observation date |
| [CryptoETF](https://cryptoetf.today/en/api) | Free-key recent daily API; public publication status | Response-wide updatedAt is not each historical row's publication |
| [Glassnode](https://docs.glassnode.com/basic-api/endpoints/institutions) | Issuer-based balance-change flow estimates | Different construction; do not silently substitute for aggregate creation/redemption flow |

No paid subscription, hidden endpoint or account was created. Historical ETF
uplift remains unverified until release-vintage evidence is obtained. A future
first-observed-time protocol would require an explicit versioned decision; this
change does not silently weaken the requested actual-publication contract.

The refresh adapter accepts a trusted HTTPS JSON list, or `{ "records": [...] }`,
with the CSV fields above. Configure `ELAB_ETF_FLOW_URL` and optionally
`ELAB_ETF_FLOW_TOKEN` (Bearer header). In GitHub Actions, use repository variable
`ELAB_ETF_FLOW_URL` and secret `ELAB_ETF_FLOW_TOKEN`. The daily workflow preserves
the updated CSV alongside the artifact. No endpoint is enabled by default.

```powershell
python -m research.etf_flow --input verified-publications.json
python -m research.etf_flow  # configured HTTPS adapter; no-op without URL
python research/run_pipeline.py --no-refresh
python research/evaluate_etf.py
```

`refresh_etf_flow(path, records=...)` is the offline/mock adapter used by tests.
It validates the complete merge before atomically replacing the CSV. Failed
refreshes leave the cache intact and appear in feed health. Never import mocked
records into `data/external/macro.csv` for deployment.

The paired evaluation writes `data/reports/etf_ablation.json`, without touching
the learning ledger. It compares identical folds/costs/quotas, including exact,
weighted and directional accuracy, expectancy, Brier, coverage and Sharpe.
With no verified history it reports a blocked comparison, not invented improvement.

## `onchain.csv`

- `onchain_active_addresses`
- `exchange_netflow`

Minimal example:

```csv
date,available_at,funding_rate,open_interest,basis
2026-07-13,2026-07-12T23:59:00Z,0.0001,12345678900,0.0018
```

Do not backfill a historical series with information published after the forecast cutoff. Such a file would create look-ahead leakage even if its `date` column appears correct.
