# Optional Point-in-Time Data Feeds

The core pipeline works when every file in this directory is absent. Missing feeds are reported as unavailable and are never replaced by zeroes. Add a file only when its historical publication timestamps are trustworthy.

Each CSV requires:

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

- `etf_net_flow`
- `dxy`
- `us10y`

## `onchain.csv`

- `onchain_active_addresses`
- `exchange_netflow`

Minimal example:

```csv
date,available_at,funding_rate,open_interest,basis
2026-07-13,2026-07-12T23:59:00Z,0.0001,12345678900,0.0018
```

Do not backfill a historical series with information published after the forecast cutoff. Such a file would create look-ahead leakage even if its `date` column appears correct.
