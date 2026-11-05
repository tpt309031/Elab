# ELAB Hybrid Quant Console

BTC decision-support research combining private energy indices, Astro scores,
technical features, historical patterns, analog search, and calibrated machine
learning. Publication-time US spot Bitcoin ETF flow is supported as a Full Hybrid
feature, conditional on a verified macro feed (currently awaiting source data).

## Validation contract

- Outer test: monthly rolling walk-forward, starting `2024-01-01`.
- Training window: 1,460 days; calibration window: 180 days, split into
  calibration fitting and separate policy selection.
- Purge: seven days between development and each test month.
- `FORECAST_LEAD_DAYS=2`; market features use only previously closed UTC candles.
- Index and Astro values are assumed known before the forecast session.
- No shuffle and no retrospective pivot labels in the predictive feature set.
- Maximum eight SIDEWAY and four NO CALL sessions per month, allocated
  chronologically without sorting by future confidence.
- Metrics include exact and weighted accuracy, directional accuracy, coverage,
  Brier score, Sharpe, profit factor, expectancy, net return, and max drawdown.

Daily scoring is fixed:

- `UP`: correct at `>= +3%`, partial from `+0.1%` to `< +3%`, otherwise wrong.
- `DOWN`: correct at `<= -3%`, partial from `> -3%` to `-0.1%`, otherwise wrong.
- `SIDEWAY`: correct inside `-1%` to `+1%`, otherwise wrong; no partial grade.

The directional target is `>70%` with positive expectancy. The dashboard shows
the measured OOS result and target gap; it does not force or relabel outcomes to
claim the target.

## Local development

```powershell
pnpm install
python -m pip install -r research/requirements-core.txt
python -m pytest tests/test_hybrid_core.py -q
python research/run_pipeline.py
pnpm dev
```

`public/data/hybrid_research.json` is the dashboard contract. The realtime API
at `/api/market` requests Binance first and falls back to OKX.

## Automated learning

`.github/workflows/daily-research.yml` runs after `03:00 UTC`, refreshes the
closed daily market data, re-grades past forecasts, refits/ranks candidates,
rebuilds the pattern registry and future calls, then commits the new artifact.
The calibrated policy keeps at most 8 SIDEWAY calls and 4 NO CALL decisions per
calendar month; weaker SIDEWAY utilities are reassigned to the stronger UP/DOWN
alternative before OOS scoring and future publication.

`data/learning_state.json` is the immutable production ledger. A published
lane/date forecast is never replaced; only its realized return, grade, and
evaluation timestamp are added after the candle is closed. Candidate-model and
matched-pattern predictions are stored with the official call, then used as
live Bayesian evidence on top of walk-forward OOS priors. Each daily run records
the active/standby selection snapshot, promotes higher adaptive ranks, demotes
weaker candidates, and runs `research/verify_artifact.py` before committing.
Every active pattern that fires for the published session is graded, while a
SHA-256 digest protects the immutable forecast fields from accidental rewrites.
The dashboard overlays this official ledger on regenerated OOS rows so the call
shown to the user is always the call that was actually issued.
The Sunday run enables the gated LSTM and Transformer candidates. Models enter
the active set only after independent OOS ranking; failed candidates remain in
standby and old forecasts stay in the historical ledger.

## ETF flow integration

Protocol: `causal-quota-calibrated-etf-v3`. See
[the external data contract](data/external/README.md) for schema, sources,
publication vintages and the refresh adapter. No historical release time is
invented, no mock observation enters production, and absent ETF data stays NaN.

Full Hybrid gains net flow, change, Z30, cumulative 3/7/14/30 reported-session
flow, flow/AUM when available, extreme inflow/outflow flags and interactions with
BTC psychology, volatility regime and index phase. Calendar remains Index + Astro.
All tabular candidates receive ETF features; Full Hybrid sequence candidates also
receive ETF channels. Two additional candidates, Flow HistGradientBoosting and
Flow Threshold Utility, require 90 valid PIT training rows. Their OOS priors and
live Bayesian grades are independent model records, with no arbitrary ranking
bonus for feature importance. Trading promotion additionally requires 365 OOS
observations, 20 live grades and the existing positive-expectancy gates.

Run `python research/evaluate_etf.py` for a paired control/treatment experiment.
The ensemble comparison includes the feature-plus-specialist-model package;
common-candidate rows compare feature effects with the same model family. This
experiment neither promotes models nor rewrites published forecasts. A historical
macro revision invalidates incompatible cached OOS evidence; ordinary later
publications do not retroactively change prior folds.

The chart workspace includes daily/cumulative ETF flow in USD, extreme-flow price
annotations at eligible forecast dates, descriptive lagged correlation and a
Logistic coefficient snapshot for new published calls. Coefficient terms are
uncalibrated logits, not ensemble probability contributions. Old immutable calls
are never given a new ETF explanation after publication. Data Health and
`/api/health` expose the macro/ETF status and latest public release timestamp.

Without a verified publication feed, the panel can show the separately cached
public TFTC history from January 2024, with attribution and an explicit
**display-only / excluded from training** notice. `python -m research.etf_sources`
refreshes that archive; the daily workflow also refreshes it and saves changed
snapshots. Capture times and provider trading-date labels are never substituted
for historical publication timestamps. See the source comparison and timestamp
limitations in [the external data contract](data/external/README.md).
