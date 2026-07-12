# ELAB Hybrid Quant Console

BTC decision-support research combining private energy indices, Astro scores,
technical features, historical patterns, analog search, and calibrated machine
learning.

## Validation contract

- Outer test: monthly rolling walk-forward, starting `2024-01-01`.
- Training window: 1,460 days; calibration window: 90 days.
- Purge: five days between development and each test month.
- Market features are shifted and use only previously closed UTC candles.
- Index and Astro values are assumed known before the forecast session.
- No shuffle and no retrospective pivot labels in the predictive feature set.
- A maximum of six `no-call` sessions is allowed in each month.
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
The calibrated policy keeps at most 8 SIDEWAY calls and 6 NO CALL decisions per
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
