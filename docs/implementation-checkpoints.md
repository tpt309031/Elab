# Hybrid Research Upgrade Checkpoints

This file is the recovery manifest for the ELAB upgrade. Each checkpoint must be independently testable and committed before the next checkpoint starts. A resumed task should read this file, inspect `git status`, and continue from the first incomplete checkpoint.

## Locked Scope

The production forecast remains a three-class decision problem: `UP`, `DOWN`, and `SIDEWAY`, with the existing `NO CALL` safety state. The official daily grade remains immutable and has only `Correct`, `Partial`, or `Wrong` for scored calls.

The following proposals are explicitly excluded:

- Seven-bin return-distribution targets.
- A rule that publishes a daily probability but permits a trade only when post-cost expectancy is positive.

Expected grading utility may still rank the three forecast actions. Expectancy remains a model metric and ranking input, but it is not a hard gate that suppresses every non-positive daily action.

## Safety Rules

- Never rewrite an issued official forecast or its immutable digest.
- Never delete the original private-index CSV files or Astro source.
- Generated JSON and cache files are replaced atomically.
- New schemas are additive and versioned; the frontend must tolerate a missing optional section.
- Every market, index, Astro, external, and model input carries source and availability metadata.
- A feature is usable only when its value was available before the forecast cutoff.
- Daily forecast grading and delayed event grading are separate ledgers.
- No in-sample result can promote a production model.
- Every checkpoint ends with tests, artifact verification, frontend checks when applicable, and a dedicated Git commit.

## Checkpoint 0: Architecture Audit

Status: complete

Deliverables:

- Inventory existing data, model, learning, workflow, and UI boundaries.
- Lock exclusions and safety rules in this manifest.
- Preserve the July 12 freshness repair and immutable learning state.

Baseline as of July 13, 2026:

- Daily market history: 2017-08-17 through 2026-07-12.
- Private indices: 2024-01-01 through 2028-12-31.
- Astro data: 2017-01-01 through 2028-12-31.
- OOS rows per lane: 924.
- Calendar ensemble directional accuracy: 50.38%.
- Full Hybrid ensemble directional accuracy: 53.01%.

## Checkpoint 1: Data Provenance and Feature Foundation

Status: complete

Files owned by this checkpoint:

- `research/data_sources.py`
- `research/hybrid_core.py` data-loading and feature sections
- `research/run_pipeline.py` artifact metadata
- `tests/test_data_sources.py`
- `data/external/README.md`

Deliverables:

- Market health object with expected candle, actual candle, provider attempts, latency, errors, cross-exchange close discrepancy, and stale state.
- Optional `available_at` support for private index and Astro values, with an explicit imputation flag when source timestamps are absent.
- Forecast cutoff and data-lineage fields.
- Optional external-data adapters for funding, open interest, basis, liquidations, options skew, ETF flow, DXY, rates, and on-chain inputs. Missing optional feeds remain visible and never become fabricated zeroes.
- Binance 1h and 4h cache refresh with strict closed-bar checks.
- Leakage-safe daily aggregates from intraday realized volatility, trend, range, wick, and signed-volume proxy.

Acceptance gate:

- Stale daily data fails closed.
- Optional missing feeds do not break the core pipeline.
- Availability timestamps after the forecast cutoff are excluded.
- Unit tests prove source fallback, lineage, and intraday aggregation behavior.

## Checkpoint 2: Evaluation, Calibration, Events, and Governance

Status: complete

Files owned by this checkpoint:

- `research/evaluation.py`
- `research/hybrid_core.py` validation and metrics sections
- `research/learning.py`
- `research/verify_artifact.py`
- `tests/test_evaluation.py`
- `tests/test_learning.py`

Deliverables:

- Nested rolling walk-forward with purge and horizon-specific embargo.
- Calibration-fit and policy-selection windows that remain strictly before each test month.
- Sigmoid, temperature, and sample-gated isotonic calibration selected only from OOS validation loss.
- Exact, weighted, balanced accuracy, MCC, class precision/recall, Brier, log loss, ECE, expectancy, profit factor, Sharpe, drawdown, turnover, and coverage.
- Moving-block bootstrap confidence intervals.
- Daily score ledger kept separate from a `+/-3 day` pivot and large-move event ledger.
- Daily reranking, minimum 20 live observations for promotion evidence, conservative lower confidence bounds, and monthly production promotion.
- Feature/class/calibration drift diagnostics using PSI and Page-Hinkley style alarms.

Acceptance gate:

- No test date appears in training, calibration, stacking, policy, or pattern ranking data.
- Daily grades cannot be changed by delayed event matches.
- A candidate with fewer than 20 live grades cannot displace a retained champion using live evidence alone.

## Checkpoint 3: Tabular and Hybrid Model Suite

Status: complete

Files owned by this checkpoint:

- `research/model_candidates.py`
- `research/hybrid_core.py` model orchestration sections
- `research/requirements-core.txt`
- `tests/test_models.py`

Models to train:

- Multinomial Logistic Regression.
- Ordinal Logistic Regression.
- Random Forest and HistGradientBoosting baselines.
- CatBoost, LightGBM, and XGBoost.
- Quantile boosting for return magnitude, converted to the existing three-class probabilities.
- Two-stage hurdle model for magnitude and conditional direction.
- HMM regime classifier and deterministic change-point regime features.
- Historical analog and pattern registry.
- Non-negative simplex-constrained OOF stacking ensemble.

Acceptance gate:

- Every available model produces purged OOS probabilities and metrics.
- Missing optional libraries produce an explicit unavailable state, not a silent omission.
- Stacking weights are learned only from pre-test OOS predictions and sum to one.

## Checkpoint 4: Intraday Deep Challengers

Status: pending

Files owned by this checkpoint:

- `research/deep_models.py`
- `research/train_deep_challengers.py`
- `research/requirements-deep.txt`
- `.github/workflows/deep-research.yml`
- `tests/test_deep_dataset.py`

Models to train on leakage-safe 1h/4h sequences:

- LSTM baseline.
- Compact Transformer baseline.
- Temporal Convolutional Network.
- PatchTST-style patch encoder.
- Compact Temporal Fusion Transformer.
- iTransformer-style variable-token encoder.

Deliverables:

- Multi-timeframe sequences use bars strictly earlier than the target UTC day.
- Deep OOS predictions and latest challenger state are persisted separately from the official ledger.
- Weekly training is isolated from the daily freshness job.
- Deep candidates remain challengers until confidence-bound and live-evidence promotion gates are satisfied.

Acceptance gate:

- Dataset leakage tests pass.
- Every architecture completes at least one chronological training run and records its sample count, OOS period, calibration metrics, and status.
- Daily production still operates when the deep artifact is missing or stale.

## Checkpoint 5: Dashboard and Alerts

Status: pending

Files owned by this checkpoint:

- `src/lib/types.ts`
- `src/components/dashboard/*`
- `src/app/api/health/route.ts`
- `.github/workflows/daily-research.yml`

Deliverables:

- Data Health panel with expected/actual candle, source attempts, discrepancy, last evaluation, retry state, and stale alert.
- Existing three-class probability and expected-score presentation, without a seven-bin chart.
- Champion/challenger table with confidence intervals, sample size, calibration, expectancy, and promotion reason.
- Confusion matrix and precision/recall by class, year, and regime.
- Reliability, ECE, confidence-risk, and forecast-distribution charts.
- Drift dashboard for features, classes, calibration, and live outcomes.
- Historical analog mini charts with source windows and subsequent returns.
- Separate Pivot/Event Lab for delayed event research.
- GitHub failure issue/summary and visible in-app pipeline alert.
- Responsive layouts for desktop and mobile.

Acceptance gate:

- Missing optional sections render an informative empty state.
- No horizontal calendar overflow on mobile.
- Lint, TypeScript, production build, and smoke fetch pass.

## Checkpoint 6: Full Rebuild and Release

Status: pending

Deliverables:

- Rebuild OOS research from 2024 through the latest closed UTC candle.
- Run core and deep test suites.
- Verify immutable daily and event ledgers.
- Compare the new champion against the frozen baseline with confidence intervals.
- Publish generated artifacts, push GitHub, wait for Vercel `READY`, inspect runtime errors, and verify production data freshness.
- Record limitations honestly; the 70% target is a gate, not a value to manufacture.

## Resume Procedure

1. Read this manifest and `docs/accuracy-roadmap.md`.
2. Run `git status --short` and do not overwrite unrelated changes.
3. Verify the latest completed checkpoint commit.
4. Run the checkpoint-specific tests before editing the next ownership area.
5. Update one status in this file only after its acceptance gate passes.
