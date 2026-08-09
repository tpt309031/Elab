export type ForecastDirection = "up" | "down" | "sideway" | "no-call";
export type ForecastStatus = "correct" | "partial" | "wrong" | "pending" | "no-call";

export interface MarketRow {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndexRow {
  date: string;
  index_BTC: number | null;
  index_me: number | null;
}

export interface SimilarCase {
  date: string;
  move: number;
  outcome: "up" | "down" | "sideway";
  similarity: number;
}

export interface SourceAttempt {
  provider: string;
  status: string;
  detail?: string;
}

export interface MarketHealth {
  status: string;
  expected_closed_utc: string;
  actual_closed_utc: string;
  cache_latest_before_refresh?: string;
  selected_provider?: string | null;
  provider_count?: number;
  cross_exchange_close_discrepancy_bps?: number | null;
  stale: boolean;
  attempts: SourceAttempt[];
}

export interface IntradayHealth {
  timeframe: string;
  status: string;
  provider?: string | null;
  expected_open_utc?: string | null;
  actual_open_utc?: string | null;
  stale?: boolean | null;
  rows: number;
  error?: string;
}

export interface ExternalHealth {
  source: string;
  available: boolean;
  status: string;
  rows: number;
  usable_rows: number;
}

export interface PatternReference {
  id: string;
  name: string;
  direction: ForecastDirection;
  occurrences: number;
  weighted_accuracy: number;
  rank: number;
  signal_lag_days?: number;
  duration_days?: number;
}

export interface ForecastRow {
  date: string;
  lane?: string;
  model?: string;
  fold?: string;
  forecast: ForecastDirection;
  status: ForecastStatus;
  score?: number | null;
  daily_return?: number | null;
  strategy_return?: number | null;
  confidence?: number;
  prob_down: number;
  prob_sideway: number;
  prob_up: number;
  expected_score: number;
  decision_margin: number;
  entropy: number;
  top_pattern?: PatternReference | null;
  similar_cases?: SimilarCase[];
  model_members?: string[];
  model_weights?: number[];
  policy_mode?: string;
  sideway_penalty?: number;
  sideway_cap_override?: boolean;
  contract_version?: number;
  information_cutoff_utc?: string;
  target_start_utc?: string;
  target_end_utc?: string;
  trade_action?: "up" | "down" | "flat";
  trade_eligible?: boolean;
  trade_gate_reason?: string;
  expected_net_return?: number | null;
  expectancy_lcb?: number | null;
  execution_model?: string | null;
  execution_model_forecast?: ForecastDirection | null;
}

export interface OfficialForecastRow extends ForecastRow {
  forecast_id: string;
  target_date: string;
  issued_at: string;
  closed_through_at_issue: string;
  actual_return: number | null;
  evaluated_at: string | null;
  immutable_digest?: string;
}

export interface ModelMetric {
  lane: string;
  rank: number;
  model: string;
  observations: number;
  calls: number;
  coverage: number;
  no_calls: number;
  exact_accuracy: number;
  weighted_accuracy: number;
  directional_accuracy: number;
  balanced_accuracy: number;
  mcc?: number;
  brier: number;
  log_loss?: number;
  ece?: number;
  sharpe: number;
  profit_factor: number | null;
  max_drawdown: number;
  expectancy: number;
  net_return: number;
  turnover?: number;
  exact_lcb?: number;
  exact_ucb?: number;
  weighted_lcb?: number;
  weighted_ucb?: number;
  directional_lcb?: number;
  directional_ucb?: number;
  expectancy_lcb?: number;
  expectancy_ucb?: number;
  rank_score: number;
  status: "active" | "standby";
  live_samples?: number;
  live_weighted_accuracy?: number | null;
  live_directional_accuracy?: number | null;
  live_expectancy?: number | null;
  adjusted_weighted_accuracy?: number;
  adaptive_rank_score?: number;
  selection_change?: "promoted" | "demoted" | "retained" | "standby";
  replacement_reason?: string;
  trade_eligible?: boolean;
  drift_guard_active?: boolean;
}

export interface PatternMetric {
  pattern_id: string;
  pattern: string;
  expression: string;
  direction: ForecastDirection;
  occurrences: number;
  weighted_accuracy: number;
  weighted_lcb?: number;
  exact_accuracy: number;
  expectancy: number;
  false_discovery_q?: number;
  eligible?: boolean;
  examples: string[];
  rank: number;
  status: "active" | "standby";
  live_occurrences?: number;
  live_weighted_accuracy?: number | null;
  adjusted_weighted_accuracy?: number;
  adaptive_rank_score?: number;
  selection_change?: "promoted" | "demoted" | "retained" | "standby";
  replacement_reason?: string;
  pattern_family?: string;
  signal_lag_days?: number;
  duration_days?: number;
  accuracy_lift?: number;
  conservative_lift?: number;
  statistically_supported?: boolean;
}

export interface SourceRevision {
  recorded_at: string;
  digest: string;
  changed_sources: string[];
  sources: Array<Record<string, string | number | null>>;
}

export interface LearningSummary {
  official_forecasts: number;
  evaluated_forecasts: number;
  pending_forecasts: number;
  no_calls: number;
  correct: number;
  partial: number;
  wrong: number;
  live_weighted_accuracy: number | null;
  last_evaluated_date: string | null;
  last_selection_date: string | null;
  event_evaluations?: number;
  evaluated_events?: number;
  matched_events?: number;
  event_match_rate?: number | null;
}

export interface EventEvaluation {
  event_id: string;
  forecast_id: string;
  target_date: string;
  lane: string;
  source_type: "official" | "model" | "pattern";
  source_name: string;
  direction: ForecastDirection;
  window_start: string;
  window_end: string;
  matures_after: string;
  status: "pending" | "matched" | "not-matched";
  score: number | null;
  event_date: string | null;
  lead_lag_days: number | null;
  matched_event_types: string[];
  evaluated_at: string | null;
}

export interface EquityPoint {
  date: string;
  equity: number;
  benchmark: number;
  drawdown: number;
}

export interface MonthlyMetric {
  lane: string;
  month: string;
  days: number;
  calls: number;
  no_calls: number;
  correct: number;
  partial: number;
  wrong: number;
  exact_accuracy: number;
  weighted_accuracy: number;
  expectancy: number;
}

export interface ImportanceRow {
  feature: string;
  importance: number;
  importance_std: number;
  model: string;
  method: string;
}

export interface ClassMetric {
  lane: string;
  class: "down" | "sideway" | "up";
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ConfusionCell {
  lane: string;
  actual: "down" | "sideway" | "up";
  predicted: "down" | "sideway" | "up";
  count: number;
  row_rate: number;
}

export interface ConfidenceRiskPoint {
  lane: string;
  coverage: number;
  minimum_confidence: number;
  exact_accuracy: number;
  weighted_accuracy: number;
  expectancy: number;
}

export interface GroupedPerformance {
  lane: string;
  dimension: string;
  value: string;
  calls: number;
  exact_accuracy: number;
  weighted_accuracy: number;
  directional_accuracy: number;
  expectancy: number;
}

export interface FeatureDrift {
  feature: string;
  psi: number;
  mean_shift_z: number;
  reference_missing: number;
  recent_missing: number;
  status: "stable" | "watch" | "alert";
}

export interface ClassDrift {
  class: "down" | "sideway" | "up";
  reference_share: number;
  recent_share: number;
  change: number;
}

export interface FoldMetric {
  lane: string;
  fold: string;
  train_start: string;
  train_end: string;
  calibration_fit_end: string;
  policy_start: string;
  calibration_end: string;
  test_start: string;
  test_end: string;
  members: string[];
  weights: number[];
  stacking_method?: string;
  stacking_log_loss?: number;
  uniform_ensemble_log_loss?: number;
}

export interface ResearchArtifact {
  meta: {
    schema_version: number;
    generated_at: string;
    market_provider: string;
    latest_closed_utc: string;
    pipeline_mode?: "fast-daily" | "full-research";
    forecast_cutoff_utc?: string;
    first_publishable_target_utc?: string;
    oos_start: string;
    oos_end: string;
    index_start: string;
    index_end: string;
    target_directional_accuracy: number;
    achieved_directional_accuracy: number;
    target_reached: boolean;
    deep_research_enabled: boolean;
    scoring: Record<string, string>;
    availability_assumption: string;
    data_lineage?: Array<Record<string, string | number | boolean | null>>;
    provenance?: {
      status: "verified" | "research-only";
      private_point_in_time_coverage: number;
      warnings: string[];
      revision_digest?: string;
    };
    validation: Record<string, string | number | boolean | null>;
  };
  health?: {
    market: MarketHealth;
    intraday: IntradayHealth[];
    external: ExternalHealth[];
    last_evaluation: {
      latest_closed_utc: string;
      evaluated_forecasts_this_run: number;
      run_at: string;
    };
  };
  market: MarketRow[];
  indices: IndexRow[];
  forecast: {
    calendar: ForecastRow[];
    full_hybrid_next_session: ForecastRow[];
    historical_calendar_oos: ForecastRow[];
    historical_full_hybrid_oos: ForecastRow[];
  };
  performance: {
    model_rankings: ModelMetric[];
    monthly: MonthlyMetric[];
    calendar_equity: EquityPoint[];
    full_hybrid_equity: EquityPoint[];
    calendar_reliability: Array<{ bucket: number; confidence: number; observed_accuracy: number; count: number }>;
    full_hybrid_reliability: Array<{ bucket: number; confidence: number; observed_accuracy: number; count: number }>;
    calendar_folds?: FoldMetric[];
    full_hybrid_folds?: FoldMetric[];
    calendar_no_calls?: Array<{ month: string; days: number; no_calls: number }>;
    full_hybrid_no_calls?: Array<{ month: string; days: number; no_calls: number }>;
    class_metrics?: ClassMetric[];
    confusion_matrix?: ConfusionCell[];
    confidence_risk?: ConfidenceRiskPoint[];
    grouped?: GroupedPerformance[];
  };
  models: {
    availability: Array<{ model: string; family: string; available: boolean; cadence: string }>;
    calendar_latest_selection: Array<Record<string, string | number | boolean | null>>;
    full_hybrid_latest_selection: Array<Record<string, string | number | boolean | null>>;
  };
  patterns: {
    calendar: PatternMetric[];
    full_hybrid: PatternMetric[];
  };
  explainability: {
    method: string;
    calendar: ImportanceRow[];
    full_hybrid: ImportanceRow[];
  };
  research: {
    correlation_heatmap: Array<Record<string, string | number | null>>;
    feature_groups: Record<string, string[]>;
    drift?: {
      features: FeatureDrift[];
      classes: ClassDrift[];
      performance: Record<string, {
        alarm: boolean;
        action?: string;
        statistic: number;
        threshold: number;
        observations: number;
        baseline_loss?: number;
        recent_loss?: number;
        deterioration?: number;
      }>;
    };
    event_definitions?: Record<string, string | number>;
  };
  learning?: {
    summary: LearningSummary;
    official_forecast_ledger: OfficialForecastRow[];
    event_evaluation_ledger?: EventEvaluation[];
    selection_history: Array<Record<string, unknown>>;
    source_revisions?: SourceRevision[];
    evaluated_this_run: number;
    bootstrapped_this_run: number;
  };
}

export interface DeepModelMetric {
  model: string;
  rank: number;
  observations: number;
  oos_start: string;
  oos_end: string;
  exact_accuracy: number;
  weighted_accuracy: number;
  directional_accuracy: number;
  directional_lcb: number;
  mcc: number;
  log_loss: number;
  ece: number;
  expectancy: number;
  profit_factor: number | null;
  net_return: number;
  max_drawdown: number;
  rank_score: number;
  status: "challenger";
  promotion_eligible: boolean;
  promotion_reason: string;
}

export interface DeepResearchArtifact {
  meta: {
    schema_version: number;
    generated_at: string;
    latest_closed_daily_utc: string;
    dataset_start: string;
    dataset_end: string;
    samples: number;
    lookback_4h_steps: number;
    input_features: number;
    epochs: number;
    folds: number;
    official_ledger_isolation: boolean;
    promotion_gate: string;
  };
  dataset: {
    feature_names: string[];
    last_bar_rule: string;
    maximum_last_bar_violation: number;
  };
  models: {
    availability: Array<{ model: string; architecture: string; available: boolean; status: string }>;
    rankings: DeepModelMetric[];
    folds: Array<Record<string, string | number>>;
  };
  latest_forecasts: Array<{
    model: string;
    target_date: string;
    available_through_utc: string;
    forecast: ForecastDirection;
    confidence: number;
    prob_down: number;
    prob_sideway: number;
    prob_up: number;
    status: "research-only";
  }>;
}

export interface SystemHealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  checkedAt: string;
  expectedClosedUtc: string;
  artifact: {
    latestClosedUtc: string | null;
    generatedAt: string | null;
    stale: boolean;
    marketHealth: string | null;
    error?: string;
  };
  deep: {
    available: boolean;
    generatedAt: string | null;
    stale: boolean;
    architectures: number;
    error?: string;
  };
}

export interface LiveMarketResponse {
  provider: string;
  timeframe: string;
  rows: MarketRow[];
  fetchedAt: string;
}
