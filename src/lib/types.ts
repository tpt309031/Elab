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

export interface PatternReference {
  id: string;
  name: string;
  direction: ForecastDirection;
  occurrences: number;
  weighted_accuracy: number;
  rank: number;
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
  brier: number;
  sharpe: number;
  profit_factor: number | null;
  max_drawdown: number;
  expectancy: number;
  net_return: number;
  rank_score: number;
  status: "active" | "standby";
}

export interface PatternMetric {
  pattern_id: string;
  pattern: string;
  expression: string;
  direction: ForecastDirection;
  occurrences: number;
  weighted_accuracy: number;
  exact_accuracy: number;
  expectancy: number;
  examples: string[];
  rank: number;
  status: "active" | "standby";
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

export interface ResearchArtifact {
  meta: {
    schema_version: number;
    generated_at: string;
    market_provider: string;
    latest_closed_utc: string;
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
    validation: Record<string, string | number>;
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
  };
  models: {
    availability: Array<{ model: string; family: string; available: boolean; cadence: string }>;
    calendar_latest_selection: Array<Record<string, string | number>>;
    full_hybrid_latest_selection: Array<Record<string, string | number>>;
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
  };
}

export interface LiveMarketResponse {
  provider: string;
  timeframe: string;
  rows: MarketRow[];
  fetchedAt: string;
}
