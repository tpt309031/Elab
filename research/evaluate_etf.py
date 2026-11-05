"""Paired, publication-time ETF ablation; never opens or writes the learning ledger."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.checkpoints import checkpointed_walk_forward
from research.data_sources import build_intraday_daily_features, load_external_features
from research.etf_flow import MIN_FLOW_TRAINING_ROWS, load_macro
from research.hybrid_core import RESEARCH_PROTOCOL, build_feature_frame, load_astro, load_indices, serialize_frame

METRICS = ("exact_accuracy", "weighted_accuracy", "directional_accuracy", "expectancy",
           "brier", "coverage", "sharpe", "profit_factor", "max_drawdown", "observations")


def compare_metrics(control: pd.DataFrame, treatment: pd.DataFrame) -> list[dict[str, object]]:
    """Only compare matching candidate names; ensemble names are lane-specific."""
    rows = []
    for model in sorted(set(control["model"]) & set(treatment["model"])):
        left = control.loc[control["model"] == model].iloc[0]
        right = treatment.loc[treatment["model"] == model].iloc[0]
        def finite(value: object) -> float | None:
            return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

        baseline = {key: finite(left[key]) for key in METRICS}
        added = {key: finite(right[key]) for key in METRICS}
        rows.append(dict(model=model, control=baseline, treatment=added,
            delta={key: added[key] - baseline[key] if added[key] is not None and baseline[key] is not None else None
                   for key in METRICS}))
    return rows


@threadpool_limits.wrap(limits=2)
def evaluate(root: Path = ROOT) -> dict[str, object]:
    macro, health = load_macro(root / "data/external/macro.csv")
    known = macro.dropna(subset=["etf_net_flow"]).drop_duplicates("date")
    report: dict[str, object] = dict(protocol=RESEARCH_PROTOCOL, health=health,
        metrics=list(METRICS), verified_observations=len(known),
        comparison=[], status="blocked_missing_publication_time_history",
        reason="Need verified ETF publication history; mock observations cannot establish OOS improvement.")
    if len(known) < MIN_FLOW_TRAINING_ROWS:
        artifact = root / "data/hybrid_research.json"
        if artifact.exists():
            previous = json.loads(artifact.read_text(encoding="utf-8"))
            report["published_reference"] = dict(generated_at=previous["meta"]["generated_at"],
                protocol=previous["meta"]["validation"]["protocol"],
                metrics=[{k: row.get(k) for k in ("lane", "model", *METRICS)}
                    for row in previous["performance"]["model_rankings"] if "Ensemble" in row["model"]])
        return report
    market = pd.read_csv(root / "data/cache/BTCUSDT_1d.csv", parse_dates=["timestamp"])
    indices = load_indices(root / "data/newdata/index_btc.csv", root / "data/newdata/index_me.csv")
    astro = load_astro(root / "public/data/astro_scores.json")
    intraday = [(timeframe, pd.read_csv(root / f"data/cache/BTCUSDT_{timeframe}.csv", parse_dates=["timestamp"]))
                for timeframe in ("1h", "4h")]
    external, _, _ = load_external_features(root / "data/external")
    volume_path = root / "data/cache/binance_spot_volume_1d.csv"
    volume = pd.read_csv(volume_path, parse_dates=["date"]) if volume_path.exists() else None
    frame, groups = build_feature_frame(indices, market, astro, build_intraday_daily_features(intraday), external, volume, macro)
    analog = ["index_BTC", "index_me", "gap_index", "index_btc_change_1", "index_me_change_1",
        "index_btc_slope_3", "index_me_slope_3", "index_corr_5", "composite", "finance",
        "volatility", "astro_composite_change_1", "astro_volatility_z30", "dow_sin", "dow_cos",
        "market_return_1", "volatility_7", "rsi14", "atr14_pct"]
    control_columns = [column for column in groups["full"] if column not in groups["etf"]]
    control = checkpointed_walk_forward(root, frame, control_columns, analog, "ETF control", [], 0, False)
    treatment = checkpointed_walk_forward(root, frame, groups["full"], analog, "ETF treatment", [], 0, False)
    assert control.forecasts["date"].tolist() == treatment.forecasts["date"].tolist()
    left = control.model_metrics.copy()
    right = treatment.model_metrics.copy()
    left["model"] = left["model"].replace({"ETF control Ensemble": "Ensemble"})
    right["model"] = right["model"].replace({"ETF treatment Ensemble": "Ensemble"})
    report.update(status="evaluated", reason=None, comparison=compare_metrics(left, right),
        treatment_models=serialize_frame(right), control_folds=serialize_frame(control.fold_metrics),
        treatment_folds=serialize_frame(treatment.fold_metrics),
        start=control.forecasts["date"].min().strftime("%Y-%m-%d"),
        end=control.forecasts["date"].max().strftime("%Y-%m-%d"),
        note="Same monthly purges, calibration, costs and quotas. Ensemble measures the feature-plus-candidate package; common candidate rows isolate the feature change. No live promotion is performed.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/etf_ablation.json")
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verified_observations": result["verified_observations"], "output": str(args.output)}))
