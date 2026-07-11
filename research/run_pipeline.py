from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.hybrid_core import (
    OOS_START,
    build_feature_frame,
    equity_curve,
    feature_heatmap,
    fit_latest_forecasts,
    load_astro,
    load_indices,
    refresh_daily_market,
    reliability_bins,
    run_walk_forward,
    serialize_frame,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe hybrid BTC research artifacts")
    parser.add_argument("--deep", action="store_true", help="Evaluate LSTM and Transformer candidates")
    parser.add_argument("--no-refresh", action="store_true", help="Use the committed daily market cache")
    parser.add_argument("--output", type=Path, default=ROOT / "public" / "data" / "hybrid_research.json")
    return parser.parse_args()


def _monthly_metrics(forecasts: pd.DataFrame, lane: str) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    data = forecasts.copy()
    data["month"] = data["date"].dt.strftime("%Y-%m")
    rows = []
    for month, group in data.groupby("month"):
        calls = group[group["forecast"] != "no-call"]
        rows.append({
            "lane": lane,
            "month": month,
            "days": int(len(group)),
            "calls": int(len(calls)),
            "no_calls": int((group["forecast"] == "no-call").sum()),
            "correct": int((calls["status"] == "correct").sum()),
            "partial": int((calls["status"] == "partial").sum()),
            "wrong": int((calls["status"] == "wrong").sum()),
            "exact_accuracy": float((calls["status"] == "correct").mean()) if len(calls) else np.nan,
            "weighted_accuracy": float(calls["score"].mean()) if len(calls) else np.nan,
            "expectancy": float(calls["strategy_return"].mean()) if len(calls) else np.nan,
        })
    return pd.DataFrame(rows)


def _compact_ohlcv(market: pd.DataFrame) -> list[dict[str, object]]:
    output = market.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime("%Y-%m-%d")
    return output.replace({np.nan: None}).to_dict("records")


def _availability(include_deep: bool) -> list[dict[str, object]]:
    import importlib.util

    rows = [
        {"model": "Logistic", "family": "linear", "available": True, "cadence": "daily"},
        {"model": "Random Forest", "family": "tree", "available": True, "cadence": "daily"},
        {"model": "HistGradientBoosting", "family": "tree", "available": True, "cadence": "daily"},
        {"model": "XGBoost", "family": "boosting", "available": importlib.util.find_spec("xgboost") is not None, "cadence": "daily"},
        {"model": "LightGBM", "family": "boosting", "available": importlib.util.find_spec("lightgbm") is not None, "cadence": "daily"},
        {"model": "LSTM", "family": "sequence", "available": include_deep and importlib.util.find_spec("torch") is not None, "cadence": "weekly gated"},
        {"model": "Transformer", "family": "sequence", "available": include_deep and importlib.util.find_spec("torch") is not None, "cadence": "weekly gated"},
        {"model": "Pattern Registry", "family": "rules", "available": True, "cadence": "daily"},
        {"model": "Historical Analog", "family": "nearest-neighbor", "available": True, "cadence": "daily"},
    ]
    return rows


def main() -> None:
    args = parse_args()
    indices = load_indices(
        ROOT / "data" / "newdata" / "index_btc.csv",
        ROOT / "data" / "newdata" / "index_me.csv",
    )
    astro = load_astro(ROOT / "public" / "data" / "astro_scores.json")
    cache_path = ROOT / "data" / "cache" / "BTCUSDT_1d.csv"
    if args.no_refresh:
        market = pd.read_csv(cache_path, parse_dates=["timestamp"])
        provider = "committed cache"
    else:
        market, provider = refresh_daily_market(cache_path)
    frame, groups = build_feature_frame(indices, market, astro)
    analog_columns = [
        "index_BTC", "index_me", "gap_index", "index_btc_change_1", "index_me_change_1",
        "index_btc_slope_3", "index_me_slope_3", "index_corr_5", "composite", "finance",
        "volatility", "astro_composite_change_1", "astro_volatility_z30", "dow_sin", "dow_cos",
    ]

    calendar = run_walk_forward(
        frame,
        groups["calendar"],
        analog_columns,
        "Calendar",
        groups["sequence_calendar"],
        len(groups["sequence_calendar_base"]),
        args.deep,
    )
    full = run_walk_forward(
        frame,
        groups["full"],
        analog_columns + ["market_return_1", "volatility_7", "rsi14", "atr14_pct"],
        "Full Hybrid",
        groups["sequence_full"],
        len(groups["sequence_full_base"]),
        args.deep,
    )
    calendar_preferred = calendar.model_metrics.loc[
        (calendar.model_metrics["status"] == "active")
        & (calendar.model_metrics["expectancy"] > 0)
        & ~calendar.model_metrics["model"].str.contains("Ensemble"),
        "model",
    ].tolist()
    full_preferred = full.model_metrics.loc[
        (full.model_metrics["status"] == "active")
        & (full.model_metrics["expectancy"] > 0)
        & ~full.model_metrics["model"].str.contains("Ensemble"),
        "model",
    ].tolist()
    calendar_future, calendar_selection, calendar_registry = fit_latest_forecasts(
        frame,
        groups["calendar"],
        analog_columns,
        "Calendar",
        groups["sequence_calendar"],
        len(groups["sequence_calendar_base"]),
        args.deep,
        policy_history=calendar.forecasts,
        preferred_models=calendar_preferred,
    )
    full_future, full_selection, full_registry = fit_latest_forecasts(
        frame,
        groups["full"],
        analog_columns + ["market_return_1", "volatility_7", "rsi14", "atr14_pct"],
        "Full Hybrid",
        groups["sequence_full"],
        len(groups["sequence_full_base"]),
        args.deep,
        max_future_days=1,
        policy_history=full.forecasts,
        preferred_models=full_preferred,
    )
    all_metrics = pd.concat([
        calendar.model_metrics.assign(lane="Calendar"),
        full.model_metrics.assign(lane="Full Hybrid"),
    ], ignore_index=True)
    all_metrics = all_metrics.sort_values(["lane", "rank"]).reset_index(drop=True)
    monthly = pd.concat([
        _monthly_metrics(calendar.forecasts, "Calendar"),
        _monthly_metrics(full.forecasts, "Full Hybrid"),
    ], ignore_index=True)
    latest_closed = pd.to_datetime(market["timestamp"]).max().normalize()
    target_accuracy = 0.70
    eligible_metrics = all_metrics[(all_metrics["expectancy"] > 0) & (all_metrics["coverage"] >= 0.8)]
    achieved = eligible_metrics["directional_accuracy"].max() if not eligible_metrics.empty else 0
    explanation_methods = sorted(set(
        calendar.feature_importance.get("method", pd.Series(dtype=str)).dropna().astype(str).tolist()
        + full.feature_importance.get("method", pd.Series(dtype=str)).dropna().astype(str).tolist()
    ))
    payload = {
        "meta": {
            "schema_version": 2,
            "generated_at": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "market_provider": provider,
            "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
            "oos_start": OOS_START.strftime("%Y-%m-%d"),
            "oos_end": latest_closed.strftime("%Y-%m-%d"),
            "index_start": indices["date"].min().strftime("%Y-%m-%d"),
            "index_end": indices["date"].max().strftime("%Y-%m-%d"),
            "target_directional_accuracy": target_accuracy,
            "achieved_directional_accuracy": float(achieved),
            "target_reached": bool(achieved >= target_accuracy),
            "deep_research_enabled": bool(args.deep),
            "scoring": {
                "up": "correct >= +3%; partial +0.1% to < +3%; wrong <= 0%",
                "down": "correct <= -3%; partial > -3% to -0.1%; wrong >= 0%",
                "sideway": "correct within -1% to +1%; otherwise wrong; no partial",
            },
            "availability_assumption": "Index and Astro values for a forecast date are known before that UTC session. Market features are shifted and use only prior closed candles.",
            "validation": {
                "outer": "monthly rolling walk-forward",
                "rolling_train_days": 1460,
                "purge_days": 5,
                "calibration_days": 90,
                "maximum_no_calls_per_month": 6,
                "transaction_cost_bps": 5,
            },
        },
        "market": _compact_ohlcv(market),
        "indices": serialize_frame(indices),
        "forecast": {
            "calendar": serialize_frame(calendar_future),
            "full_hybrid_next_session": serialize_frame(full_future),
            "historical_calendar_oos": serialize_frame(calendar.forecasts),
            "historical_full_hybrid_oos": serialize_frame(full.forecasts),
        },
        "performance": {
            "model_rankings": serialize_frame(all_metrics),
            "monthly": serialize_frame(monthly, date_columns=()),
            "calendar_folds": serialize_frame(calendar.fold_metrics),
            "full_hybrid_folds": serialize_frame(full.fold_metrics),
            "calendar_equity": serialize_frame(equity_curve(calendar.forecasts)),
            "full_hybrid_equity": serialize_frame(equity_curve(full.forecasts)),
            "calendar_reliability": serialize_frame(reliability_bins(calendar.forecasts), date_columns=()),
            "full_hybrid_reliability": serialize_frame(reliability_bins(full.forecasts), date_columns=()),
            "calendar_no_calls": serialize_frame(calendar.no_call_summary, date_columns=()),
            "full_hybrid_no_calls": serialize_frame(full.no_call_summary, date_columns=()),
        },
        "models": {
            "availability": _availability(args.deep),
            "calendar_latest_selection": serialize_frame(calendar_selection, date_columns=()),
            "full_hybrid_latest_selection": serialize_frame(full_selection, date_columns=()),
        },
        "patterns": {
            "calendar": serialize_frame(calendar_registry),
            "full_hybrid": serialize_frame(full_registry),
        },
        "explainability": {
            "method": "; ".join(explanation_methods) or "Explainability unavailable; model remains gated.",
            "calendar": serialize_frame(calendar.feature_importance, date_columns=()),
            "full_hybrid": serialize_frame(full.feature_importance, date_columns=()),
        },
        "research": {
            "correlation_heatmap": serialize_frame(feature_heatmap(frame, groups), date_columns=()),
            "feature_groups": {key: value for key, value in groups.items() if not key.endswith("_base") and not key.startswith("sequence")},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    args.output.write_text(content, encoding="utf-8")
    mirror = ROOT / "data" / "hybrid_research.json"
    mirror.write_text(content, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "bytes": len(content),
        "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
        "calendar_oos_rows": len(calendar.forecasts),
        "full_hybrid_oos_rows": len(full.forecasts),
        "future_rows": len(calendar_future),
        "achieved_directional_accuracy": round(float(achieved), 4),
        "target_reached": bool(achieved >= target_accuracy),
    }, indent=2))


if __name__ == "__main__":
    main()
