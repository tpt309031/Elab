from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from quant_core import (
    analysis_tables,
    btc_move_research_summary,
    build_advanced_daily_features,
    build_dataset,
    build_pattern_registry,
    evaluate_registry_forecast_ledger,
    load_ohlcv,
    pattern_signals,
    read_private_indices_csv,
    registry_future_forecasts,
    scenario_accuracy_summary,
    scenario_pattern_forecasts,
    update_scenario_ledger,
    walk_forward_models,
    write_daily_learning_state,
)


PUBLIC_DATA = ROOT / "data" / "dashboard.json"
ROOT_DATA = ROOT / "data" / "dashboard.json"
SOURCE_NAME = "newdata CSV 2024-2028"
SOURCE_SLUG = "newdata_csv"


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, date_columns: tuple[str, ...] = ("date", "timestamp", "as_of")) -> list[dict]:
    if frame is None or frame.empty:
        return []
    output = frame.copy()
    for column in date_columns:
        if column in output:
            output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in output.replace({np.nan: None}).to_dict("records")
    ]


def main() -> None:
    indices = read_private_indices_csv(
        ROOT / "data" / "newdata" / "index_btc.csv",
        ROOT / "data" / "newdata" / "index_me.csv",
    )
    start = indices["date"].min() - pd.Timedelta(days=70)
    try:
        daily_ohlcv, provider = load_ohlcv("1d", start, refresh=True, cache_dir=ROOT / "data" / "cache")
    except Exception:
        daily_ohlcv, provider = load_ohlcv("1d", start, refresh=False, cache_dir=ROOT / "data" / "cache")

    dataset = build_dataset(indices, daily_ohlcv, "1d")
    tables = analysis_tables(dataset)
    daily_signals, turning_patterns, _ = pattern_signals(dataset)
    scenario_forecasts = scenario_pattern_forecasts(indices, dataset)
    scenario_ledger = update_scenario_ledger(
        scenario_forecasts,
        daily_ohlcv,
        SOURCE_NAME,
        ROOT / "data" / "scenario_ledger.csv",
    )
    registry_path = ROOT / "data" / "registry_forecast_ledger.csv"
    existing_registry_ledger = (
        pd.read_csv(registry_path, parse_dates=["date"]) if registry_path.exists() else pd.DataFrame()
    )
    existing_registry_ledger = evaluate_registry_forecast_ledger(
        existing_registry_ledger, daily_ohlcv, SOURCE_NAME, registry_path
    )
    pattern_registry, _ = build_pattern_registry(
        indices,
        daily_ohlcv,
        SOURCE_NAME,
        ROOT / "data" / "registries" / f"{SOURCE_SLUG}.csv",
        forecast_ledger=existing_registry_ledger,
        active_per_direction=8,
    )
    registry_ledger = registry_future_forecasts(
        indices,
        daily_ohlcv["timestamp"].max().normalize(),
        pattern_registry,
        SOURCE_NAME,
        registry_path,
    )
    registry_ledger = evaluate_registry_forecast_ledger(registry_ledger, daily_ohlcv, SOURCE_NAME, registry_path)
    advanced_daily = build_advanced_daily_features(indices, daily_ohlcv)
    move_research = btc_move_research_summary(advanced_daily)
    ledger_accuracy = scenario_accuracy_summary(scenario_ledger)
    try:
        background_model = walk_forward_models(dataset, "target_pivot_high_next_N")
        background_metrics = background_model.metrics
    except Exception:
        background_metrics = pd.DataFrame()
    learning_state = write_daily_learning_state(
        ROOT / "data" / "learning_state.json",
        pattern_registry,
        scenario_ledger,
        registry_ledger,
        background_metrics,
    )

    enriched_ohlcv = dataset[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "index_BTC",
            "index_me",
            "pivot_high",
            "pivot_low",
            "gap_index",
        ]
    ].copy()
    enriched_ohlcv["timestamp"] = pd.to_datetime(enriched_ohlcv["timestamp"]).dt.normalize()

    scenario_view = scenario_forecasts.copy()
    if not scenario_ledger.empty:
        ledger_columns = [
            "date",
            "forecast",
            "confidence",
            "matches",
            "mean_similarity",
            "trap_warning",
            "similar_dates",
            "actual",
            "status",
            "actual_return_3d",
            "actual_range_2d",
        ]
        scenario_view = pd.concat([scenario_view, scenario_ledger[ledger_columns]], ignore_index=True)
        scenario_view = scenario_view.drop_duplicates("date", keep="last").sort_values("date")

    payload = {
        "meta": {
            "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": SOURCE_NAME,
            "ohlcv_provider": provider,
            "latest_market_date": pd.to_datetime(daily_ohlcv["timestamp"]).max().strftime("%Y-%m-%d"),
            "index_start": indices["date"].min().strftime("%Y-%m-%d"),
            "index_end": indices["date"].max().strftime("%Y-%m-%d"),
            "index_records": int(len(indices)),
            "ohlcv_records": int(len(daily_ohlcv)),
            "learning_state": learning_state,
        },
        "indices": _records(indices),
        "ohlcv": _records(enriched_ohlcv),
        "scenario_forecasts": _records(scenario_view),
        "scenario_ledger": _records(scenario_ledger),
        "registry_ledger": _records(registry_ledger),
        "active_patterns": _records(
            pattern_registry[pattern_registry["status"] == "active"]
            .sort_values(["direction", "rank"])
            [["direction", "rank", "pattern", "occurrences", "hit_rate", "acceptable_rate", "is_new"]]
        ),
        "unused_patterns": _records(
            pattern_registry[pattern_registry["status"] != "active"]
            .sort_values(["score", "occurrences"], ascending=False)
            .head(16)[["direction", "rank", "pattern", "occurrences", "hit_rate", "acceptable_rate", "status"]]
        ),
        "turning_patterns": _records(turning_patterns),
        "daily_signals": _records(daily_signals),
        "tables": {name: _records(frame) for name, frame in tables.items()},
        "move_research": _records(move_research),
        "ledger_accuracy": _records(ledger_accuracy),
    }

    PUBLIC_DATA.parent.mkdir(parents=True, exist_ok=True)
    ROOT_DATA.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    PUBLIC_DATA.write_text(content, encoding="utf-8")
    ROOT_DATA.write_text(content, encoding="utf-8")
    print(f"Wrote {PUBLIC_DATA} ({PUBLIC_DATA.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {ROOT_DATA} ({ROOT_DATA.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
