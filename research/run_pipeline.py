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
    FORECAST_LEAD_DAYS,
    MAX_NO_CALL_PER_MONTH,
    MAX_SIDEWAY_PER_MONTH,
    OOS_START,
    ROUND_TRIP_COST,
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
from research.data_sources import (
    build_intraday_daily_features,
    load_external_features,
    refresh_intraday_market,
    source_lineage,
)
from research.evaluation import (
    class_drift_rows,
    class_metric_rows,
    confidence_risk_rows,
    confusion_rows,
    feature_drift_rows,
    grouped_performance_rows,
    page_hinkley_alarm,
)
from research.learning import (
    append_official_forecast,
    apply_live_model_ranking,
    apply_live_pattern_ranking,
    grade_event_state,
    grade_learning_state,
    learning_summary,
    load_learning_state,
    record_source_revision,
    record_selection_snapshot,
    serialize_learning_state,
)
from research.model_candidates import model_availability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe hybrid BTC research artifacts")
    parser.add_argument("--deep", action="store_true", help="Evaluate LSTM and Transformer candidates")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reuse the latest verified OOS research and run only daily grade/rerank/publish work",
    )
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


def _capacity_history(
    forecasts: pd.DataFrame,
    state: dict[str, object],
    lane: str,
) -> pd.DataFrame:
    """Overlay immutable official calls on OOS history for monthly policy capacity."""
    base = forecasts[["date", "forecast"]].copy() if not forecasts.empty else pd.DataFrame(columns=["date", "forecast"])
    official_rows = []
    for row in state.get("forecasts", []):
        if not isinstance(row, dict) or row.get("lane") != lane:
            continue
        date = row.get("date") or row.get("target_date")
        if date and row.get("forecast"):
            official_rows.append({"date": date, "forecast": row["forecast"]})
    official = pd.DataFrame(official_rows, columns=["date", "forecast"])
    combined = pd.concat([base, official], ignore_index=True)
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    return combined.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _compact_ohlcv(market: pd.DataFrame) -> list[dict[str, object]]:
    output = market.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime("%Y-%m-%d")
    return output.replace({np.nan: None}).to_dict("records")


def _pick_fields(rows: list[dict[str, object]], fields: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {field: row.get(field) for field in fields if field in row}
        for row in rows
        if isinstance(row, dict)
    ]


def _core_payload(payload: dict[str, object]) -> dict[str, object]:
    """Build the fast initial-load artifact; detailed research remains lazy-loaded."""
    core = dict(payload)
    latest_closed = str(payload.get("meta", {}).get("latest_closed_utc", "9999-12-31"))
    core["market"] = [
        row for row in payload.get("market", [])
        if isinstance(row, dict) and str(row.get("timestamp", "")) >= OOS_START.strftime("%Y-%m-%d")
    ]
    core["indices"] = [
        row for row in payload.get("indices", [])
        if isinstance(row, dict) and str(row.get("date", "")) <= latest_closed
    ]
    forecast_fields = (
        "date", "lane", "model", "fold", "forecast", "status", "score", "daily_return",
        "strategy_return", "confidence", "prob_down", "prob_sideway", "prob_up", "expected_score",
        "decision_margin", "entropy", "top_pattern",
        "policy_mode", "sideway_penalty", "sideway_cap_override", "contract_version",
        "information_cutoff_utc", "target_start_utc", "target_end_utc", "trade_action",
        "trade_eligible", "trade_gate_reason", "expected_net_return", "expectancy_lcb",
        "execution_model", "execution_model_forecast",
    )
    forecast = dict(payload.get("forecast", {}))
    forecast["calendar"] = _pick_fields(list(forecast.get("calendar", []))[:45], forecast_fields)
    forecast["historical_calendar_oos"] = []
    for key in ("historical_full_hybrid_oos",):
        forecast[key] = _pick_fields(list(forecast.get(key, [])), forecast_fields)
    core["forecast"] = forecast
    performance = dict(payload.get("performance", {}))
    for key in ("calendar_folds", "full_hybrid_folds"):
        performance[key] = list(performance.get(key, []))[-12:]
    for key in ("calendar_equity", "full_hybrid_equity"):
        performance[key] = list(performance.get(key, []))[-365:]
    core["performance"] = performance
    patterns = dict(payload.get("patterns", {}))
    patterns["calendar"] = list(patterns.get("calendar", []))[:32]
    patterns["full_hybrid"] = list(patterns.get("full_hybrid", []))[:32]
    core["patterns"] = patterns
    learning = dict(payload.get("learning", {}))
    official_fields = forecast_fields + (
        "forecast_id", "target_date", "issued_at", "closed_through_at_issue", "actual_return",
        "evaluated_at", "immutable_digest", "executed_strategy_return",
    )
    learning["official_forecast_ledger"] = _pick_fields(
        list(learning.get("official_forecast_ledger", [])),
        official_fields,
    )
    learning["event_evaluation_ledger"] = []
    learning["selection_history"] = list(learning.get("selection_history", []))[-7:]
    learning["source_revisions"] = list(learning.get("source_revisions", []))[-20:]
    core["learning"] = learning
    return core


def _availability(include_deep: bool) -> list[dict[str, object]]:
    import importlib.util

    rows = model_availability_rows() + [
        {"model": "LSTM", "family": "sequence", "available": include_deep and importlib.util.find_spec("torch") is not None, "cadence": "weekly gated"},
        {"model": "Transformer", "family": "sequence", "available": include_deep and importlib.util.find_spec("torch") is not None, "cadence": "weekly gated"},
        {"model": "Pattern Registry", "family": "rules", "available": True, "cadence": "daily"},
        {"model": "Historical Analog", "family": "nearest-neighbor", "available": True, "cadence": "daily"},
    ]
    return rows


def _read_previous_artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _restore_frame(
    rows: object,
    date_columns: tuple[str, ...] = ("date",),
) -> pd.DataFrame:
    frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return frame


def _restore_backtest(payload: dict[str, object], lane: str) -> object | None:
    """Hydrate the immutable weekly OOS result for a fast daily publication run."""
    from research.hybrid_core import BacktestResult

    meta = payload.get("meta", {})
    validation = meta.get("validation", {}) if isinstance(meta, dict) else {}
    if not isinstance(validation, dict) or (
        validation.get("maximum_sideway_calls_per_month") != MAX_SIDEWAY_PER_MONTH
        or validation.get("maximum_no_calls_per_month") != MAX_NO_CALL_PER_MONTH
    ):
        return None
    forecast_payload = payload.get("forecast", {})
    performance = payload.get("performance", {})
    patterns = payload.get("patterns", {})
    explainability = payload.get("explainability", {})
    if not all(isinstance(item, dict) for item in (forecast_payload, performance, patterns, explainability)):
        return None
    calendar_lane = lane == "Calendar"
    forecast_key = "historical_calendar_oos" if calendar_lane else "historical_full_hybrid_oos"
    fold_key = "calendar_folds" if calendar_lane else "full_hybrid_folds"
    no_call_key = "calendar_no_calls" if calendar_lane else "full_hybrid_no_calls"
    registry_key = "calendar" if calendar_lane else "full_hybrid"
    forecasts = _restore_frame(forecast_payload.get(forecast_key))
    rankings = _restore_frame(performance.get("model_rankings"), ())
    if "lane" in rankings:
        rankings = rankings[rankings["lane"] == lane].drop(columns="lane").reset_index(drop=True)
    folds = _restore_frame(
        performance.get(fold_key),
        ("train_start", "train_end", "calibration_fit_end", "policy_start", "calibration_end", "test_start", "test_end"),
    )
    importance = _restore_frame(explainability.get(registry_key), ())
    registry = _restore_frame(patterns.get(registry_key), ("last_seen",))
    no_calls = _restore_frame(performance.get(no_call_key), ())
    if forecasts.empty or rankings.empty:
        return None
    # Candidate-level rows are not needed during fast publication; weekly OOS
    # research remains authoritative for metrics and ranking priors.
    return BacktestResult(forecasts, pd.DataFrame(), rankings, folds, importance, registry, no_calls)


def _bootstrap_published_forecasts(state: dict[str, object], payload: dict[str, object]) -> int:
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    forecast_payload = payload.get("forecast", {}) if isinstance(payload.get("forecast"), dict) else {}
    model_payload = payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}
    issued_at = str(meta.get("generated_at", ""))
    closed_through = str(meta.get("latest_closed_utc", ""))
    if not issued_at or not closed_through:
        return 0
    lane_sources = [
        ("Calendar", "calendar", "calendar_latest_selection"),
        ("Full Hybrid", "full_hybrid_next_session", "full_hybrid_latest_selection"),
    ]
    added = 0
    for lane, forecast_key, selection_key in lane_sources:
        rows = forecast_payload.get(forecast_key, [])
        if not isinstance(rows, list):
            continue
        candidates = sorted(
            (row for row in rows if isinstance(row, dict) and str(row.get("date", "")) > closed_through),
            key=lambda row: str(row.get("date", "")),
        )
        if not candidates:
            continue
        selection = model_payload.get(selection_key, [])
        legacy_forecast = dict(candidates[0])
        legacy_forecast.setdefault("contract_version", 1)
        added += int(append_official_forecast(
            state,
            legacy_forecast,
            lane,
            issued_at,
            closed_through,
            selection if isinstance(selection, list) else [],
        ))
    return added


def _enrich_selection(selection: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or metrics.empty:
        return selection
    columns = [
        "model", "rank", "live_samples", "live_weighted_accuracy", "live_directional_accuracy",
        "live_expectancy", "adjusted_weighted_accuracy", "adaptive_rank_score", "selection_change",
        "replacement_reason", "expectancy", "expectancy_lcb", "trade_eligible", "status",
    ]
    available = [column for column in columns if column in metrics]
    metric_view = metrics[available].rename(columns={"status": "production_status"})
    return selection.merge(metric_view, on="model", how="left")


def _apply_trade_gate(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the daily forecast, but execute only robust positive-expectancy signals."""
    if forecasts.empty:
        return forecasts
    output = forecasts.copy()
    eligible = metrics.copy()
    if not eligible.empty:
        ensemble_mask = eligible["model"].astype(str).str.contains("Ensemble", case=False, na=False)
        eligible = eligible[
            ~ensemble_mask
            & (eligible.get("status", "standby") == "active")
            & (pd.to_numeric(eligible.get("expectancy_lcb"), errors="coerce") > 0)
        ].sort_values("rank")
    champion = eligible.iloc[0] if not eligible.empty else None
    champion_name = str(champion["model"]) if champion is not None else None
    champion_selection = (
        selection[selection["model"] == champion_name].iloc[0]
        if (
            champion_name is not None
            and not selection.empty
            and "model" in selection
            and champion_name in set(selection["model"])
        )
        else None
    )
    champion_direction = (
        str(champion_selection.get("next_forecast")) if champion_selection is not None else None
    )
    for index, row in output.iterrows():
        direction = str(row.get("forecast"))
        direction_agrees = champion_direction == direction
        executable = champion is not None and direction in {"up", "down"} and direction_agrees
        output.at[index, "trade_action"] = direction if executable else "flat"
        output.at[index, "trade_eligible"] = bool(executable)
        output.at[index, "execution_model"] = champion_name
        output.at[index, "execution_model_forecast"] = champion_direction
        output.at[index, "expected_net_return"] = (
            float(champion["expectancy"]) if executable else np.nan
        )
        output.at[index, "expectancy_lcb"] = (
            float(champion["expectancy_lcb"]) if executable else np.nan
        )
        if champion is None:
            reason = "FLAT: no candidate has a positive out-of-sample net-expectancy lower bound"
        elif direction == "sideway":
            reason = "FLAT: SIDEWAY is an outlook class, not a directional trade"
        elif not direction_agrees:
            reason = f"FLAT: {champion_name} does not confirm the ensemble direction"
        else:
            reason = f"TRADE: {champion_name} confirms direction and passed the after-cost expectancy gate"
        output.at[index, "trade_gate_reason"] = reason
    return output


def main() -> None:
    args = parse_args()
    run_at = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    previous_payload = _read_previous_artifact(args.output)
    learning_state_path = ROOT / "data" / "learning_state.json"
    learning_state = load_learning_state(learning_state_path)
    bootstrapped_forecasts = _bootstrap_published_forecasts(learning_state, previous_payload)
    index_btc_path = ROOT / "data" / "newdata" / "index_btc.csv"
    index_me_path = ROOT / "data" / "newdata" / "index_me.csv"
    astro_path = ROOT / "public" / "data" / "astro_scores.json"
    indices = load_indices(index_btc_path, index_me_path)
    astro = load_astro(astro_path)
    cache_path = ROOT / "data" / "cache" / "BTCUSDT_1d.csv"
    if args.no_refresh:
        market = pd.read_csv(cache_path, parse_dates=["timestamp"])
        provider = "committed cache"
        expected_closed = utc_now().tz_localize(None).normalize() - pd.Timedelta(days=1)
        cached_latest = pd.to_datetime(market["timestamp"]).max().normalize()
        market_health = {
            "status": "healthy" if cached_latest >= expected_closed else "stale",
            "expected_closed_utc": expected_closed.strftime("%Y-%m-%d"),
            "actual_closed_utc": cached_latest.strftime("%Y-%m-%d"),
            "cache_latest_before_refresh": cached_latest.strftime("%Y-%m-%d"),
            "selected_provider": provider,
            "provider_count": 0,
            "cross_exchange_close_discrepancy_bps": None,
            "stale": bool(cached_latest < expected_closed),
            "attempts": [],
        }
    else:
        market, provider, market_health = refresh_daily_market(cache_path, include_health=True)
    intraday_frames: list[tuple[str, pd.DataFrame]] = []
    intraday_health: list[dict[str, object]] = []
    intraday_paths: dict[str, Path] = {}
    for timeframe in ("1h", "4h"):
        intraday_path = ROOT / "data" / "cache" / f"BTCUSDT_{timeframe}.csv"
        intraday_paths[timeframe] = intraday_path
        try:
            if args.no_refresh:
                intraday_frame = pd.read_csv(intraday_path, parse_dates=["timestamp"])
                latest_bar = pd.to_datetime(intraday_frame["timestamp"]).max()
                health = {
                    "timeframe": timeframe,
                    "status": "committed-cache",
                    "provider": "committed cache",
                    "expected_open_utc": None,
                    "actual_open_utc": latest_bar.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "stale": None,
                    "rows": int(len(intraday_frame)),
                    "attempts": [],
                }
            else:
                intraday_frame, health = refresh_intraday_market(intraday_path, timeframe)
            intraday_frames.append((timeframe, intraday_frame))
            intraday_health.append(health)
        except Exception as exc:
            intraday_health.append({
                "timeframe": timeframe,
                "status": "unavailable",
                "provider": None,
                "expected_open_utc": None,
                "actual_open_utc": None,
                "stale": True,
                "rows": 0,
                "attempts": [],
                "error": f"{type(exc).__name__}: {exc}",
            })
    intraday_features = build_intraday_daily_features(intraday_frames)
    external, external_health, external_lineage = load_external_features(ROOT / "data" / "external")
    latest_closed = pd.to_datetime(market["timestamp"]).max().normalize()
    evaluated_forecasts = grade_learning_state(learning_state, market, latest_closed, run_at)
    frame, groups = build_feature_frame(indices, market, astro, intraday_features, external)
    analog_columns = [
        "index_BTC", "index_me", "gap_index", "index_btc_change_1", "index_me_change_1",
        "index_btc_slope_3", "index_me_slope_3", "index_corr_5", "composite", "finance",
        "volatility", "astro_composite_change_1", "astro_volatility_z30", "dow_sin", "dow_cos",
    ]

    calendar = _restore_backtest(previous_payload, "Calendar") if args.fast else None
    full = _restore_backtest(previous_payload, "Full Hybrid") if args.fast else None
    pipeline_mode = "fast-daily" if calendar is not None and full is not None else "full-research"
    if calendar is None or full is None:
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
    performance_drift = {
        "calendar": page_hinkley_alarm(1 - calendar.forecasts["score"].dropna().to_numpy(dtype=float)),
        "full_hybrid": page_hinkley_alarm(1 - full.forecasts["score"].dropna().to_numpy(dtype=float)),
    }
    calendar.model_metrics = apply_live_model_ranking(
        calendar.model_metrics,
        learning_state,
        "Calendar",
        as_of_closed=latest_closed,
        execution_suspended=bool(performance_drift["calendar"]["alarm"]),
    )
    full.model_metrics = apply_live_model_ranking(
        full.model_metrics,
        learning_state,
        "Full Hybrid",
        as_of_closed=latest_closed,
        execution_suspended=bool(performance_drift["full_hybrid"]["alarm"]),
    )
    calendar_preferred = calendar.model_metrics.loc[
        (calendar.model_metrics["status"] == "active")
        & ~calendar.model_metrics["model"].str.contains("Ensemble"),
        "model",
    ].tolist()
    full_preferred = full.model_metrics.loc[
        (full.model_metrics["status"] == "active")
        & ~full.model_metrics["model"].str.contains("Ensemble"),
        "model",
    ].tolist()
    calendar_capacity = _capacity_history(calendar.forecasts, learning_state, "Calendar")
    full_capacity = _capacity_history(full.forecasts, learning_state, "Full Hybrid")
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
        capacity_histories=[full_capacity],
        reserved_histories=[full_capacity],
        preferred_models=full_preferred,
        pattern_adjuster=lambda registry: apply_live_pattern_ranking(
            registry, learning_state, "Full Hybrid", as_of_closed=latest_closed,
        ),
    )
    fusion_capacity = full_capacity
    if not full_future.empty:
        fusion_capacity = (
            pd.concat([full_capacity, full_future[["date", "forecast"]]], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    calendar_future, calendar_selection, calendar_registry = fit_latest_forecasts(
        frame,
        groups["calendar"],
        analog_columns,
        "Calendar",
        groups["sequence_calendar"],
        len(groups["sequence_calendar_base"]),
        args.deep,
        policy_history=calendar.forecasts,
        capacity_histories=[calendar_capacity, fusion_capacity],
        reserved_histories=[calendar_capacity, full_capacity],
        preferred_models=calendar_preferred,
        pattern_adjuster=lambda registry: apply_live_pattern_ranking(
            registry, learning_state, "Calendar", as_of_closed=latest_closed,
        ),
    )
    calendar_selection = _enrich_selection(calendar_selection, calendar.model_metrics)
    full_selection = _enrich_selection(full_selection, full.model_metrics)
    calendar_future = _apply_trade_gate(calendar_future, calendar.model_metrics, calendar_selection)
    full_future = _apply_trade_gate(full_future, full.model_metrics, full_selection)
    if not calendar_future.empty:
        append_official_forecast(
            learning_state,
            serialize_frame(calendar_future.head(1))[0],
            "Calendar",
            run_at,
            latest_closed.strftime("%Y-%m-%d"),
            serialize_frame(calendar_selection, date_columns=()),
        )
    if not full_future.empty:
        append_official_forecast(
            learning_state,
            serialize_frame(full_future.head(1))[0],
            "Full Hybrid",
            run_at,
            latest_closed.strftime("%Y-%m-%d"),
            serialize_frame(full_selection, date_columns=()),
        )
    evaluated_events = grade_event_state(learning_state, market, latest_closed, run_at)
    record_selection_snapshot(
        learning_state,
        latest_closed.strftime("%Y-%m-%d"),
        run_at,
        {"Calendar": calendar.model_metrics, "Full Hybrid": full.model_metrics},
        {"Calendar": calendar_registry, "Full Hybrid": full_registry},
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
    class_metrics = pd.concat([
        class_metric_rows(calendar.forecasts, "Calendar"),
        class_metric_rows(full.forecasts, "Full Hybrid"),
    ], ignore_index=True)
    confusion = pd.concat([
        confusion_rows(calendar.forecasts, "Calendar"),
        confusion_rows(full.forecasts, "Full Hybrid"),
    ], ignore_index=True)
    confidence_risk = pd.concat([
        confidence_risk_rows(calendar.forecasts, "Calendar"),
        confidence_risk_rows(full.forecasts, "Full Hybrid"),
    ], ignore_index=True)
    grouped_performance = pd.concat([
        grouped_performance_rows(calendar.forecasts, frame, "Calendar"),
        grouped_performance_rows(full.forecasts, frame, "Full Hybrid"),
    ], ignore_index=True)
    feature_drift = feature_drift_rows(frame, groups["full"])
    class_drift = class_drift_rows(frame)
    target_accuracy = 0.70
    eligible_metrics = all_metrics[(all_metrics["expectancy"] > 0) & (all_metrics["coverage"] >= 0.8)]
    achieved = eligible_metrics["directional_accuracy"].max() if not eligible_metrics.empty else 0
    oos_end = max(calendar.forecasts["date"].max(), full.forecasts["date"].max())
    explanation_methods = sorted(set(
        calendar.feature_importance.get("method", pd.Series(dtype=str)).dropna().astype(str).tolist()
        + full.feature_importance.get("method", pd.Series(dtype=str)).dropna().astype(str).tolist()
    ))
    data_lineage = [
        source_lineage(
            index_btc_path,
            "private-index-btc",
            len(indices),
            indices["date"].min(),
            indices["date"].max(),
            int((~indices["index_BTC_availability_imputed"]).sum()),
        ),
        source_lineage(
            index_me_path,
            "private-index-me",
            len(indices),
            indices["date"].min(),
            indices["date"].max(),
            int((~indices["index_me_availability_imputed"]).sum()),
        ),
        source_lineage(
            astro_path,
            "astro-scores",
            len(astro),
            astro["date"].min(),
            astro["date"].max(),
            int((~astro["astro_availability_imputed"]).sum()),
        ),
        source_lineage(
            cache_path,
            f"btc-daily-{provider}",
            len(market),
            market["timestamp"].min(),
            market["timestamp"].max(),
            availability_mode="closed-bar-enforced",
        ),
    ]
    for timeframe, intraday_frame in intraday_frames:
        path = intraday_paths[timeframe]
        if path.exists():
            data_lineage.append(source_lineage(
                path,
                f"btc-{timeframe}",
                len(intraday_frame),
                intraday_frame["timestamp"].min(),
                intraday_frame["timestamp"].max(),
                availability_mode="closed-bar-enforced",
            ))
    data_lineage.extend(external_lineage)
    private_lineage = [
        item for item in data_lineage
        if str(item.get("source", "")).startswith("private-") or item.get("source") == "astro-scores"
    ]
    explicit_rows = sum(int(item.get("explicit_availability_rows", 0)) for item in private_lineage)
    private_rows = sum(int(item.get("rows", 0)) for item in private_lineage)
    point_in_time_coverage = explicit_rows / private_rows if private_rows else 0.0
    provenance_warnings = [
        f"{item['source']} has no complete available_at history; historical results assume prepublication"
        for item in private_lineage
        if item.get("availability_mode") != "explicit"
    ]
    record_source_revision(learning_state, data_lineage, run_at)
    information_cutoff = (latest_closed + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    first_target = (latest_closed + pd.Timedelta(days=FORECAST_LEAD_DAYS)).strftime("%Y-%m-%dT00:00:00Z")
    payload = {
        "meta": {
            "schema_version": 5,
            "generated_at": run_at,
            "market_provider": provider,
            "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
            "pipeline_mode": pipeline_mode,
            "forecast_cutoff_utc": information_cutoff,
            "first_publishable_target_utc": first_target,
            "oos_start": OOS_START.strftime("%Y-%m-%d"),
            "oos_end": pd.Timestamp(oos_end).strftime("%Y-%m-%d"),
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
            "availability_assumption": "Explicit available_at timestamps are enforced before the target UTC session. Sources without timestamps are marked prepublished-imputed; market and intraday features lag two sessions so the daily artifact is fully available before its target opens.",
            "data_lineage": data_lineage,
            "provenance": {
                "status": "verified" if not provenance_warnings else "research-only",
                "private_point_in_time_coverage": point_in_time_coverage,
                "warnings": provenance_warnings,
                "revision_digest": learning_state.get("source_revisions", [{}])[-1].get("digest"),
            },
            "validation": {
                "outer": "monthly rolling walk-forward",
                "rolling_train_days": 1460,
                "purge_days": 7,
                "calibration_days": 180,
                "calibration_partition": "first 67% calibrator fit; final 33% policy and ensemble selection",
                "calibration_methods": "identity, sigmoid, temperature, isotonic when sample-gated",
                "stacking": "non-negative simplex weights learned only from pre-test OOS policy validation probabilities",
                "maximum_no_calls_per_month": MAX_NO_CALL_PER_MONTH,
                "maximum_sideway_calls_per_month": MAX_SIDEWAY_PER_MONTH,
                "one_way_transaction_cost_bps": int(ROUND_TRIP_COST * 10_000 / 2),
                "round_trip_transaction_cost_bps": int(ROUND_TRIP_COST * 10_000),
                "daily_evaluation_utc": "03:20",
                "official_forecasts_are_immutable": True,
                "official_forecast_contract": "issued before target UTC open with a two-session lead",
                "trade_gate": "directional execution requires positive OOS net-expectancy lower bound after costs",
                "daily_and_event_grades_are_independent": True,
                "minimum_live_grades_for_promotion": 20,
                "production_promotion_cadence": "monthly",
                "oos_refit_cadence": "weekly full research; daily grade/rerank/publish reuses the latest verified OOS artifact",
                "drift_action": "recent Page-Hinkley deterioration suspends execution but preserves forecasts",
                "pattern_lead_search_days": 3,
                "pattern_shape_duration_days": "1-6",
            },
        },
        "health": {
            "market": market_health,
            "intraday": intraday_health,
            "external": external_health,
            "last_evaluation": {
                "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
                "evaluated_forecasts_this_run": int(evaluated_forecasts),
                "run_at": run_at,
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
            "class_metrics": serialize_frame(class_metrics, date_columns=()),
            "confusion_matrix": serialize_frame(confusion, date_columns=()),
            "confidence_risk": serialize_frame(confidence_risk, date_columns=()),
            "grouped": serialize_frame(grouped_performance, date_columns=()),
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
            "drift": {
                "features": serialize_frame(feature_drift, date_columns=()),
                "classes": serialize_frame(class_drift, date_columns=()),
                "performance": performance_drift,
            },
            "event_definitions": {
                "window_days": 3,
                "pivot_confirmation_days": 5,
                "pump": "1D >= +4%, 3D >= +8%, or 5D >= +12%",
                "dump": "1D <= -4%, 3D <= -8%, or 5D <= -12%",
                "sideway_cluster": "three-day high-low range <= 2%",
                "daily_grade_is_independent": True,
            },
        },
        "learning": {
            "summary": learning_summary(learning_state),
            "official_forecast_ledger": learning_state.get("forecasts", []),
            "event_evaluation_ledger": learning_state.get("event_evaluations", []),
            "selection_history": learning_state.get("selection_history", []),
            "source_revisions": learning_state.get("source_revisions", []),
            "evaluated_this_run": evaluated_forecasts,
            "events_evaluated_this_run": evaluated_events,
            "bootstrapped_this_run": bootstrapped_forecasts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    args.output.write_text(content, encoding="utf-8")
    mirror = ROOT / "data" / "hybrid_research.json"
    mirror.write_text(content, encoding="utf-8")
    core_content = json.dumps(_core_payload(payload), ensure_ascii=True, separators=(",", ":"))
    core_output = args.output.with_name("hybrid_research_core.json")
    core_output.write_text(core_content, encoding="utf-8")
    (ROOT / "data" / "hybrid_research_core.json").write_text(core_content, encoding="utf-8")
    learning_state_path.write_text(serialize_learning_state(learning_state), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "bytes": len(content),
        "core_bytes": len(core_content),
        "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
        "calendar_oos_rows": len(calendar.forecasts),
        "full_hybrid_oos_rows": len(full.forecasts),
        "future_rows": len(calendar_future),
        "achieved_directional_accuracy": round(float(achieved), 4),
        "target_reached": bool(achieved >= target_accuracy),
        "pipeline_mode": pipeline_mode,
        "evaluated_forecasts": evaluated_forecasts,
        "evaluated_events": evaluated_events,
        "official_ledger_rows": len(learning_state.get("forecasts", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
