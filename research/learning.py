from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.hybrid_core import SIDEWAY_LIMIT, TRADING_COST, grade_forecast


STATE_SCHEMA_VERSION = 2
MODEL_PRIOR_STRENGTH = 30.0
PATTERN_PRIOR_STRENGTH = 12.0
MAX_SELECTION_HISTORY = 730


def empty_learning_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at": None,
        "forecasts": [],
        "selection_history": [],
    }


def load_learning_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return empty_learning_state()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load production learning ledger: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Production learning ledger must be a JSON object")
    source_schema = int(payload.get("schema_version", 1))
    if source_schema < 1 or source_schema > STATE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported learning ledger schema: {source_schema}")
    if not isinstance(payload.get("forecasts", []), list) or not isinstance(payload.get("selection_history", []), list):
        raise RuntimeError("Production learning ledger has invalid collection fields")
    state = empty_learning_state()
    state.update(payload)
    state["forecasts"] = list(payload.get("forecasts", []))
    state["selection_history"] = list(payload.get("selection_history", []))[-MAX_SELECTION_HISTORY:]
    for row in state["forecasts"]:
        if not isinstance(row, dict):
            raise RuntimeError("Production learning ledger contains a non-object forecast")
        expected_digest = official_forecast_digest(row)
        stored_digest = row.get("immutable_digest")
        if source_schema >= 2 and stored_digest != expected_digest:
            raise RuntimeError(f"Published forecast integrity check failed: {row.get('forecast_id')}")
        row["immutable_digest"] = expected_digest
    state["schema_version"] = STATE_SCHEMA_VERSION
    return state


def serialize_learning_state(state: dict[str, Any]) -> str:
    state["schema_version"] = STATE_SCHEMA_VERSION
    return json.dumps(state, ensure_ascii=True, separators=(",", ":"))


def _published_model_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "model", "forecast", "prob_down", "prob_sideway", "prob_up", "source",
        "selection_status", "calibration_score",
    )
    return {field: prediction.get(field) for field in fields}


def _published_pattern_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = "pattern_id", "pattern", "direction", "rank_at_issue"
    return {field: prediction.get(field) for field in fields}


def _pattern_predictions(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = forecast.get("pattern_predictions")
    if isinstance(predictions, list) and predictions:
        return [item for item in predictions if isinstance(item, dict)]
    legacy = forecast.get("pattern_evaluation")
    return [legacy] if isinstance(legacy, dict) else []


def official_forecast_digest(forecast: dict[str, Any]) -> str:
    fields = (
        "forecast_id", "date", "target_date", "lane", "issued_at", "closed_through_at_issue",
        "forecast", "confidence", "prob_down", "prob_sideway", "prob_up", "expected_score",
        "decision_margin", "entropy", "policy_mode", "sideway_penalty", "model_members",
        "model_weights", "top_pattern",
    )
    published = {field: forecast.get(field) for field in fields}
    published["model_predictions"] = [
        _published_model_prediction(item)
        for item in forecast.get("model_predictions", [])
        if isinstance(item, dict)
    ]
    published["pattern_predictions"] = [
        _published_pattern_prediction(item) for item in _pattern_predictions(forecast)
    ]
    canonical = json.dumps(published, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strategy_return(direction: str, daily_return: float) -> float:
    if direction == "up":
        return daily_return - TRADING_COST
    if direction == "down":
        return -daily_return - TRADING_COST
    return 0.0


def _directional_hit(direction: str, daily_return: float) -> bool:
    if direction == "up":
        return daily_return > 0
    if direction == "down":
        return daily_return < 0
    return abs(daily_return) <= SIDEWAY_LIMIT


def append_official_forecast(
    state: dict[str, Any],
    forecast: dict[str, Any],
    lane: str,
    issued_at: str,
    closed_through: str,
    model_selection: Sequence[dict[str, Any]] | None = None,
) -> bool:
    target_date = str(forecast.get("date", ""))[:10]
    if not target_date:
        return False
    forecast_id = f"{lane}|{target_date}"
    existing = {
        str(row.get("forecast_id")): row
        for row in state.get("forecasts", [])
        if isinstance(row, dict)
    }
    if forecast_id in existing:
        row = existing[forecast_id]
        if row.get("immutable_digest") != official_forecast_digest(row):
            raise RuntimeError(f"Published forecast integrity check failed: {forecast_id}")
        return False

    direction = str(forecast.get("forecast", "no-call"))
    pending_status = "no-call" if direction == "no-call" else "pending"
    model_predictions: list[dict[str, Any]] = [{
        "model": f"{lane} Ensemble",
        "forecast": direction,
        "status": pending_status,
        "score": None,
        "strategy_return": None,
        "prob_down": forecast.get("prob_down"),
        "prob_sideway": forecast.get("prob_sideway"),
        "prob_up": forecast.get("prob_up"),
        "source": "ensemble",
    }]
    seen_models = {f"{lane} Ensemble"}
    for candidate in model_selection or []:
        model = str(candidate.get("model", ""))
        candidate_direction = candidate.get("next_forecast")
        if not model or model in seen_models or candidate_direction not in {"up", "down", "sideway", "no-call"}:
            continue
        seen_models.add(model)
        model_predictions.append({
            "model": model,
            "forecast": candidate_direction,
            "status": "no-call" if candidate_direction == "no-call" else "pending",
            "score": None,
            "strategy_return": None,
            "prob_down": candidate.get("next_prob_down"),
            "prob_sideway": candidate.get("next_prob_sideway"),
            "prob_up": candidate.get("next_prob_up"),
            "source": "candidate",
            "selection_status": candidate.get("status"),
            "calibration_score": candidate.get("calibration_score"),
        })

    top_pattern = copy.deepcopy(forecast.get("top_pattern"))
    matches = forecast.get("matching_patterns", [])
    if not isinstance(matches, list) or not matches:
        matches = [top_pattern] if isinstance(top_pattern, dict) else []
    pattern_predictions: list[dict[str, Any]] = []
    seen_patterns: set[tuple[str, str]] = set()
    for match in matches:
        if not isinstance(match, dict) or not match.get("id") or not match.get("direction"):
            continue
        key = str(match["id"]), str(match["direction"])
        if key in seen_patterns:
            continue
        seen_patterns.add(key)
        pattern_predictions.append({
            "pattern_id": match.get("id"),
            "pattern": match.get("name"),
            "direction": match.get("direction"),
            "rank_at_issue": match.get("rank"),
            "status": "pending",
            "score": None,
            "strategy_return": None,
        })
    pattern_evaluation = copy.deepcopy(pattern_predictions[0]) if pattern_predictions else None

    entry = {
        "forecast_id": forecast_id,
        "date": target_date,
        "target_date": target_date,
        "lane": lane,
        "issued_at": issued_at,
        "closed_through_at_issue": closed_through,
        "forecast": direction,
        "status": pending_status,
        "score": None,
        "actual_return": None,
        "evaluated_at": None,
        "confidence": forecast.get("confidence"),
        "prob_down": forecast.get("prob_down"),
        "prob_sideway": forecast.get("prob_sideway"),
        "prob_up": forecast.get("prob_up"),
        "expected_score": forecast.get("expected_score"),
        "decision_margin": forecast.get("decision_margin"),
        "entropy": forecast.get("entropy"),
        "policy_mode": forecast.get("policy_mode"),
        "sideway_penalty": forecast.get("sideway_penalty"),
        "model_members": copy.deepcopy(forecast.get("model_members", [])),
        "model_weights": copy.deepcopy(forecast.get("model_weights", [])),
        "top_pattern": top_pattern,
        "model_predictions": model_predictions,
        "pattern_predictions": pattern_predictions,
        "pattern_evaluation": pattern_evaluation,
    }
    entry["immutable_digest"] = official_forecast_digest(entry)
    state.setdefault("forecasts", []).append(entry)
    state["forecasts"] = sorted(
        state["forecasts"], key=lambda row: (str(row.get("target_date", "")), str(row.get("lane", ""))),
    )
    state["updated_at"] = issued_at
    return True


def grade_learning_state(
    state: dict[str, Any],
    market: pd.DataFrame,
    latest_closed: pd.Timestamp,
    evaluated_at: str,
) -> int:
    candles = market.copy()
    candles["date"] = pd.to_datetime(candles["timestamp"]).dt.normalize()
    candles["actual_return"] = candles["close"] / candles["open"] - 1
    realized = candles.drop_duplicates("date", keep="last").set_index("date")
    evaluated = 0
    closed_date = pd.Timestamp(latest_closed).normalize()
    for row in state.get("forecasts", []):
        if row.get("evaluated_at"):
            continue
        target = pd.Timestamp(row.get("target_date", row.get("date"))).normalize()
        if target > closed_date or target not in realized.index:
            continue
        actual_return = float(realized.at[target, "actual_return"])
        row["actual_return"] = actual_return
        row["evaluated_at"] = evaluated_at
        row["market_close"] = float(realized.at[target, "close"])
        if row.get("forecast") == "no-call":
            row["status"] = "no-call"
            row["score"] = None
            row["strategy_return"] = 0.0
        else:
            status, score = grade_forecast(str(row.get("forecast")), actual_return)
            row["status"] = status
            row["score"] = score
            row["strategy_return"] = _strategy_return(str(row.get("forecast")), actual_return)
        for prediction in row.get("model_predictions", []):
            direction = str(prediction.get("forecast", "no-call"))
            if direction == "no-call":
                prediction["status"] = "no-call"
                prediction["score"] = None
                prediction["strategy_return"] = 0.0
            else:
                status, score = grade_forecast(direction, actual_return)
                prediction["status"] = status
                prediction["score"] = score
                prediction["strategy_return"] = _strategy_return(direction, actual_return)
                prediction["directional_hit"] = _directional_hit(direction, actual_return)
        for pattern in _pattern_predictions(row):
            direction = str(pattern.get("direction", "no-call"))
            status, score = grade_forecast(direction, actual_return)
            pattern["status"] = status
            pattern["score"] = score
            pattern["strategy_return"] = _strategy_return(direction, actual_return)
        if row.get("pattern_predictions") and isinstance(row.get("pattern_evaluation"), dict):
            row["pattern_evaluation"].update(copy.deepcopy(row["pattern_predictions"][0]))
        evaluated += 1
    state["updated_at"] = evaluated_at
    return evaluated


def _latest_active_models(state: dict[str, Any], lane: str) -> set[str]:
    for snapshot in reversed(state.get("selection_history", [])):
        lane_data = snapshot.get("lanes", {}).get(lane)
        if isinstance(lane_data, dict):
            return set(lane_data.get("active_models", []))
    return set()


def _latest_active_patterns(state: dict[str, Any], lane: str) -> set[str]:
    for snapshot in reversed(state.get("selection_history", [])):
        lane_data = snapshot.get("lanes", {}).get(lane)
        if isinstance(lane_data, dict):
            return set(lane_data.get("active_patterns", []))
    return set()


def apply_live_model_ranking(
    metrics: pd.DataFrame,
    state: dict[str, Any],
    lane: str,
    active_limit: int = 4,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    output = metrics.copy()
    observations: dict[str, list[dict[str, Any]]] = {}
    for forecast in state.get("forecasts", []):
        if forecast.get("lane") != lane or not forecast.get("evaluated_at"):
            continue
        for prediction in forecast.get("model_predictions", []):
            if prediction.get("status") not in {"correct", "partial", "wrong"}:
                continue
            observations.setdefault(str(prediction.get("model")), []).append(prediction)

    live_samples: list[int] = []
    live_weighted: list[float] = []
    live_exact: list[float] = []
    live_directional: list[float] = []
    live_expectancy: list[float] = []
    adjusted_accuracy: list[float] = []
    adaptive_scores: list[float] = []
    for _, row in output.iterrows():
        samples = observations.get(str(row["model"]), [])
        count = len(samples)
        scores = [float(item["score"]) for item in samples]
        exact = [item["status"] == "correct" for item in samples]
        directional = [bool(item.get("directional_hit")) for item in samples]
        returns = [float(item.get("strategy_return", 0.0)) for item in samples]
        weighted_value = float(np.mean(scores)) if scores else math.nan
        exact_value = float(np.mean(exact)) if exact else math.nan
        directional_value = float(np.mean(directional)) if directional else math.nan
        expectancy_value = float(np.mean(returns)) if returns else math.nan
        posterior = float(
            (float(row["weighted_accuracy"]) * MODEL_PRIOR_STRENGTH + sum(scores))
            / (MODEL_PRIOR_STRENGTH + count)
        )
        evidence = min(1.0, count / 20.0)
        expectancy_scaled = float(np.clip(expectancy_value if np.isfinite(expectancy_value) else row["expectancy"], -0.01, 0.01) / 0.02 + 0.5)
        live_quality = (
            posterior * 0.55
            + (directional_value if np.isfinite(directional_value) else float(row["directional_accuracy"])) * 0.30
            + expectancy_scaled * 0.15
        )
        adaptive = float(row["rank_score"]) * (1 - 0.40 * evidence) + live_quality * (0.40 * evidence)
        live_samples.append(count)
        live_weighted.append(weighted_value)
        live_exact.append(exact_value)
        live_directional.append(directional_value)
        live_expectancy.append(expectancy_value)
        adjusted_accuracy.append(posterior)
        adaptive_scores.append(adaptive)

    output["live_samples"] = live_samples
    output["live_weighted_accuracy"] = live_weighted
    output["live_exact_accuracy"] = live_exact
    output["live_directional_accuracy"] = live_directional
    output["live_expectancy"] = live_expectancy
    output["adjusted_weighted_accuracy"] = adjusted_accuracy
    output["adaptive_rank_score"] = adaptive_scores
    output = output.sort_values("adaptive_rank_score", ascending=False).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    eligible = (output["expectancy"] > 0) & (output["coverage"] >= 0.80)
    active_indices = output.index[eligible][:active_limit]
    if len(active_indices) < min(2, len(output)):
        active_indices = output.index[: min(2, len(output))]
    output["status"] = "standby"
    output.loc[active_indices, "status"] = "active"

    previous = _latest_active_models(state, lane)
    current = set(output.loc[output["status"] == "active", "model"].astype(str))
    output["selection_change"] = [
        "promoted" if model in current and model not in previous
        else "demoted" if model not in current and model in previous
        else "retained" if model in current
        else "standby"
        for model in output["model"].astype(str)
    ]
    output["replacement_reason"] = np.where(
        output["status"] == "active",
        "top adaptive OOS + live rank",
        np.where(output["expectancy"] <= 0, "non-positive OOS expectancy", "below active rank"),
    )
    return output


def apply_live_pattern_ranking(
    registry: pd.DataFrame,
    state: dict[str, Any],
    lane: str,
    active_limit: int = 16,
) -> pd.DataFrame:
    if registry.empty:
        return registry
    output = registry.copy()
    observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for forecast in state.get("forecasts", []):
        if forecast.get("lane") != lane or not forecast.get("evaluated_at"):
            continue
        for pattern in _pattern_predictions(forecast):
            if pattern.get("status") not in {"correct", "partial", "wrong"}:
                continue
            key = (str(pattern.get("pattern_id")), str(pattern.get("direction")))
            observations.setdefault(key, []).append(pattern)

    live_counts: list[int] = []
    live_weighted: list[float] = []
    adjusted_weighted: list[float] = []
    adaptive_scores: list[float] = []
    for _, row in output.iterrows():
        samples = observations.get((str(row["pattern_id"]), str(row["direction"])), [])
        scores = [float(item["score"]) for item in samples]
        returns = [float(item.get("strategy_return", 0.0)) for item in samples]
        count = len(samples)
        posterior = float(
            (float(row["weighted_accuracy"]) * PATTERN_PRIOR_STRENGTH + sum(scores))
            / (PATTERN_PRIOR_STRENGTH + count)
        )
        live_expectancy = float(np.mean(returns)) if returns else float(row["expectancy"])
        live_rank = posterior * math.log1p(float(row["occurrences"])) + float(np.clip(live_expectancy, -0.03, 0.03) * 8)
        evidence = min(1.0, count / 12.0)
        adaptive = float(row["rank_score"]) * (1 - 0.40 * evidence) + live_rank * (0.40 * evidence)
        live_counts.append(count)
        live_weighted.append(float(np.mean(scores)) if scores else math.nan)
        adjusted_weighted.append(posterior)
        adaptive_scores.append(adaptive)
    output["live_occurrences"] = live_counts
    output["live_weighted_accuracy"] = live_weighted
    output["adjusted_weighted_accuracy"] = adjusted_weighted
    output["adaptive_rank_score"] = adaptive_scores
    output["eligible"] = output["eligible"].astype(bool) & (output["adjusted_weighted_accuracy"] >= 0.36)
    output = output.sort_values(["eligible", "adaptive_rank_score", "occurrences"], ascending=False).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    output["status"] = np.where(output["eligible"] & (output["rank"] <= active_limit), "active", "standby")
    previous = _latest_active_patterns(state, lane)
    tokens = output["pattern_id"].astype(str) + "|" + output["direction"].astype(str)
    current = set(tokens[output["status"] == "active"])
    output["selection_change"] = [
        "promoted" if token in current and token not in previous
        else "demoted" if token not in current and token in previous
        else "retained" if token in current
        else "standby"
        for token in tokens
    ]
    output["replacement_reason"] = np.where(
        output["status"] == "active",
        "top adaptive OOS + live rank",
        np.where(~output["eligible"], "adjusted hit below 36% or insufficient OOS evidence", "below top-16 rank"),
    )
    return output


def record_selection_snapshot(
    state: dict[str, Any],
    as_of_closed: str,
    generated_at: str,
    model_metrics: dict[str, pd.DataFrame],
    pattern_registries: dict[str, pd.DataFrame],
) -> bool:
    history = state.setdefault("selection_history", [])
    if any(str(item.get("as_of_closed")) == as_of_closed for item in history):
        return False
    lanes: dict[str, Any] = {}
    for lane, metrics in model_metrics.items():
        registry = pattern_registries.get(lane, pd.DataFrame())
        lanes[lane] = {
            "active_models": metrics.loc[metrics["status"] == "active", "model"].astype(str).tolist(),
            "promoted_models": metrics.loc[metrics.get("selection_change", "") == "promoted", "model"].astype(str).tolist(),
            "demoted_models": metrics.loc[metrics.get("selection_change", "") == "demoted", "model"].astype(str).tolist(),
            "active_patterns": (
                registry.loc[registry.get("status", "") == "active", "pattern_id"].astype(str)
                + "|"
                + registry.loc[registry.get("status", "") == "active", "direction"].astype(str)
            ).head(16).tolist() if not registry.empty else [],
        }
    history.append({"as_of_closed": as_of_closed, "generated_at": generated_at, "lanes": lanes})
    state["selection_history"] = history[-MAX_SELECTION_HISTORY:]
    state["updated_at"] = generated_at
    return True


def learning_summary(state: dict[str, Any]) -> dict[str, Any]:
    forecasts = state.get("forecasts", [])
    evaluated = [row for row in forecasts if row.get("evaluated_at")]
    scored = [row for row in evaluated if row.get("status") in {"correct", "partial", "wrong"}]
    return {
        "official_forecasts": len(forecasts),
        "evaluated_forecasts": len(evaluated),
        "pending_forecasts": sum(row.get("status") == "pending" for row in forecasts),
        "no_calls": sum(row.get("status") == "no-call" for row in forecasts),
        "correct": sum(row.get("status") == "correct" for row in forecasts),
        "partial": sum(row.get("status") == "partial" for row in forecasts),
        "wrong": sum(row.get("status") == "wrong" for row in forecasts),
        "live_weighted_accuracy": float(np.mean([row["score"] for row in scored])) if scored else None,
        "last_evaluated_date": max((row.get("target_date") for row in evaluated), default=None),
        "last_selection_date": state.get("selection_history", [{}])[-1].get("as_of_closed") if state.get("selection_history") else None,
    }
