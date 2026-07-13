from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.hybrid_core import grade_forecast
from research.learning import load_learning_state, official_forecast_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the published learning and forecast contract")
    parser.add_argument("--artifact", type=Path, default=ROOT / "public" / "data" / "hybrid_research.json")
    parser.add_argument("--state", type=Path, default=ROOT / "data" / "learning_state.json")
    return parser.parse_args()


def verify(artifact: dict[str, object], state: dict[str, object]) -> dict[str, object]:
    meta = artifact.get("meta", {})
    if not isinstance(meta, dict) or int(meta.get("schema_version", 0)) < 3:
        raise AssertionError("artifact schema must include immutable learning state")
    latest_closed = pd.Timestamp(meta["latest_closed_utc"]).normalize()
    generated_at = pd.Timestamp(meta["generated_at"])
    if generated_at.tzinfo is not None:
        generated_at = generated_at.tz_convert("UTC").tz_localize(None)
    expected_closed = generated_at.normalize() - pd.Timedelta(days=1)
    if latest_closed < expected_closed:
        raise AssertionError(
            f"market artifact is stale: latest={latest_closed:%Y-%m-%d}, expected={expected_closed:%Y-%m-%d}",
        )
    if int(meta.get("schema_version", 0)) >= 4:
        health = artifact.get("health", {})
        market_health = health.get("market", {}) if isinstance(health, dict) else {}
        if not isinstance(market_health, dict) or bool(market_health.get("stale", True)):
            raise AssertionError("schema v4 artifact must contain a non-stale market health record")
        if str(market_health.get("actual_closed_utc")) != latest_closed.strftime("%Y-%m-%d"):
            raise AssertionError("market health actual date must match artifact latest closed date")
        lineage = meta.get("data_lineage", [])
        if not isinstance(lineage, list) or len(lineage) < 4:
            raise AssertionError("schema v4 artifact must contain source lineage")
        if any(not isinstance(item, dict) or not item.get("sha256") for item in lineage):
            raise AssertionError("every lineage source must include a SHA-256 digest")
    forecasts = state.get("forecasts", [])
    if not isinstance(forecasts, list):
        raise AssertionError("learning forecasts must be a list")
    forecast_ids = [str(row.get("forecast_id")) for row in forecasts if isinstance(row, dict)]
    if len(forecast_ids) != len(set(forecast_ids)):
        raise AssertionError("official forecast IDs must be unique")

    evaluated = 0
    pending = 0
    for row in forecasts:
        if not isinstance(row, dict):
            raise AssertionError("ledger row must be an object")
        target = pd.Timestamp(row["target_date"]).normalize()
        if target <= pd.Timestamp(row["closed_through_at_issue"]).normalize():
            raise AssertionError(f"forecast was issued after its target closed: {row['forecast_id']}")
        if row.get("immutable_digest") != official_forecast_digest(row):
            raise AssertionError(f"published forecast was mutated: {row['forecast_id']}")
        if target <= latest_closed and not row.get("evaluated_at"):
            raise AssertionError(f"closed forecast was not evaluated: {row['forecast_id']}")
        if row.get("evaluated_at"):
            evaluated += 1
            if row.get("forecast") != "no-call":
                expected_status, expected_score = grade_forecast(str(row["forecast"]), float(row["actual_return"]))
                if row.get("status") != expected_status or float(row.get("score")) != expected_score:
                    raise AssertionError(f"invalid official grade: {row['forecast_id']}")
            for prediction in row.get("model_predictions", []):
                if prediction.get("forecast") == "no-call":
                    continue
                expected_status, expected_score = grade_forecast(
                    str(prediction["forecast"]), float(row["actual_return"]),
                )
                if prediction.get("status") != expected_status or float(prediction.get("score")) != expected_score:
                    raise AssertionError(f"invalid model grade: {row['forecast_id']} / {prediction.get('model')}")
            patterns = row.get("pattern_predictions") or ([row.get("pattern_evaluation")] if row.get("pattern_evaluation") else [])
            for prediction in patterns:
                expected_status, expected_score = grade_forecast(
                    str(prediction["direction"]), float(row["actual_return"]),
                )
                if prediction.get("status") != expected_status or float(prediction.get("score")) != expected_score:
                    raise AssertionError(
                        f"invalid pattern grade: {row['forecast_id']} / {prediction.get('pattern_id')}",
                    )
        elif row.get("status") == "pending":
            pending += 1

    events = state.get("event_evaluations", [])
    if not isinstance(events, list):
        raise AssertionError("event evaluation ledger must be a list")
    event_ids = [str(row.get("event_id")) for row in events if isinstance(row, dict)]
    if len(event_ids) != len(set(event_ids)):
        raise AssertionError("event evaluation IDs must be unique")
    for event in events:
        if not isinstance(event, dict):
            raise AssertionError("event evaluation row must be an object")
        maturity = pd.Timestamp(event["matures_after"]).normalize()
        if maturity <= latest_closed and not event.get("evaluated_at"):
            raise AssertionError(f"mature event was not evaluated: {event.get('event_id')}")
        if event.get("status") not in {"pending", "matched", "not-matched"}:
            raise AssertionError(f"invalid event status: {event.get('event_id')}")
        if event.get("evaluated_at") and float(event.get("score")) not in {0.0, 1.0}:
            raise AssertionError(f"invalid event score: {event.get('event_id')}")

    performance = artifact.get("performance", {})
    rankings = performance.get("model_rankings", []) if isinstance(performance, dict) else []
    models_payload = artifact.get("models", {})
    availability = models_payload.get("availability", []) if isinstance(models_payload, dict) else []
    required_oos_models = {
        str(row["model"])
        for row in availability
        if row.get("available") and row.get("cadence") == "daily"
    }
    for lane in ("Calendar", "Full Hybrid"):
        lane_rows = [row for row in rankings if row.get("lane") == lane]
        active = [row for row in lane_rows if row.get("status") == "active"]
        if not active or len(active) > 4:
            raise AssertionError(f"{lane} must have between 1 and 4 active models")
        ranks = [int(row["rank"]) for row in lane_rows]
        if ranks != sorted(ranks):
            raise AssertionError(f"{lane} model rankings must be sorted")
        if any("weighted_lcb" not in row or "ece" not in row for row in lane_rows):
            raise AssertionError(f"{lane} rankings are missing uncertainty or calibration metrics")
        oos_by_model = {str(row["model"]): int(row.get("observations", 0)) for row in lane_rows}
        missing_oos = sorted(required_oos_models - set(oos_by_model))
        if missing_oos:
            raise AssertionError(f"{lane} is missing OOS predictions for: {', '.join(missing_oos)}")
        if any(oos_by_model[model] < 365 for model in required_oos_models):
            raise AssertionError(f"{lane} contains an available daily model without a sufficient OOS history")
    for fold_key in ("calendar_folds", "full_hybrid_folds"):
        for fold in performance.get(fold_key, []):
            train_end = pd.Timestamp(fold["train_end"])
            calibration_fit_end = pd.Timestamp(fold["calibration_fit_end"])
            policy_start = pd.Timestamp(fold["policy_start"])
            test_start = pd.Timestamp(fold["test_start"])
            if not (train_end < calibration_fit_end < policy_start < test_start):
                raise AssertionError(f"invalid nested walk-forward ordering in {fold_key}: {fold.get('fold')}")
            weights = [float(value) for value in fold.get("weights", [])]
            members = fold.get("members", [])
            if fold.get("stacking_method") != "nonnegative-simplex-oof":
                raise AssertionError(f"invalid stacking method in {fold_key}: {fold.get('fold')}")
            if not weights or len(weights) != len(members) or any(value < 0 for value in weights):
                raise AssertionError(f"invalid stacking weights in {fold_key}: {fold.get('fold')}")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise AssertionError(f"stacking weights do not sum to one in {fold_key}: {fold.get('fold')}")

    patterns = artifact.get("patterns", {})
    if not isinstance(patterns, dict):
        raise AssertionError("patterns payload is missing")
    for key in ("calendar", "full_hybrid"):
        active = [row for row in patterns.get(key, []) if row.get("status") == "active"]
        if len(active) > 16:
            raise AssertionError(f"{key} has more than 16 active patterns")
        if any(not row.get("eligible") for row in active):
            raise AssertionError(f"{key} contains an ineligible active pattern")

    learning = artifact.get("learning", {})
    summary = learning.get("summary", {}) if isinstance(learning, dict) else {}
    if int(summary.get("official_forecasts", -1)) != len(forecasts):
        raise AssertionError("artifact learning summary is out of sync with state")
    artifact_ledger = learning.get("official_forecast_ledger", []) if isinstance(learning, dict) else []
    artifact_digests = {
        str(row.get("forecast_id")): official_forecast_digest(row)
        for row in artifact_ledger if isinstance(row, dict)
    }
    state_digests = {str(row["forecast_id"]): official_forecast_digest(row) for row in forecasts}
    if artifact_digests != state_digests:
        raise AssertionError("published artifact and production ledger disagree")
    artifact_events = learning.get("event_evaluation_ledger", []) if isinstance(learning, dict) else []
    if {str(row.get("event_id")) for row in artifact_events} != set(event_ids):
        raise AssertionError("published event ledger and production state disagree")
    selection_history = state.get("selection_history", [])
    if not selection_history or selection_history[-1].get("as_of_closed") != latest_closed.strftime("%Y-%m-%d"):
        raise AssertionError("daily selection snapshot is missing")
    return {
        "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
        "official_forecasts": len(forecasts),
        "evaluated": evaluated,
        "pending": pending,
        "event_evaluations": len(events),
        "selection_snapshots": len(selection_history),
    }


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    state = load_learning_state(args.state)
    print(json.dumps(verify(artifact, state), indent=2))


if __name__ == "__main__":
    main()
