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

    performance = artifact.get("performance", {})
    rankings = performance.get("model_rankings", []) if isinstance(performance, dict) else []
    for lane in ("Calendar", "Full Hybrid"):
        lane_rows = [row for row in rankings if row.get("lane") == lane]
        active = [row for row in lane_rows if row.get("status") == "active"]
        if not active or len(active) > 4:
            raise AssertionError(f"{lane} must have between 1 and 4 active models")
        ranks = [int(row["rank"]) for row in lane_rows]
        if ranks != sorted(ranks):
            raise AssertionError(f"{lane} model rankings must be sorted")

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
    selection_history = state.get("selection_history", [])
    if not selection_history or selection_history[-1].get("as_of_closed") != latest_closed.strftime("%Y-%m-%d"):
        raise AssertionError("daily selection snapshot is missing")
    return {
        "latest_closed_utc": latest_closed.strftime("%Y-%m-%d"),
        "official_forecasts": len(forecasts),
        "evaluated": evaluated,
        "pending": pending,
        "selection_snapshots": len(selection_history),
    }


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    state = load_learning_state(args.state)
    print(json.dumps(verify(artifact, state), indent=2))


if __name__ == "__main__":
    main()
