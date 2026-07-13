from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = {"LSTM", "Transformer", "TCN", "PatchTST", "Compact TFT", "iTransformer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the isolated deep challenger artifact")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "public" / "data" / "deep_research.json",
    )
    return parser.parse_args()


def verify(payload: dict[str, object]) -> dict[str, object]:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict) or int(meta.get("schema_version", 0)) < 1:
        raise AssertionError("deep artifact schema is missing")
    if not meta.get("official_ledger_isolation"):
        raise AssertionError("deep research must remain isolated from the official ledger")
    lineage = meta.get("source_lineage", [])
    if not isinstance(lineage, list) or len(lineage) < 6 or any(not row.get("sha256") for row in lineage):
        raise AssertionError("deep source lineage is incomplete")
    dataset = payload.get("dataset", {})
    if not isinstance(dataset, dict) or int(dataset.get("maximum_last_bar_violation", -1)) != 0:
        raise AssertionError("deep dataset contains a target-time leakage violation")

    models = payload.get("models", {})
    if not isinstance(models, dict):
        raise AssertionError("deep models payload is missing")
    available = {str(row.get("model")) for row in models.get("availability", []) if row.get("available")}
    if available != REQUIRED_MODELS:
        raise AssertionError(f"deep model availability mismatch: {sorted(available)}")
    rankings = models.get("rankings", [])
    ranked = {str(row.get("model")) for row in rankings}
    if ranked != REQUIRED_MODELS:
        raise AssertionError(f"not every deep architecture has OOS metrics: {sorted(ranked)}")
    if any(int(row.get("observations", 0)) < 180 for row in rankings):
        raise AssertionError("a deep architecture has insufficient chronological OOS observations")
    if any(row.get("status") != "challenger" for row in rankings):
        raise AssertionError("deep architectures cannot bypass live promotion governance")

    for fold in models.get("folds", []):
        train_end = pd.Timestamp(fold["train_end"])
        calibration_start = pd.Timestamp(fold["calibration_start"])
        calibration_end = pd.Timestamp(fold["calibration_end"])
        test_start = pd.Timestamp(fold["test_start"])
        if not (train_end < calibration_start <= calibration_end < test_start):
            raise AssertionError(f"invalid deep chronology: {fold.get('model')} / {fold.get('fold')}")

    predictions = payload.get("oos_predictions", [])
    identifiers = [(str(row.get("architecture")), str(row.get("date"))) for row in predictions]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("deep OOS predictions contain duplicate model-date rows")
    for row in predictions:
        probability_sum = sum(float(row[key]) for key in ("prob_down", "prob_sideway", "prob_up"))
        if abs(probability_sum - 1.0) > 1e-6:
            raise AssertionError("deep OOS probabilities do not sum to one")

    latest = payload.get("latest_forecasts", [])
    latest_models = {str(row.get("model")) for row in latest}
    if latest_models and latest_models != REQUIRED_MODELS:
        raise AssertionError("latest deep forecasts are incomplete")
    for row in latest:
        if pd.Timestamp(row["available_through_utc"]) > pd.Timestamp(row["target_date"], tz="UTC"):
            raise AssertionError(f"latest deep forecast leaks a future bar: {row.get('model')}")
    return {
        "samples": int(meta["samples"]),
        "architectures": len(ranked),
        "oos_predictions": len(predictions),
        "latest_forecasts": len(latest),
        "dataset_end": str(meta["dataset_end"]),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    print(json.dumps(verify(payload), indent=2))


if __name__ == "__main__":
    main()
