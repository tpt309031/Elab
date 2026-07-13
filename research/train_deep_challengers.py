from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.deep_dataset import (
    build_intraday_feature_table,
    build_intraday_sequence_dataset,
    chronological_folds,
    sequence_for_target,
)
from research.deep_models import TorchSequenceClassifier
from research.evaluation import ProbabilityCalibrator, conservative_beta_lower_bound, multiclass_diagnostics
from research.hybrid_core import TRADING_COST, build_feature_frame, grade_forecast, load_astro, load_indices


ARCHITECTURES = {
    "LSTM": "lstm",
    "Transformer": "transformer",
    "TCN": "tcn",
    "PatchTST": "patchtst",
    "Compact TFT": "tft",
    "iTransformer": "itransformer",
}
CLASS_NAMES = ("down", "sideway", "up")
CONTEXT_COLUMNS = (
    "index_BTC", "index_me", "gap_index", "index_btc_change_1", "index_me_change_1",
    "index_btc_slope_3", "index_me_slope_3", "same_phase", "opposite_phase",
    "finance", "volatility", "composite", "astro_composite_change_1", "astro_event",
    "dow_sin", "dow_cos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated leakage-safe intraday deep challengers")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=42, help="Number of closed 4h steps per sample")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "public" / "data" / "deep_research.json")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(probabilities, dtype=float), 1e-8, None)
    return output / output.sum(axis=1, keepdims=True)


def _prediction_rows(
    architecture: str,
    fold_id: str,
    dates: np.ndarray,
    returns: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date, daily_return, actual, probability in zip(dates, returns, labels, probabilities):
        predicted = int(np.argmax(probability))
        direction = CLASS_NAMES[predicted]
        status, score = grade_forecast(direction, float(daily_return))
        if direction == "up":
            strategy_return = float(daily_return) - TRADING_COST
        elif direction == "down":
            strategy_return = -float(daily_return) - TRADING_COST
        else:
            strategy_return = 0.0
        rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "architecture": architecture,
            "fold": fold_id,
            "forecast": direction,
            "actual_class": CLASS_NAMES[int(actual)],
            "status": status,
            "score": score,
            "daily_return": float(daily_return),
            "strategy_return": strategy_return,
            "directional_hit": bool(predicted == int(actual)),
            "prob_down": float(probability[0]),
            "prob_sideway": float(probability[1]),
            "prob_up": float(probability[2]),
            "confidence": float(probability[predicted]),
        })
    return rows


def _metric_rows(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(predictions)
    rows: list[dict[str, object]] = []
    for architecture, group in frame.groupby("architecture"):
        probabilities = group[["prob_down", "prob_sideway", "prob_up"]].to_numpy(dtype=float)
        labels = group["actual_class"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
        diagnostics = multiclass_diagnostics(probabilities, labels)
        strategy = group["strategy_return"].to_numpy(dtype=float)
        equity = np.cumprod(1 + strategy)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1
        positive = strategy[strategy > 0].sum()
        negative = -strategy[strategy < 0].sum()
        hits = int(group["directional_hit"].sum())
        directional_accuracy = float(hits / len(group))
        expectancy = float(strategy.mean())
        rows.append({
            "model": architecture,
            "family": "intraday-sequence",
            "observations": int(len(group)),
            "oos_start": str(group["date"].min()),
            "oos_end": str(group["date"].max()),
            "exact_accuracy": float((group["status"] == "correct").mean()),
            "weighted_accuracy": float(group["score"].mean()),
            "directional_accuracy": directional_accuracy,
            "directional_lcb": conservative_beta_lower_bound(
                group["directional_hit"].astype(float).to_numpy(),
                prior_mean=1 / 3,
            ),
            "mcc": diagnostics["mcc"],
            "log_loss": diagnostics["log_loss"],
            "ece": diagnostics["ece"],
            "expectancy": expectancy,
            "profit_factor": float(positive / negative) if negative > 0 else math.inf,
            "net_return": float(equity[-1] - 1),
            "max_drawdown": float(drawdown.min()),
            "status": "challenger",
            "promotion_eligible": False,
            "promotion_reason": "Requires at least 20 immutable live grades and a monthly promotion review.",
        })
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return []
    metrics["rank_score"] = (
        metrics["directional_lcb"] * 0.50
        + metrics["weighted_accuracy"] * 0.20
        + (1 - metrics["ece"].clip(0, 1)) * 0.15
        + metrics["expectancy"].clip(-0.01, 0.01).add(0.01).div(0.02) * 0.15
    )
    metrics = metrics.sort_values(["rank_score", "log_loss"], ascending=[False, True]).reset_index(drop=True)
    metrics["rank"] = np.arange(1, len(metrics) + 1)
    return metrics.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    one_hour_path = ROOT / "data" / "cache" / "BTCUSDT_1h.csv"
    four_hour_path = ROOT / "data" / "cache" / "BTCUSDT_4h.csv"
    daily_path = ROOT / "data" / "cache" / "BTCUSDT_1d.csv"
    index_btc_path = ROOT / "data" / "newdata" / "index_btc.csv"
    index_me_path = ROOT / "data" / "newdata" / "index_me.csv"
    astro_path = ROOT / "public" / "data" / "astro_scores.json"

    one_hour = pd.read_csv(one_hour_path, parse_dates=["timestamp"])
    four_hour = pd.read_csv(four_hour_path, parse_dates=["timestamp"])
    daily = pd.read_csv(daily_path, parse_dates=["timestamp"])
    indices = load_indices(index_btc_path, index_me_path)
    astro = load_astro(astro_path)
    context, _ = build_feature_frame(indices, daily, astro)
    dataset = build_intraday_sequence_dataset(
        one_hour,
        four_hour,
        daily,
        context=context[["date"] + list(CONTEXT_COLUMNS)],
        context_columns=CONTEXT_COLUMNS,
        lookback=args.lookback,
    )
    folds = chronological_folds(dataset.dates, n_folds=args.folds)
    if not folds:
        raise RuntimeError("No chronological deep-research folds satisfy the minimum train history")
    flattened = dataset.values.reshape(len(dataset.values), -1)
    input_features = len(dataset.feature_names)
    predictions: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    latest_models: dict[str, tuple[TorchSequenceClassifier, ProbabilityCalibrator]] = {}

    for model_number, (architecture_name, architecture_key) in enumerate(ARCHITECTURES.items()):
        prototype = TorchSequenceClassifier(
            architecture=architecture_key,
            lookback=args.lookback,
            input_features=input_features,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            random_state=42 + model_number * 101,
        )
        for fold in folds:
            model = clone(prototype)
            model.fit(flattened[fold.train_index], dataset.labels[fold.train_index])
            calibration_raw = _normalise(model.predict_proba(flattened[fold.calibration_index]))
            calibrator = ProbabilityCalibrator().fit(
                calibration_raw,
                dataset.labels[fold.calibration_index],
            )
            test_probabilities = calibrator.transform(
                _normalise(model.predict_proba(flattened[fold.test_index])),
            )
            predictions.extend(_prediction_rows(
                architecture_name,
                fold.fold_id,
                dataset.dates[fold.test_index],
                dataset.returns[fold.test_index],
                dataset.labels[fold.test_index],
                test_probabilities,
            ))
            fold_rows.append({
                "model": architecture_name,
                "fold": fold.fold_id,
                "train_start": pd.Timestamp(dataset.dates[fold.train_index[0]]).strftime("%Y-%m-%d"),
                "train_end": pd.Timestamp(dataset.dates[fold.train_index[-1]]).strftime("%Y-%m-%d"),
                "calibration_start": pd.Timestamp(dataset.dates[fold.calibration_index[0]]).strftime("%Y-%m-%d"),
                "calibration_end": pd.Timestamp(dataset.dates[fold.calibration_index[-1]]).strftime("%Y-%m-%d"),
                "test_start": pd.Timestamp(dataset.dates[fold.test_index[0]]).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(dataset.dates[fold.test_index[-1]]).strftime("%Y-%m-%d"),
                "calibration_method": calibrator.diagnostics_.method,
                "calibration_selection_log_loss": calibrator.diagnostics_.selection_log_loss,
                "validation_loss": float(model.validation_loss_),
            })

        calibration_size = 90
        train_end = len(flattened) - calibration_size - 1
        initial = clone(prototype)
        initial.fit(flattened[:train_end], dataset.labels[:train_end])
        latest_calibrator = ProbabilityCalibrator().fit(
            _normalise(initial.predict_proba(flattened[-calibration_size:])),
            dataset.labels[-calibration_size:],
        )
        final_model = clone(prototype)
        final_model.fit(flattened, dataset.labels)
        latest_models[architecture_name] = (final_model, latest_calibrator)

    intraday_features = build_intraday_feature_table(one_hour, four_hour)
    latest_closed = pd.to_datetime(daily["timestamp"]).max().normalize()
    target_date = latest_closed + pd.Timedelta(days=1)
    context_lookup = context.set_index("date")
    context_row = context_lookup.loc[target_date] if target_date in context_lookup.index else None
    latest_sequence = sequence_for_target(
        intraday_features,
        target_date,
        context_row,
        CONTEXT_COLUMNS,
        args.lookback,
    )
    latest_forecasts: list[dict[str, object]] = []
    if latest_sequence is not None:
        latest_values, available_through = latest_sequence
        latest_flattened = latest_values.reshape(1, -1)
        for architecture_name, (model, calibrator) in latest_models.items():
            probability = calibrator.transform(_normalise(model.predict_proba(latest_flattened)))[0]
            predicted = int(np.argmax(probability))
            latest_forecasts.append({
                "model": architecture_name,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "available_through_utc": pd.Timestamp(available_through).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecast": CLASS_NAMES[predicted],
                "confidence": float(probability[predicted]),
                "prob_down": float(probability[0]),
                "prob_sideway": float(probability[1]),
                "prob_up": float(probability[2]),
                "status": "research-only",
            })

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "meta": {
            "schema_version": 1,
            "generated_at": generated_at,
            "latest_closed_daily_utc": latest_closed.strftime("%Y-%m-%d"),
            "dataset_start": pd.Timestamp(dataset.dates.min()).strftime("%Y-%m-%d"),
            "dataset_end": pd.Timestamp(dataset.dates.max()).strftime("%Y-%m-%d"),
            "samples": int(len(dataset.values)),
            "lookback_4h_steps": args.lookback,
            "input_features": input_features,
            "epochs": args.epochs,
            "folds": len(folds),
            "official_ledger_isolation": True,
            "promotion_gate": "Research-only until at least 20 immutable live grades and monthly review.",
            "source_lineage": [
                {"file": path.name, "sha256": _sha256(path)}
                for path in (one_hour_path, four_hour_path, daily_path, index_btc_path, index_me_path, astro_path)
            ],
        },
        "dataset": {
            "feature_names": list(dataset.feature_names),
            "last_bar_rule": "Every 4h bar must close at or before target-day 00:00 UTC.",
            "maximum_last_bar_violation": int((dataset.last_bar_closed_at > dataset.dates).sum()),
        },
        "models": {
            "availability": [
                {"model": name, "architecture": architecture, "available": True, "status": "challenger"}
                for name, architecture in ARCHITECTURES.items()
            ],
            "rankings": _metric_rows(predictions),
            "folds": fold_rows,
        },
        "oos_predictions": predictions,
        "latest_forecasts": latest_forecasts,
    }
    public_output = args.output
    data_output = ROOT / "data" / "deep_research.json"
    _atomic_json(public_output, payload)
    _atomic_json(data_output, payload)
    print(json.dumps({
        "output": str(public_output),
        "bytes": public_output.stat().st_size,
        "samples": len(dataset.values),
        "oos_predictions": len(predictions),
        "architectures": len(ARCHITECTURES),
        "latest_forecasts": len(latest_forecasts),
        "best_directional_accuracy": max(
            (row["directional_accuracy"] for row in payload["models"]["rankings"]),
            default=None,
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
