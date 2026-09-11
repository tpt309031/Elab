"""Purged, calibrated magnitude forecasts, independent of daily direction grades."""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "large-move-wf-v1"
HORIZONS = {1: 0.04, 3: 0.08, 5: 0.12}


def event_labels(market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError("Unsupported event horizon")
    data = market.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.normalize()
    data = data.drop_duplicates("date", keep="last").set_index("date").sort_index().asfreq("D")
    move = data["close"].shift(1 - horizon) / data["open"] - 1
    complete = data[["open", "high", "low", "close"]].notna().all(axis=1).astype(int)
    complete = complete.rolling(horizon).sum().shift(1 - horizon).eq(horizon)
    move = move.where(complete)
    high = data["high"].rolling(horizon).max().shift(1 - horizon)
    low = data["low"].rolling(horizon).min().shift(1 - horizon)
    return pd.DataFrame({"move": move, "event": (move.abs() > HORIZONS[horizon]).astype(float).where(move.notna()),
                         "range": ((high - low) / data["open"]).where(complete)}).reset_index()


def _fit_candidates(train: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame,
                    target: pd.DataFrame, columns: list[str]) -> tuple[dict, dict, str, float]:
    base = float((train["event"].sum() + 1) / (len(train) + 2))
    predictions = {"Base rate": np.full(len(target), base)}
    validation_probs = {"Base rate": np.full(len(validation), base)}
    models = {
        "Logistic": make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=0.05, max_iter=400, random_state=42)),
        "Histogram GB": make_pipeline(SimpleImputer(keep_empty_features=True),
                                      HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7,
                                        min_samples_leaf=40, l2_regularization=10, early_stopping=False, random_state=42)),
    }
    if train["event"].nunique() == 2 and len(calibration) >= 30:
        for name, model in models.items():
            model.fit(train[columns], train["event"].astype(int))
            def probability(data: pd.DataFrame) -> np.ndarray:
                return np.clip(model.predict_proba(data[columns])[:, 1], 1e-5, 1 - 1e-5)
            p_cal = probability(calibration)
            calibrator = None
            if calibration["event"].value_counts().min() >= 5 and calibration["event"].nunique() == 2:
                calibrator = LogisticRegression(C=0.1, random_state=42).fit(
                    np.log(p_cal / (1 - p_cal)).reshape(-1, 1), calibration["event"].astype(int))
            def calibrated(data: pd.DataFrame) -> np.ndarray:
                p = probability(data)
                return calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1] if calibrator else p
            predictions[name] = calibrated(target)
            validation_probs[name] = calibrated(validation)
    losses = {name: float(brier_score_loss(validation["event"], p)) for name, p in validation_probs.items()}
    champion = min(losses, key=losses.get)
    # A rare-event warning is not the same as a likely UP/DOWN call.
    threshold = float(min(0.7, max(0.35, 2 * base)))
    return predictions, losses, champion, threshold


def _records(target: pd.DataFrame, columns: list[str], history: pd.DataFrame,
             lane: str, horizon: int, cutoff: pd.Timestamp, provenance: str) -> list[dict]:
    # Labels and all forward outcomes must be fully known two sessions before target.
    history = history[history["date"] + pd.Timedelta(days=horizon - 1) <= cutoff - pd.Timedelta(days=2)]
    cal_start = cutoff - pd.Timedelta(days=150)
    train = history[history["date"].between(cutoff - pd.Timedelta(days=1460), cal_start - pd.Timedelta(days=7), inclusive="left")]
    cal = history[(history["date"] >= cal_start) & (history["date"] < cutoff - pd.Timedelta(days=75))]
    val = history[history["date"] >= cutoff - pd.Timedelta(days=68)]
    if len(train) < 365 or len(cal) < 30 or len(val) < 30 or target.empty:
        return []
    predictions, losses, champion, threshold = _fit_candidates(train, cal, val, target, columns)
    rows = []
    for i, (_, day) in enumerate(target.iterrows()):
        p = float(predictions[champion][i])
        actual = float(day["event"]) if pd.notna(day.get("event")) else None
        alert = p >= threshold
        rows.append(dict(date=day["date"].strftime("%Y-%m-%d"), lane=lane, horizon=horizon,
                         threshold_move=HORIZONS[horizon], probability=p, alert=bool(alert), alert_threshold=threshold,
                         model=champion, base_probability=float(predictions["Base rate"][i]),
                         validation_brier=losses[champion], validation_baseline_brier=losses["Base rate"],
                         candidates={name: float(values[i]) for name, values in predictions.items()},
                         actual_event=actual, actual_move=float(day["move"]) if actual is not None else None,
                         actual_range=float(day["range"]) if actual is not None else None,
                         status=_status(alert, actual), source=provenance,
                         training_end=train["date"].max().strftime("%Y-%m-%d"),
                         selection_end=val["date"].max().strftime("%Y-%m-%d"),
                         information_cutoff=(cutoff - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")))
    return rows


def _status(alert: bool, actual: float | None) -> str:
    if actual is None:
        return "pending"
    return ("hit" if actual else "false-alarm") if alert else ("missed" if actual else "quiet")


def _digest(record: dict) -> str:
    keys = ("id", "date", "lane", "horizon", "probability", "alert", "alert_threshold", "model",
            "candidates", "issued_at", "information_cutoff", "version")
    return hashlib.sha256(json.dumps({key: record.get(key) for key in keys}, sort_keys=True).encode()).hexdigest()


def grade_event_forecasts(state: dict, market: pd.DataFrame, run_at: str) -> int:
    labels = {h: event_labels(market, h).set_index("date") for h in HORIZONS}
    evaluated = 0
    for record in state.setdefault("large_move_forecasts", []):
        if _digest(record) != record.get("digest"):
            raise ValueError("Large-move forecast integrity check failed")
        if record.get("evaluated_at"):
            continue
        target = pd.Timestamp(record["date"])
        h = int(record["horizon"])
        if target not in labels[h].index or pd.isna(labels[h].loc[target, "event"]):
            continue
        # Do not evaluate until after the scheduled 03:00 UTC close-processing gate.
        mature_at = (target + pd.Timedelta(days=h, hours=3)).tz_localize("UTC")
        if pd.Timestamp(run_at) < mature_at:
            continue
        actual = labels[h].loc[target]
        record.update(actual_event=float(actual["event"]), actual_move=float(actual["move"]),
                      actual_range=float(actual["range"]), status=_status(record["alert"], actual["event"]), evaluated_at=run_at)
        evaluated += 1
    return evaluated


def event_metrics(rows: list[dict]) -> list[dict]:
    results = []
    data = pd.DataFrame([row for row in rows if row.get("actual_event") is not None])
    if data.empty:
        return results
    for (source, lane, horizon), group in data.groupby(["source", "lane", "horizon"]):
        actual = group["actual_event"].to_numpy(dtype=int)
        candidates = ["Selected", *sorted(set().union(*(row.keys() for row in group["candidates"])))]
        for model in candidates:
            if model == "Selected":
                p = group["probability"].to_numpy(dtype=float)
            else:
                if any(model not in values for values in group["candidates"]):
                    continue
                p = np.array([values[model] for values in group["candidates"]])
            alerts = p >= group["alert_threshold"].to_numpy()
            tp, fp = int((alerts & (actual == 1)).sum()), int((alerts & (actual == 0)).sum())
            fn = int((~alerts & (actual == 1)).sum())
            n = tp + fp
            precision = tp / n if n else None
            # Wilson interval exposes uncertainty when only a few alerts fired.
            lower = ((precision + 1.9208/n - 1.96*np.sqrt(precision*(1-precision)/n + .9604/n**2))/(1+3.8416/n)) if n else None
            baseline_brier = brier_score_loss(actual, group["base_probability"])
            brier = brier_score_loss(actual, p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap = float(average_precision_score(actual, p)) if actual.sum() else None
            results.append(dict(source=source, lane=lane, horizon=int(horizon), model=model, samples=len(group),
                                events=int(actual.sum()), alerts=n, hits=tp, false_alarms=fp, missed=fn,
                                precision=precision, precision_lcb=lower, recall=tp/(tp+fn) if tp+fn else None,
                                base_rate=float(actual.mean()), average_precision=ap, brier=float(brier),
                                brier_skill=float(1-brier/baseline_brier) if baseline_brier else None,
                                start=str(group["date"].min()), end=str(group["date"].max())))
    return results


def build_large_move_research(frame: pd.DataFrame, groups: dict, market: pd.DataFrame,
                              state: dict, run_at: str, previous: dict | None = None, fast: bool = False) -> dict:
    latest = pd.to_datetime(market["timestamp"]).max().normalize()
    first = latest + pd.Timedelta(days=2)
    feature_sets = {"Index + Astro": groups["calendar"],
                    "Hybrid": [c for c in groups["full"] if not c.startswith("spot_")],
                    "Hybrid + Volume": groups["full"]}
    previous = previous or {}
    reuse = fast and previous.get("version") == VERSION and previous.get("historical")
    historical = list(previous["historical"]) if reuse else []
    next_rows = []
    for horizon in HORIZONS:
        labels = event_labels(market, horizon)
        joined = frame.drop(columns=["event", "move", "range"], errors="ignore").merge(labels, on="date", how="left")
        history = joined[joined["event"].notna()]
        for lane, columns in feature_sets.items():
            print(f"[Magnitude] {lane} / {horizon}D", flush=True)
            if not reuse:
                for month in pd.date_range("2024-01-01", latest, freq="MS"):
                    target = joined[(joined["date"] >= month) & (joined["date"] < month + pd.offsets.MonthBegin()) & (joined["date"] <= latest)]
                    historical.extend(_records(target, columns, history, lane, horizon, month, "walk-forward"))
            target = joined[joined["date"] == first]
            next_rows.extend(_records(target, columns, history, lane, horizon, first, "official"))
    # Reused weekly tail can mature daily; never refit its stored predictions.
    label_maps = {h: event_labels(market, h).set_index("date") for h in HORIZONS}
    for row in historical:
        date, horizon = pd.Timestamp(row["date"]), int(row["horizon"])
        if row.get("actual_event") is None and date in label_maps[horizon].index:
            actual = label_maps[horizon].loc[date]
            if pd.notna(actual["event"]):
                row.update(actual_event=float(actual["event"]), actual_move=float(actual["move"]),
                           actual_range=float(actual["range"]), status=_status(row["alert"], actual["event"]))
    ledger = state.setdefault("large_move_forecasts", [])
    existing = {row["id"] for row in ledger}
    for row in next_rows:
        identity = f"{row['lane']}|{row['horizon']}|{row['date']}"
        if identity in existing or pd.Timestamp(run_at) >= pd.Timestamp(row["date"], tz="UTC"):
            continue
        record = dict(row, id=identity, issued_at=run_at, version=VERSION, evaluated_at=None)
        record["digest"] = _digest(record)
        ledger.append(record)
        existing.add(identity)
    evaluated = grade_event_forecasts(state, market, run_at)
    next_rows = [row for row in ledger if row["date"] == first.strftime("%Y-%m-%d")]
    return dict(version=VERSION, generated_at=run_at, next=next_rows, historical=historical,
                official=ledger, metrics=event_metrics([*historical, *ledger]), evaluated_this_run=evaluated,
                definitions={"move": "Absolute open-to-final-close return: 1D >4%, 3D >8%, 5D >12%.",
                             "range": "High-low range is diagnostic, not a substitute for a magnitude hit.",
                             "validation": "Monthly walk-forward from 2024; 7-day purge; disjoint fit/calibration/selection; selection uses Brier loss, with a base-rate candidate.",
                             "alert": "Probability >= max(35%, twice training base rate), capped at 70%; not a trade recommendation.",
                             "provenance": "Index/Astro history without available_at assumes prepublication; retrospective research is not a live track record."})
