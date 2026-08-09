from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


CLASS_NAMES = ("down", "sideway", "up")


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1)
    total = clipped.sum(axis=1, keepdims=True)
    return np.divide(clipped, total, out=np.full_like(clipped, 1 / 3), where=total > 0)


def _temperature_transform(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-7, 1)) / max(0.1, float(temperature))
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


@dataclass
class CalibrationDiagnostics:
    method: str
    selection_log_loss: float
    fit_rows: int
    validation_rows: int


class ProbabilityCalibrator:
    """Chronologically select and refit a multiclass probability calibrator."""

    def __init__(self, minimum_isotonic_rows: int = 120) -> None:
        self.minimum_isotonic_rows = minimum_isotonic_rows
        self.method_ = "identity"
        self.models_: list[object | None] = []
        self.temperature_ = 1.0
        self.diagnostics_ = CalibrationDiagnostics("identity", math.nan, 0, 0)

    def _fit_sigmoid(self, probabilities: np.ndarray, labels: np.ndarray) -> list[LogisticRegression | None]:
        models: list[LogisticRegression | None] = []
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        for class_index in range(3):
            binary = (labels == class_index).astype(int)
            if np.unique(binary).size < 2:
                models.append(None)
                continue
            feature = np.log(clipped[:, class_index] / (1 - clipped[:, class_index])).reshape(-1, 1)
            model = LogisticRegression(C=0.5, max_iter=500)
            model.fit(feature, binary)
            models.append(model)
        return models

    def _transform_sigmoid(
        self,
        probabilities: np.ndarray,
        models: Sequence[LogisticRegression | None],
    ) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        calibrated = np.zeros_like(clipped)
        for class_index, model in enumerate(models):
            if model is None:
                calibrated[:, class_index] = clipped[:, class_index]
                continue
            feature = np.log(clipped[:, class_index] / (1 - clipped[:, class_index])).reshape(-1, 1)
            calibrated[:, class_index] = model.predict_proba(feature)[:, 1]
        return _normalize(calibrated)

    def _fit_isotonic(self, probabilities: np.ndarray, labels: np.ndarray) -> list[IsotonicRegression | None]:
        models: list[IsotonicRegression | None] = []
        for class_index in range(3):
            binary = (labels == class_index).astype(int)
            if np.unique(binary).size < 2 or binary.sum() < 10:
                models.append(None)
                continue
            model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            model.fit(probabilities[:, class_index], binary)
            models.append(model)
        return models

    @staticmethod
    def _transform_isotonic(probabilities: np.ndarray, models: Sequence[IsotonicRegression | None]) -> np.ndarray:
        calibrated = np.zeros_like(probabilities)
        for class_index, model in enumerate(models):
            calibrated[:, class_index] = (
                probabilities[:, class_index]
                if model is None else model.predict(probabilities[:, class_index])
            )
        return _normalize(calibrated)

    @staticmethod
    def _best_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
        candidates = np.linspace(0.45, 3.0, 52)
        losses = [
            log_loss(labels, _temperature_transform(probabilities, value), labels=[0, 1, 2])
            for value in candidates
        ]
        return float(candidates[int(np.argmin(losses))])

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        values = _normalize(probabilities)
        targets = np.asarray(labels, dtype=int)
        if len(values) < 30:
            self.diagnostics_ = CalibrationDiagnostics("identity", math.nan, len(values), 0)
            return self
        split = max(20, min(len(values) - 12, int(len(values) * 0.70)))
        fit_values, validation_values = values[:split], values[split:]
        fit_labels, validation_labels = targets[:split], targets[split:]
        candidates: list[tuple[str, float]] = [
            ("identity", float(log_loss(validation_labels, validation_values, labels=[0, 1, 2]))),
        ]

        sigmoid_models = self._fit_sigmoid(fit_values, fit_labels)
        sigmoid_validation = self._transform_sigmoid(validation_values, sigmoid_models)
        candidates.append(("sigmoid", float(log_loss(validation_labels, sigmoid_validation, labels=[0, 1, 2]))))

        fit_temperature = self._best_temperature(fit_values, fit_labels)
        temperature_validation = _temperature_transform(validation_values, fit_temperature)
        candidates.append(("temperature", float(log_loss(validation_labels, temperature_validation, labels=[0, 1, 2]))))

        if len(values) >= self.minimum_isotonic_rows:
            isotonic_models = self._fit_isotonic(fit_values, fit_labels)
            isotonic_validation = self._transform_isotonic(validation_values, isotonic_models)
            candidates.append(("isotonic", float(log_loss(validation_labels, isotonic_validation, labels=[0, 1, 2]))))

        self.method_, selection_loss = min(candidates, key=lambda item: item[1])
        if self.method_ == "sigmoid":
            self.models_ = list(self._fit_sigmoid(values, targets))
        elif self.method_ == "isotonic":
            self.models_ = list(self._fit_isotonic(values, targets))
        elif self.method_ == "temperature":
            self.temperature_ = self._best_temperature(values, targets)
        self.diagnostics_ = CalibrationDiagnostics(self.method_, selection_loss, split, len(values) - split)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _normalize(probabilities)
        if self.method_ == "sigmoid":
            return self._transform_sigmoid(values, self.models_)
        if self.method_ == "isotonic":
            return self._transform_isotonic(values, self.models_)
        if self.method_ == "temperature":
            return _temperature_transform(values, self.temperature_)
        return values


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    values = _normalize(probabilities)
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    targets = np.asarray(labels, dtype=int)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            (confidence <= edges[index + 1]) if upper_inclusive else (confidence < edges[index + 1])
        )
        if not mask.any():
            continue
        accuracy = float((predicted[mask] == targets[mask]).mean())
        error += float(mask.mean()) * abs(accuracy - float(confidence[mask].mean()))
    return float(error)


def multiclass_diagnostics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = _normalize(probabilities)
    targets = np.asarray(labels, dtype=int)
    predicted = values.argmax(axis=1)
    return {
        "mcc": float(matthews_corrcoef(targets, predicted)),
        "log_loss": float(log_loss(targets, values, labels=[0, 1, 2])),
        "ece": expected_calibration_error(values, targets),
    }


def class_metric_rows(predictions: pd.DataFrame, lane: str) -> pd.DataFrame:
    calls = predictions[predictions["forecast"].isin(CLASS_NAMES)].copy()
    if calls.empty:
        return pd.DataFrame()
    actual = calls["actual_class"].astype(int).to_numpy()
    predicted = calls["forecast"].map({name: index for index, name in enumerate(CLASS_NAMES)}).astype(int).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return pd.DataFrame([
        {
            "lane": lane,
            "class": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    ])


def confusion_rows(predictions: pd.DataFrame, lane: str) -> pd.DataFrame:
    calls = predictions[predictions["forecast"].isin(CLASS_NAMES)].copy()
    if calls.empty:
        return pd.DataFrame()
    actual = calls["actual_class"].astype(int).to_numpy()
    predicted = calls["forecast"].map({name: index for index, name in enumerate(CLASS_NAMES)}).astype(int).to_numpy()
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])
    rows = []
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        total = int(matrix[actual_index].sum())
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            count = int(matrix[actual_index, predicted_index])
            rows.append({
                "lane": lane,
                "actual": actual_name,
                "predicted": predicted_name,
                "count": count,
                "row_rate": float(count / total) if total else math.nan,
            })
    return pd.DataFrame(rows)


def confidence_risk_rows(predictions: pd.DataFrame, lane: str) -> pd.DataFrame:
    calls = predictions[predictions["forecast"].isin(CLASS_NAMES)].copy()
    if calls.empty:
        return pd.DataFrame()
    calls["confidence"] = calls[["prob_down", "prob_sideway", "prob_up"]].max(axis=1)
    calls = calls.sort_values("confidence", ascending=False)
    rows = []
    for coverage in np.linspace(0.10, 1.0, 10):
        count = max(1, int(math.ceil(len(calls) * coverage)))
        selected = calls.head(count)
        rows.append({
            "lane": lane,
            "coverage": float(count / len(calls)),
            "minimum_confidence": float(selected["confidence"].min()),
            "exact_accuracy": float((selected["status"] == "correct").mean()),
            "weighted_accuracy": float(selected["score"].mean()),
            "expectancy": float(selected["strategy_return"].mean()),
        })
    return pd.DataFrame(rows)


def grouped_performance_rows(predictions: pd.DataFrame, frame: pd.DataFrame, lane: str) -> pd.DataFrame:
    calls = predictions[predictions["forecast"].isin(CLASS_NAMES)].copy()
    if calls.empty:
        return pd.DataFrame()
    context_columns = [column for column in ("date", "regime", "volatility_21") if column in frame]
    context = frame[context_columns].drop_duplicates("date", keep="last")
    calls = calls.merge(context, on="date", how="left")
    calls["year"] = calls["date"].dt.year.astype(str)
    calls["weekday"] = calls["date"].dt.day_name()
    regime = calls["regime"] if "regime" in calls else pd.Series("unknown", index=calls.index)
    calls["astro_regime"] = regime.fillna("unknown").astype(str)
    volatility = pd.to_numeric(calls.get("volatility_21", pd.Series(np.nan, index=calls.index)), errors="coerce")
    valid = volatility.dropna()
    if len(valid) >= 30:
        lower, upper = valid.quantile([1 / 3, 2 / 3])
        calls["volatility_regime"] = np.select(
            [volatility <= lower, volatility >= upper],
            ["low", "high"],
            default="mid",
        )
    else:
        calls["volatility_regime"] = "unknown"
    rows: list[dict[str, object]] = []
    for dimension in ("year", "weekday", "astro_regime", "volatility_regime"):
        for value, group in calls.groupby(dimension, dropna=False):
            rows.append({
                "lane": lane,
                "dimension": dimension,
                "value": str(value),
                "calls": int(len(group)),
                "exact_accuracy": float((group["status"] == "correct").mean()),
                "weighted_accuracy": float(group["score"].mean()),
                "directional_accuracy": float(group["directional_hit"].mean()),
                "expectancy": float(group["strategy_return"].mean()),
            })
    return pd.DataFrame(rows)


def moving_block_confidence_intervals(
    predictions: pd.DataFrame,
    samples: int = 240,
    block_size: int = 14,
    random_state: int = 42,
) -> dict[str, float]:
    calls = predictions[predictions["forecast"].isin(CLASS_NAMES)].reset_index(drop=True)
    if len(calls) < block_size * 2:
        return {
            "exact_lcb": math.nan,
            "exact_ucb": math.nan,
            "weighted_lcb": math.nan,
            "weighted_ucb": math.nan,
            "directional_lcb": math.nan,
            "directional_ucb": math.nan,
            "expectancy_lcb": math.nan,
            "expectancy_ucb": math.nan,
        }
    rng = np.random.default_rng(random_state)
    size = len(calls)
    exact_values = (calls["status"] == "correct").to_numpy(dtype=float)
    weighted_values = calls["score"].to_numpy(dtype=float)
    directional_values = calls["directional_hit"].to_numpy(dtype=float)
    expectancy_values = calls["strategy_return"].to_numpy(dtype=float)
    results = np.zeros((samples, 4), dtype=float)
    blocks_needed = int(math.ceil(size / block_size))
    for sample_index in range(samples):
        starts = rng.integers(0, size, size=blocks_needed)
        indices = np.concatenate([
            (np.arange(start, start + block_size) % size) for start in starts
        ])[:size]
        results[sample_index] = [
            exact_values[indices].mean(),
            weighted_values[indices].mean(),
            directional_values[indices].mean(),
            expectancy_values[indices].mean(),
        ]
    lower = np.quantile(results, 0.05, axis=0)
    upper = np.quantile(results, 0.95, axis=0)
    return {
        "exact_lcb": float(lower[0]),
        "exact_ucb": float(upper[0]),
        "weighted_lcb": float(lower[1]),
        "weighted_ucb": float(upper[1]),
        "directional_lcb": float(lower[2]),
        "directional_ucb": float(upper[2]),
        "expectancy_lcb": float(lower[3]),
        "expectancy_ucb": float(upper[3]),
    }


def conservative_beta_lower_bound(
    scores: Sequence[float],
    prior_mean: float,
    prior_strength: float = 20.0,
    z_score: float = 1.645,
) -> float:
    values = np.asarray(list(scores), dtype=float)
    alpha = max(1e-6, prior_mean * prior_strength + float(values.sum()))
    beta = max(1e-6, (1 - prior_mean) * prior_strength + float(len(values) - values.sum()))
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return float(max(0.0, mean - z_score * math.sqrt(variance)))


def _population_stability_index(reference: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    left = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(recent, errors="coerce").dropna().to_numpy(dtype=float)
    if len(left) < 30 or len(right) < 20:
        return math.nan
    edges = np.unique(np.quantile(left, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    left_hist = np.histogram(left, bins=edges)[0] / len(left)
    right_hist = np.histogram(right, bins=edges)[0] / len(right)
    left_hist = np.clip(left_hist, 1e-5, 1)
    right_hist = np.clip(right_hist, 1e-5, 1)
    return float(np.sum((right_hist - left_hist) * np.log(right_hist / left_hist)))


def feature_drift_rows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    recent_days: int = 90,
    reference_days: int = 365,
) -> pd.DataFrame:
    realized = frame[frame["daily_return"].notna()].sort_values("date")
    recent = realized.tail(recent_days)
    reference = realized.iloc[-(recent_days + reference_days):-recent_days]
    if recent.empty or reference.empty:
        return pd.DataFrame()
    rows = []
    for column in dict.fromkeys(feature_columns):
        if column not in realized:
            continue
        psi = _population_stability_index(reference[column], recent[column])
        reference_std = float(pd.to_numeric(reference[column], errors="coerce").std())
        mean_shift = (
            float((pd.to_numeric(recent[column], errors="coerce").mean() - pd.to_numeric(reference[column], errors="coerce").mean()) / reference_std)
            if np.isfinite(reference_std) and reference_std > 0 else math.nan
        )
        status = "insufficient" if not np.isfinite(psi) else "alert" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable"
        rows.append({
            "feature": column,
            "psi": psi,
            "mean_shift_z": mean_shift,
            "reference_missing": float(reference[column].isna().mean()),
            "recent_missing": float(recent[column].isna().mean()),
            "status": status,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def class_drift_rows(frame: pd.DataFrame, recent_days: int = 90, reference_days: int = 365) -> pd.DataFrame:
    realized = frame[frame["target"].notna()].sort_values("date")
    recent = realized.tail(recent_days)
    reference = realized.iloc[-(recent_days + reference_days):-recent_days]
    rows = []
    for class_index, name in enumerate(CLASS_NAMES):
        reference_share = float((reference["target"] == class_index).mean()) if len(reference) else math.nan
        recent_share = float((recent["target"] == class_index).mean()) if len(recent) else math.nan
        rows.append({
            "class": name,
            "reference_share": reference_share,
            "recent_share": recent_share,
            "change": recent_share - reference_share,
        })
    return pd.DataFrame(rows)


def page_hinkley_alarm(
    values: Sequence[float],
    delta: float = 0.005,
    threshold: float = 5.0,
    recent_window: int = 60,
) -> dict[str, object]:
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 20:
        return {
            "alarm": False,
            "action": "insufficient-evidence",
            "statistic": math.nan,
            "observations": int(len(series)),
        }
    mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    maximum_statistic = 0.0
    alarm_indices: list[int] = []
    for index, value in enumerate(series, start=1):
        mean += (value - mean) / index
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        maximum_statistic = max(maximum_statistic, cumulative - minimum)
        if cumulative - minimum >= threshold:
            alarm_indices.append(index - 1)
            cumulative = 0.0
            minimum = 0.0
    recent_count = min(recent_window, max(20, len(series) // 5))
    baseline = series[:-recent_count]
    recent = series[-recent_count:]
    baseline_mean = float(np.mean(baseline)) if len(baseline) else float(np.mean(series))
    recent_mean = float(np.mean(recent))
    last_alarm_index = alarm_indices[-1] if alarm_indices else None
    alarm_is_recent = last_alarm_index is not None and last_alarm_index >= len(series) - recent_count
    deterioration = recent_mean - baseline_mean
    alarm = bool(alarm_is_recent and deterioration >= 0.05)
    return {
        "alarm": alarm,
        "action": "suspend-execution" if alarm else "monitor",
        "statistic": float(maximum_statistic),
        "threshold": threshold,
        "observations": int(len(series)),
        "last_alarm_index": last_alarm_index,
        "recent_window": recent_count,
        "baseline_loss": baseline_mean,
        "recent_loss": recent_mean,
        "deterioration": deterioration,
    }
