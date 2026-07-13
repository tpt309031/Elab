from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from research.deep_dataset import build_intraday_sequence_dataset, chronological_folds
from research.deep_models import TorchSequenceClassifier
from research.train_deep_challengers import _metric_rows


def _bars(frequency: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range("2023-12-01", periods=periods, freq=frequency)
    baseline = 40_000 + np.arange(periods, dtype=float) * 2
    close = baseline + np.sin(np.arange(periods) / 5) * 20
    return pd.DataFrame({
        "timestamp": dates,
        "open": baseline,
        "high": np.maximum(baseline, close) + 25,
        "low": np.minimum(baseline, close) - 25,
        "close": close,
        "volume": 100 + np.arange(periods) % 17,
    })


def test_intraday_sequences_use_only_bars_closed_by_target_cutoff() -> None:
    one_hour = _bars("1h", 24 * 150)
    four_hour = _bars("4h", 6 * 150)
    daily_dates = pd.date_range("2024-01-01", periods=110, freq="D")
    daily = pd.DataFrame({
        "timestamp": daily_dates,
        "open": 40_000 + np.arange(len(daily_dates)) * 10,
        "close": 40_010 + np.arange(len(daily_dates)) * 10,
    })
    context = pd.DataFrame({
        "date": daily_dates,
        "index_BTC": np.linspace(20, 80, len(daily_dates)),
        "index_me": np.linspace(80, 20, len(daily_dates)),
    })
    dataset = build_intraday_sequence_dataset(
        one_hour,
        four_hour,
        daily,
        context=context,
        context_columns=("index_BTC", "index_me"),
        lookback=12,
        start_date="2024-01-01",
    )

    assert dataset.values.shape[1:] == (12, 12)
    assert (dataset.last_bar_closed_at <= dataset.dates).all()
    assert (dataset.last_bar_closed_at == dataset.dates).all()
    assert np.diff(dataset.dates).min() >= np.timedelta64(1, "D")


def test_deep_folds_keep_train_calibration_and_test_strictly_ordered() -> None:
    dates = pd.date_range("2022-01-01", periods=1_400, freq="D").to_numpy()
    folds = chronological_folds(dates, n_folds=2)
    assert len(folds) == 2
    for fold in folds:
        assert fold.train_index[-1] < fold.calibration_index[0]
        assert fold.calibration_index[-1] < fold.test_index[0]
        assert set(fold.train_index).isdisjoint(fold.test_index)


def test_deep_metric_summary_uses_live_compatible_conservative_bound() -> None:
    predictions = []
    for index in range(180):
        actual = ("down", "sideway", "up")[index % 3]
        predictions.append({
            "architecture": "LSTM",
            "date": f"2025-{index // 28 + 1:02d}-{index % 28 + 1:02d}",
            "actual_class": actual,
            "status": "correct" if index % 2 == 0 else "wrong",
            "score": 1.0 if index % 2 == 0 else 0.0,
            "strategy_return": 0.002 if index % 2 == 0 else -0.001,
            "directional_hit": index % 2 == 0,
            "prob_down": 0.6 if actual == "down" else 0.2,
            "prob_sideway": 0.6 if actual == "sideway" else 0.2,
            "prob_up": 0.6 if actual == "up" else 0.2,
        })
    metrics = _metric_rows(predictions)
    assert len(metrics) == 1
    assert 0 <= metrics[0]["directional_lcb"] <= metrics[0]["directional_accuracy"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is an optional deep dependency")
@pytest.mark.parametrize("architecture", ["lstm", "transformer", "tcn", "patchtst", "tft", "itransformer"])
def test_each_sequence_architecture_completes_a_training_pass(architecture: str) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(220, 12 * 4)).astype(np.float32)
    labels = np.tile(np.arange(3), 74)[:220]
    model = TorchSequenceClassifier(
        architecture=architecture,
        lookback=12,
        input_features=4,
        hidden_size=16,
        epochs=1,
        batch_size=64,
        random_state=11,
        device="cpu",
    )
    model.fit(values, labels)
    probabilities = model.predict_proba(values[-7:])
    assert probabilities.shape == (7, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
