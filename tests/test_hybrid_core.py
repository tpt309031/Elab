from __future__ import annotations

import numpy as np
import pandas as pd

from research.hybrid_core import (
    allocate_monthly_directions,
    build_feature_frame,
    grade_forecast,
    monthly_purged_folds,
)


def test_exact_scoring_boundaries() -> None:
    assert grade_forecast("up", 0.03) == ("correct", 1.0)
    assert grade_forecast("up", 0.001) == ("partial", 0.5)
    assert grade_forecast("up", 0.0) == ("wrong", 0.0)
    assert grade_forecast("up", -0.14) == ("wrong", 0.0)
    assert grade_forecast("down", -0.03) == ("correct", 1.0)
    assert grade_forecast("down", -0.001) == ("partial", 0.5)
    assert grade_forecast("down", 0.0) == ("wrong", 0.0)
    assert grade_forecast("sideway", -0.01) == ("correct", 1.0)
    assert grade_forecast("sideway", 0.01) == ("correct", 1.0)
    assert grade_forecast("sideway", 0.0101) == ("wrong", 0.0)


def test_market_features_are_shifted_one_closed_candle() -> None:
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 160, len(dates)))
    market = pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.linspace(1000, 2000, len(dates)),
    })
    indices = pd.DataFrame({"date": dates, "index_BTC": 50.0, "index_me": 50.0})
    astro = pd.DataFrame({
        "date": dates,
        "finance": 50.0,
        "career": 50.0,
        "volatility": 50.0,
        "composite": 50.0,
        "event": False,
        "regime": "normal",
    })
    frame, _ = build_feature_frame(indices, market, astro)
    expected = close.pct_change().iloc[80]
    observed = frame.loc[frame["date"] == dates[81], "market_return_1"].iloc[0]
    assert np.isclose(observed, expected)


def test_monthly_sideway_calls_are_capped_at_eight() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    probabilities = np.tile(np.array([0.20, 0.60, 0.20]), (len(dates), 1))
    directions, _, _, overrides = allocate_monthly_directions(
        dates, probabilities, np.eye(3), policy_mode="probability",
    )
    result = pd.DataFrame({"date": dates, "forecast": directions, "override": overrides})
    monthly_sideway = result.assign(month=result["date"].dt.strftime("%Y-%m")).groupby("month")["forecast"].apply(
        lambda values: int((values == "sideway").sum())
    )
    assert (monthly_sideway <= 8).all()
    assert monthly_sideway.loc["2026-01"] == 8
    assert result["override"].any()


def test_monthly_folds_are_purged_and_chronological() -> None:
    dates = pd.date_range("2019-01-01", "2024-04-30", freq="D")
    frame = pd.DataFrame({"date": dates, "target": 1.0, "daily_return": 0.0})
    folds = monthly_purged_folds(frame, calibration_days=60, rolling_days=1000)
    assert folds
    for fold in folds:
        train_end = frame.loc[fold.train_index, "date"].max()
        calibration_start = frame.loc[fold.calibration_index, "date"].min()
        calibration_end = frame.loc[fold.calibration_index, "date"].max()
        test_start = frame.loc[fold.test_index, "date"].min()
        assert train_end < calibration_start
        assert calibration_end <= test_start - pd.Timedelta(days=5)
