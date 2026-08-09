from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research.hybrid_core as hybrid_core

from research.hybrid_core import (
    allocate_monthly_directions,
    build_feature_frame,
    build_pattern_registry,
    grade_forecast,
    maximum_monthly_forecast_counts,
    monthly_purged_folds,
    reserved_forecast_dates,
)


def test_pattern_registry_learns_signal_lead_without_delayed_grades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-01", periods=160, freq="D")
    pulse = np.zeros(len(dates))
    returns = np.zeros(len(dates))
    for position in range(5, 145, 10):
        pulse[position] = 1
        returns[position + 2] = 0.04
    frame = pd.DataFrame({
        "date": dates,
        "pulse": pulse,
        "daily_return": returns,
        "target": [hybrid_core.direction_label(value) for value in returns],
    })
    monkeypatch.setattr(hybrid_core, "PATTERN_DEFINITIONS", {
        "pulse": ("pulse > 0", "Synthetic pulse"),
    })

    registry = build_pattern_registry(frame)
    lead_two = registry[
        (registry["pattern_id"] == "pulse__lead_2d")
        & (registry["direction"] == "up")
    ].iloc[0]
    same_day = registry[
        (registry["pattern_id"] == "pulse__lead_0d")
        & (registry["direction"] == "up")
    ].iloc[0]

    assert lead_two["weighted_accuracy"] > same_day["weighted_accuracy"]
    assert lead_two["signal_lag_days"] == 2
    assert lead_two["eligible"]


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


def test_market_features_respect_two_session_publication_lead() -> None:
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
    observed = frame.loc[frame["date"] == dates[82], "market_return_1"].iloc[0]
    assert np.isclose(observed, expected)


def test_forecast_policy_caps_sideway_at_eight_per_month() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    probabilities = np.tile(np.array([0.20, 0.60, 0.20]), (len(dates), 1))
    directions, _, _, overrides = allocate_monthly_directions(
        dates, probabilities, np.eye(3), policy_mode="probability",
    )
    monthly = pd.DataFrame({"date": dates, "forecast": directions})
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
    counts = monthly.groupby("month")["forecast"].apply(lambda values: (values == "sideway").sum())
    assert counts.max() == 8
    assert overrides.sum() == 24


def test_dynamic_no_call_is_limited_to_four_uncertain_days_per_month() -> None:
    dates = pd.date_range("2026-01-01", periods=31, freq="D")
    probabilities = np.tile(np.array([0.20, 0.60, 0.20]), (len(dates), 1))
    directions, expected_scores, margins, _ = allocate_monthly_directions(
        dates,
        probabilities,
        np.eye(3),
        policy_mode="probability",
        allow_no_call=True,
    )
    assert (directions == "sideway").sum() == 8
    assert (directions == "no-call").sum() == 4
    assert np.isnan(expected_scores[directions == "no-call"]).all()
    assert np.isnan(margins[directions == "no-call"]).all()


def test_no_call_capacity_respects_locked_forecasts() -> None:
    dates = pd.date_range("2026-07-13", periods=6, freq="D")
    probabilities = np.tile(np.array([0.34, 0.33, 0.33]), (len(dates), 1))
    directions, _, _, _ = allocate_monthly_directions(
        dates,
        probabilities,
        np.eye(3),
        policy_mode="probability",
        allow_no_call=True,
        existing_no_call_per_month={"2026-07": 3},
        excluded_dates={"2026-07-13"},
    )
    assert directions[0] != "no-call"
    assert (directions == "no-call").sum() == 1


def test_future_sideway_capacity_respects_locked_monthly_calls() -> None:
    dates = pd.date_range("2026-07-13", periods=8, freq="D")
    probabilities = np.tile(np.array([0.20, 0.60, 0.20]), (len(dates), 1))
    directions, _, _, overrides = allocate_monthly_directions(
        dates,
        probabilities,
        np.eye(3),
        policy_mode="probability",
        max_sideway_per_month=8,
        existing_sideway_per_month={"2026-07": 8},
        excluded_dates={"2026-07-13"},
    )
    assert directions[0] == "sideway"
    assert all(direction != "sideway" for direction in directions[1:])
    assert overrides[1:].all()


def test_monthly_capacity_uses_most_constrained_authoritative_lane() -> None:
    calendar = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=4, freq="D"),
        "forecast": ["no-call", "no-call", "up", "down"],
    })
    fusion = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=5, freq="D"),
        "forecast": ["no-call", "no-call", "no-call", "sideway", "up"],
    })
    assert maximum_monthly_forecast_counts([calendar, fusion], "no-call") == {"2026-07": 3}


def test_full_next_session_consumes_shared_fusion_sideway_capacity() -> None:
    calendar_locked = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=7, freq="D"),
        "forecast": ["sideway"] * 7,
    })
    full_locked = calendar_locked.copy()
    full_next = pd.DataFrame({"date": [pd.Timestamp("2026-07-15")], "forecast": ["sideway"]})
    fusion_capacity = pd.concat([full_locked, full_next], ignore_index=True)
    used = maximum_monthly_forecast_counts([calendar_locked, fusion_capacity], "sideway")
    reserved = reserved_forecast_dates([calendar_locked, full_locked])
    future_dates = pd.date_range("2026-07-15", periods=5, freq="D")
    probabilities = np.tile(np.array([0.20, 0.60, 0.20]), (len(future_dates), 1))
    directions, _, _, overrides = allocate_monthly_directions(
        future_dates,
        probabilities,
        np.eye(3),
        policy_mode="probability",
        max_sideway_per_month=8,
        existing_sideway_per_month=used,
        excluded_dates=reserved,
    )
    assert used == {"2026-07": 8}
    assert all(direction != "sideway" for direction in directions)
    assert overrides.all()


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


def test_okx_string_millisecond_timestamp_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = int(pd.Timestamp("2026-07-12", tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(hybrid_core, "_request_json", lambda _: {
        "data": [[str(timestamp), "100", "105", "98", "102", "10", "0", "0", "1"]],
    })
    result = hybrid_core._fetch_okx_daily(pd.Timestamp("2026-07-12"), pd.Timestamp("2026-07-13"))
    assert result.iloc[0]["timestamp"] == pd.Timestamp("2026-07-12")
    assert result.iloc[0]["close"] == 102.0


def test_stale_cache_fails_closed_when_all_providers_fail(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "BTCUSDT_1d.csv"
    _market_rows(["2026-07-11"]).to_csv(cache, index=False)

    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(hybrid_core, "_fetch_binance_daily", fail)
    monkeypatch.setattr(hybrid_core, "_fetch_okx_daily", fail)
    monkeypatch.setattr(hybrid_core, "_fetch_coinbase_daily", fail)
    with pytest.raises(RuntimeError, match="Closed BTC candle 2026-07-12 is unavailable"):
        hybrid_core.refresh_daily_market(cache, now=pd.Timestamp("2026-07-13 03:20:00", tz="UTC"))
    persisted = pd.read_csv(cache)
    assert persisted["timestamp"].iloc[-1] == "2026-07-11"
    assert not cache.with_suffix(".csv.tmp").exists()


def test_refresh_falls_back_and_requires_expected_closed_candle(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "BTCUSDT_1d.csv"
    _market_rows(["2026-07-11"]).to_csv(cache, index=False)

    def fail_binance(*_args, **_kwargs):
        raise RuntimeError("geo blocked")

    monkeypatch.setattr(hybrid_core, "_fetch_binance_daily", fail_binance)
    monkeypatch.setattr(hybrid_core, "_fetch_okx_daily", lambda *_: _market_rows(["2026-07-12"]))
    monkeypatch.setattr(
        hybrid_core,
        "_fetch_coinbase_daily",
        lambda *_: pytest.fail("Coinbase should not run after a valid OKX candle"),
    )
    result, provider = hybrid_core.refresh_daily_market(
        cache, now=pd.Timestamp("2026-07-13 03:20:00", tz="UTC"),
    )
    assert provider == "OKX"
    assert result["timestamp"].max() == pd.Timestamp("2026-07-12")


def _market_rows(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(dates),
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 10.0,
    })
