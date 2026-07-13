from __future__ import annotations

import pandas as pd
import pytest

import research.data_sources as data_sources
from research.data_sources import build_intraday_daily_features, load_external_features, refresh_intraday_market


def _bars(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "open": [100.0] * len(timestamps),
        "high": [104.0] * len(timestamps),
        "low": [98.0] * len(timestamps),
        "close": [102.0] * len(timestamps),
        "volume": [10.0] * len(timestamps),
    })


def test_external_features_enforce_explicit_availability(tmp_path) -> None:
    directory = tmp_path / "external"
    directory.mkdir()
    pd.DataFrame({
        "date": ["2026-07-12", "2026-07-13"],
        "available_at": ["2026-07-11T23:00:00Z", "2026-07-13T12:00:00Z"],
        "funding_rate": [0.001, 0.002],
        "open_interest": [100.0, 120.0],
    }).to_csv(directory / "derivatives.csv", index=False)

    frame, health, lineage = load_external_features(directory)

    assert frame.loc[frame["date"] == pd.Timestamp("2026-07-12"), "funding_rate"].iloc[0] == 0.001
    assert pd.isna(frame.loc[frame["date"] == pd.Timestamp("2026-07-13"), "funding_rate"].iloc[0])
    derivatives = next(row for row in health if row["source"] == "derivatives")
    assert derivatives["usable_rows"] == 1
    assert lineage[0]["availability_mode"] == "explicit"


def test_intraday_daily_features_are_formed_from_closed_bars() -> None:
    one_hour = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-12 00:00", "2026-07-12 01:00"]),
        "open": [100.0, 102.0],
        "high": [103.0, 104.0],
        "low": [99.0, 101.0],
        "close": [102.0, 101.0],
        "volume": [10.0, 30.0],
    })

    result = build_intraday_daily_features([("1h", one_hour)])

    assert result.iloc[0]["date"] == pd.Timestamp("2026-07-12")
    assert result.iloc[0]["intraday_coverage_1h"] == pytest.approx(2 / 24)
    assert result.iloc[0]["intraday_trend_1h"] == pytest.approx(0.01)
    assert result.iloc[0]["intraday_signed_volume_1h"] == pytest.approx(-0.5)


def test_intraday_refresh_uses_next_endpoint_and_requires_latest_bar(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fetch(_start, _end, _timeframe, base_url):
        calls.append(base_url)
        if "data-api" in base_url:
            raise RuntimeError("blocked")
        return _bars(["2026-07-13 02:00:00"])

    monkeypatch.setattr(data_sources, "_fetch_binance_intraday_from_base", fetch)
    result, health = refresh_intraday_market(
        tmp_path / "BTCUSDT_1h.csv",
        "1h",
        now=pd.Timestamp("2026-07-13 03:20:00", tz="UTC"),
    )

    assert result["timestamp"].max() == pd.Timestamp("2026-07-13 02:00:00")
    assert health["stale"] is False
    assert len(calls) == 2
    assert health["attempts"][0]["status"] == "failed"
    assert health["attempts"][1]["status"] == "healthy"


def test_intraday_refresh_does_not_overwrite_stale_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "BTCUSDT_4h.csv"
    _bars(["2026-07-12 20:00:00"]).to_csv(cache, index=False)

    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(data_sources, "_fetch_binance_intraday_from_base", fail)
    with pytest.raises(RuntimeError, match="Closed 4h BTC bar"):
        refresh_intraday_market(
            cache,
            "4h",
            now=pd.Timestamp("2026-07-13 08:20:00", tz="UTC"),
        )
    persisted = pd.read_csv(cache)
    assert persisted["timestamp"].iloc[-1] == "2026-07-12 20:00:00"
