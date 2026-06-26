from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
TIMEFRAME_CONFIG = {
    "5m": {"pivot_n": 24, "periods_per_day": 288},
    "1h": {"pivot_n": 24, "periods_per_day": 24},
    "4h": {"pivot_n": 12, "periods_per_day": 6},
    "1d": {"pivot_n": 5, "periods_per_day": 1},
    "1w": {"pivot_n": 3, "periods_per_day": 1 / 7},
    "1M": {"pivot_n": 2, "periods_per_day": 1 / 30},
}
TIMEFRAME_DELTAS = {
    "5m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
FEATURE_COLUMNS = [
    "index_BTC", "index_me", "index_BTC_change_1", "index_BTC_change_3",
    "index_BTC_slope_3", "index_BTC_slope_7", "index_BTC_zscore_30",
    "index_me_change_1", "index_me_slope_3", "gap_index", "gap_index_change",
    "close_return_1", "close_return_3", "close_return_7", "volatility_7",
    "distance_from_MA20", "distance_from_MA50", "RSI14", "ATR14",
]


UP_1D_THRESHOLD = 0.03
DOWN_1D_THRESHOLD = -0.03
SIDEWAY_1D_THRESHOLD = 0.01
PARTIAL_SIDEWAY_1D_THRESHOLD = 0.02


def classify_btc_outcome(return_3d: float | None, range_2d: float | None = None, day_return: float | None = None) -> str:
    one_day = return_3d if day_return is None else day_return
    if one_day is not None and one_day >= UP_1D_THRESHOLD:
        return "up"
    if one_day is not None and one_day <= DOWN_1D_THRESHOLD:
        return "down"
    if one_day is not None and abs(one_day) <= SIDEWAY_1D_THRESHOLD:
        return "sideway"
    return "mixed"


def partial_score_for_forecast(forecast: str, return_3d: float | None, day_return: float | None) -> float:
    if forecast == "up":
        return 0.5 if day_return is not None and day_return > 0 else 0.0
    if forecast == "down":
        return 0.5 if day_return is not None and day_return < 0 else 0.0
    if forecast == "sideway":
        return 0.5 if day_return is not None and abs(day_return) <= PARTIAL_SIDEWAY_1D_THRESHOLD else 0.0
    return 0.0


def realized_btc_outcome(market: pd.DataFrame, date) -> tuple[str | None, float | None, float | None, int, bool, float | None]:
    """Classify closed BTC movement after a signal date.

    Directional and sideway outcomes use the signal day's closed daily candle.
    Future 3D values are still returned for context, but they are not required
    for grading.
    """
    date = pd.Timestamp(date).normalize()
    if date not in market.index:
        return None, None, None, 0, False, None
    future = market.loc[market.index > date].head(3)
    base_row = market.loc[date]
    day_return = float(base_row["close"] / base_row["open"] - 1)
    if len(future) >= 2:
        horizon = min(3, len(future))
        base = float(base_row["close"])
        realized_return = float(future.iloc[horizon - 1]["close"] / base - 1)
        first_two = future.head(2)
        realized_range = float(first_two["high"].max() / first_two["low"].min() - 1)
    elif len(future) == 1:
        horizon = 1
        base = float(base_row["close"])
        realized_return = float(future.iloc[0]["close"] / base - 1)
        realized_range = float(future.iloc[0]["high"] / future.iloc[0]["low"] - 1)
    else:
        horizon = 0
        realized_return = None
        realized_range = None
    return classify_btc_outcome(realized_return, realized_range, day_return), realized_return, realized_range, horizon, True, day_return


def _month_number(value) -> int | None:
    key = str(value).strip().lower()[:3]
    return MONTHS.get(key)


def read_private_indices(path: str | Path, fallback_first_year: int = 2025) -> pd.DataFrame:
    """Parse monthly horizontal blocks, including multiple vertical year blocks."""
    raw = pd.read_excel(path, sheet_name="Index", header=None)
    starts = [
        i for i in range(len(raw))
        if sum(bool(_month_number(raw.iloc[i, c])) for c in range(raw.shape[1])) >= 6
    ]
    records: list[dict] = []
    previous_year = fallback_first_year - 1
    for block_no, start in enumerate(starts):
        marker_values = raw.iloc[max(0, start - 6):start, 0].dropna().tolist()
        explicit_years = [
            int(v) for v in marker_values
            if isinstance(v, (int, float, np.integer, np.floating))
            and 2000 <= int(v) <= 2100
        ]
        year = explicit_years[-1] if explicit_years else previous_year + 1
        previous_year = year
        for c in range(raw.shape[1] - 3):
            month = _month_number(raw.iloc[start, c])
            if not month:
                continue
            for r in range(start, min(start + 31, len(raw))):
                day, index_me, index_btc = raw.iloc[r, c + 1:c + 4]
                if pd.isna(day) or pd.isna(index_me) or pd.isna(index_btc):
                    continue
                try:
                    date = pd.Timestamp(year=year, month=month, day=int(day))
                except ValueError:
                    continue
                records.append({
                    "date": date, "index_me": float(index_me),
                    "index_BTC": float(index_btc), "source_block": block_no + 1,
                })
    result = pd.DataFrame(records).drop_duplicates("date", keep="last").sort_values("date")
    if result.empty:
        raise ValueError("No index records found in the workbook.")
    return result.reset_index(drop=True)


def read_private_indices_csv(
    index_btc_path: str | Path, index_me_path: str | Path,
) -> pd.DataFrame:
    btc = pd.read_csv(index_btc_path).rename(columns={"score_percent": "index_BTC"})
    me = pd.read_csv(index_me_path).rename(columns={"score_percent": "index_me"})
    required_btc = {"date", "index_BTC"}
    required_me = {"date", "index_me"}
    if not required_btc.issubset(btc.columns) or not required_me.issubset(me.columns):
        raise ValueError("CSV source requires columns: date, score_percent.")
    for frame in [btc, me]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if btc["date"].duplicated().any() or me["date"].duplicated().any():
        raise ValueError("CSV source contains duplicate dates.")
    result = me[["date", "index_me"]].merge(
        btc[["date", "index_BTC"]], on="date", how="outer", validate="one_to_one"
    ).sort_values("date")
    if result[["index_me", "index_BTC"]].isna().any().any():
        raise ValueError("CSV source has missing dates or missing index values.")
    if not result[["index_me", "index_BTC"]].apply(lambda column: column.between(0, 100)).all().all():
        raise ValueError("CSV index values must stay within 0-100.")
    result["source_block"] = "newdata_csv"
    return result.reset_index(drop=True)


def merge_index_rows(indices: pd.DataFrame, rows: pd.DataFrame | None) -> pd.DataFrame:
    if rows is None or rows.empty:
        return indices
    overrides = rows.copy()
    overrides["date"] = pd.to_datetime(overrides["date"]).dt.normalize()
    base = indices.set_index("date").copy()
    update = overrides.set_index("date")[["index_me", "index_BTC"]]
    base.update(update)
    missing = update.index.difference(base.index)
    if len(missing):
        added = update.loc[missing].copy()
        added["source_block"] = "manual"
        base = pd.concat([base, added])
    return base.reset_index().sort_values("date").reset_index(drop=True)


def apply_index_overrides(indices: pd.DataFrame, override_path: str | Path = "data/index_overrides.csv") -> pd.DataFrame:
    override_path = Path(override_path)
    if not override_path.exists():
        return indices
    overrides = pd.read_csv(override_path, parse_dates=["date"])
    return merge_index_rows(indices, overrides)


def save_index_overrides(rows: pd.DataFrame, override_path: str | Path = "data/index_overrides.csv") -> None:
    override_path = Path(override_path)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    clean = rows[["date", "index_me", "index_BTC"]].copy()
    clean["date"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m-%d")
    clean[["index_me", "index_BTC"]] = clean[["index_me", "index_BTC"]].apply(pd.to_numeric, errors="raise")
    if not clean[["index_me", "index_BTC"]].apply(lambda column: column.between(0, 100)).all().all():
        raise ValueError("index_me and index_BTC must stay within 0-100.")
    if override_path.exists():
        old = pd.read_csv(override_path)
        clean = pd.concat([old, clean], ignore_index=True).drop_duplicates("date", keep="last")
    clean.sort_values("date").to_csv(override_path, index=False)


def reset_index_overrides(override_path: str | Path = "data/index_overrides.csv") -> None:
    override_path = Path(override_path)
    if override_path.exists():
        override_path.unlink()


def _http_json(url: str, timeout: int = 15):
    request = urllib.request.Request(url, headers={"User-Agent": "btc-index-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _to_millis(ts) -> int:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _normalize_ohlcv(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Exchange returned no OHLCV candles.")
    timestamp = frame["timestamp"]
    numeric_timestamp = pd.to_numeric(timestamp, errors="coerce")
    timestamp_objects = timestamp.map(lambda value: isinstance(value, (pd.Timestamp, np.datetime64)))
    if numeric_timestamp.notna().all() and not timestamp_objects.any():
        frame["timestamp"] = pd.to_datetime(numeric_timestamp, unit="ms", utc=True).dt.tz_convert(None)
    else:
        frame["timestamp"] = pd.to_datetime(timestamp, utc=True).dt.tz_convert(None)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates(
        "timestamp", keep="last"
    ).sort_values("timestamp").reset_index(drop=True)


def _candle_close_time(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if timeframe in TIMEFRAME_DELTAS:
        return timestamp + TIMEFRAME_DELTAS[timeframe]
    if timeframe == "1w":
        return timestamp + pd.Timedelta(days=7)
    if timeframe == "1M":
        return timestamp + pd.offsets.MonthBegin(1)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def closed_ohlcv(data: pd.DataFrame, timeframe: str, now=None) -> pd.DataFrame:
    """Keep only fully closed candles so realtime research never uses partial bars."""
    now = pd.Timestamp(now or pd.Timestamp.now("UTC")).tz_localize(None)
    frame = data.copy()
    close_times = frame["timestamp"].map(lambda timestamp: _candle_close_time(timestamp, timeframe))
    return frame[close_times <= now].reset_index(drop=True)


def _expected_last_closed_open(timeframe: str, now=None) -> pd.Timestamp:
    now = pd.Timestamp(now or pd.Timestamp.now("UTC")).tz_localize(None)
    if timeframe == "5m":
        return now.floor("5min") - pd.Timedelta(minutes=5)
    if timeframe == "1h":
        return now.floor("h") - pd.Timedelta(hours=1)
    if timeframe == "4h":
        return now.floor("4h") - pd.Timedelta(hours=4)
    if timeframe == "1d":
        return now.normalize() - pd.Timedelta(days=1)
    if timeframe == "1w":
        return now.normalize() - pd.Timedelta(days=now.weekday() + 7)
    if timeframe == "1M":
        return now.normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def fetch_binance(timeframe: str, start, end=None) -> pd.DataFrame:
    interval = {"5m": "5m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w", "1M": "1M"}[timeframe]
    start_ms = _to_millis(start)
    end_ms = _to_millis(end or pd.Timestamp.now("UTC"))
    rows = []
    while start_ms < end_ms:
        query = urllib.parse.urlencode({
            "symbol": "BTCUSDT", "interval": interval, "startTime": start_ms,
            "endTime": end_ms, "limit": 1000,
        })
        payload = _http_json(f"https://api.binance.com/api/v3/klines?{query}")
        if not payload:
            break
        rows.extend({
            "timestamp": item[0], "open": item[1], "high": item[2],
            "low": item[3], "close": item[4], "volume": item[5],
        } for item in payload)
        next_start = int(payload[-1][0]) + 1
        if next_start <= start_ms:
            break
        start_ms = next_start
        time.sleep(0.05)
    return _normalize_ohlcv(rows)


def fetch_okx(timeframe: str, start, end=None) -> pd.DataFrame:
    bar = {"5m": "5m", "1h": "1H", "4h": "4H", "1d": "1Dutc", "1w": "1Wutc", "1M": "1Mutc"}[timeframe]
    start_ms = _to_millis(start)
    cursor = _to_millis(end or pd.Timestamp.now("UTC"))
    rows = []
    while cursor > start_ms:
        query = urllib.parse.urlencode({"instId": "BTC-USDT", "bar": bar, "after": cursor, "limit": 300})
        payload = _http_json(f"https://www.okx.com/api/v5/market/history-candles?{query}")
        batch = payload.get("data", [])
        if not batch:
            break
        rows.extend({
            "timestamp": item[0], "open": item[1], "high": item[2],
            "low": item[3], "close": item[4], "volume": item[5],
        } for item in batch if int(item[0]) >= start_ms)
        next_cursor = min(int(item[0]) for item in batch) - 1
        if next_cursor >= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    return _normalize_ohlcv(rows)


def load_ohlcv(
    timeframe: str, start, end=None, cache_dir: str | Path = "data/cache",
    refresh: bool = False, auto_refresh: bool = True,
) -> tuple[pd.DataFrame, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"BTCUSDT_{timeframe}.csv"
    cached = None
    if cache_file.exists():
        try:
            cached = closed_ohlcv(_normalize_ohlcv(pd.read_csv(cache_file).to_dict("records")), timeframe)
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            cached = None
        cache_fresh = cached is not None and not cached.empty and cached["timestamp"].max() >= _expected_last_closed_open(timeframe)
        if not refresh and (not auto_refresh or cache_fresh):
            cached.to_csv(cache_file, index=False)
            return cached, "cache"
    errors = []
    for provider, fetcher in [("Binance", fetch_binance), ("OKX", fetch_okx)]:
        try:
            fetch_start = start
            if cached is not None and not cached.empty and not refresh:
                fetch_start = cached["timestamp"].max() - pd.Timedelta(days=3)
            data = closed_ohlcv(fetcher(timeframe, fetch_start, end), timeframe)
            if cached is not None and not cached.empty:
                data = _normalize_ohlcv(pd.concat([cached, data], ignore_index=True).to_dict("records"))
                data = closed_ohlcv(data, timeframe)
            data.to_csv(cache_file, index=False)
            return data, f"{provider} auto-refresh"
        except Exception as exc:  # exchange availability differs by region
            errors.append(f"{provider}: {exc}")
    if cache_file.exists():
        return cached, "stale cache"
    raise RuntimeError("OHLCV download failed. " + " | ".join(errors))


def _slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    return series.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_dataset(indices: pd.DataFrame, ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_CONFIG[timeframe]
    data = ohlcv.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.normalize()
    idx = indices.copy()
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    data = data.merge(idx[["date", "index_BTC", "index_me"]], how="left", on="date")
    data[["index_BTC", "index_me"]] = data[["index_BTC", "index_me"]].ffill()
    for col in ["index_BTC", "index_me"]:
        data[f"{col}_change_1"] = data[col].diff(1)
    data["index_BTC_change_3"] = data["index_BTC"].diff(3)
    data["index_BTC_slope_3"] = _slope(data["index_BTC"], 3)
    data["index_BTC_slope_7"] = _slope(data["index_BTC"], 7)
    data["index_BTC_zscore_30"] = (
        (data["index_BTC"] - data["index_BTC"].rolling(30).mean())
        / data["index_BTC"].rolling(30).std()
    )
    data["index_me_slope_3"] = _slope(data["index_me"], 3)
    data["gap_index"] = data["index_BTC"] - data["index_me"]
    data["gap_index_change"] = data["gap_index"].diff()
    for n in [1, 3, 7]:
        data[f"close_return_{n}"] = data["close"].pct_change(n)
    data["volatility_7"] = data["close_return_1"].rolling(7).std()
    data["distance_from_MA20"] = data["close"] / data["close"].rolling(20).mean() - 1
    data["distance_from_MA50"] = data["close"] / data["close"].rolling(50).mean() - 1
    data["RSI14"] = _rsi(data["close"])
    true_range = pd.concat([
        data["high"] - data["low"],
        (data["high"] - data["close"].shift()).abs(),
        (data["low"] - data["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    data["ATR14"] = true_range.rolling(14).mean()
    n = cfg["pivot_n"]
    centered_max = data["close"].rolling(2 * n + 1, center=True).max()
    centered_min = data["close"].rolling(2 * n + 1, center=True).min()
    data["pivot_high"] = (data["close"] == centered_max).astype(float).where(centered_max.notna())
    data["pivot_low"] = (data["close"] == centered_min).astype(float).where(centered_min.notna())
    periods_per_day = cfg["periods_per_day"]
    for days in [1, 3, 7]:
        periods = max(1, int(round(days * periods_per_day)))
        data[f"future_return_{days}d"] = data["close"].shift(-periods) / data["close"] - 1
    data["target_up_3d"] = (data["future_return_3d"] > 0).astype(float).where(data["future_return_3d"].notna())
    data["target_down_3d"] = (data["future_return_3d"] < 0).astype(float).where(data["future_return_3d"].notna())
    data["target_pivot_high_next_N"] = data["pivot_high"].shift(-1).rolling(n).max().shift(-(n - 1))
    data["target_pivot_low_next_N"] = data["pivot_low"].shift(-1).rolling(n).max().shift(-(n - 1))
    data["weekday"] = pd.to_datetime(data["timestamp"]).dt.day_name()
    return data


def _probability_table(data: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    work = data.assign(zone=group).dropna(subset=["zone"])
    return work.groupby("zone", observed=False).agg(
        observations=("close", "size"),
        pivot_high_next_N=("target_pivot_high_next_N", "mean"),
        pivot_low_next_N=("target_pivot_low_next_N", "mean"),
        mean_return_3d=("future_return_3d", "mean"),
        mean_return_7d=("future_return_7d", "mean"),
    ).reset_index()


def analysis_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bins = [-0.1, 20, 40, 60, 80, 100.1]
    zones = pd.cut(data["index_BTC"], bins=bins, labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
    weekday = data.groupby("weekday").agg(
        observations=("close", "size"),
        mean_return_3d=("future_return_3d", "mean"),
        mean_return_7d=("future_return_7d", "mean"),
        pivot_high_next_N=("target_pivot_high_next_N", "mean"),
        pivot_low_next_N=("target_pivot_low_next_N", "mean"),
    ).reset_index()
    divergence = data.assign(
        btc_index_peak=data["index_BTC"].eq(data["index_BTC"].rolling(7).max()),
        price_not_peak=data["close"].lt(data["close"].rolling(7).max()),
        reversal_down=data["future_return_3d"].lt(0),
    )
    divergence = divergence[divergence["btc_index_peak"] & divergence["price_not_peak"]]
    gap = data[data["index_me"] - data["index_BTC"] >= 20]
    trap = data.assign(
        trap_setup=(data["index_me_change_1"] >= 15) & (data["close_return_3"] > 0.03),
        trap_result=data["future_return_3d"] < 0,
    )
    trap = trap[trap["trap_setup"]]
    headline = pd.DataFrame([
        {"pattern": "index_BTC peak while price is not peak", "observations": len(divergence),
         "probability": divergence["reversal_down"].mean()},
        {"pattern": "index_me exceeds index_BTC by >= 20", "observations": len(gap),
         "probability": gap["future_return_3d"].lt(0).mean()},
        {"pattern": "index_me jumps >= 15 after price rises > 3%", "observations": len(trap),
         "probability": trap["trap_result"].mean()},
    ])
    gap_summary = pd.DataFrame([{
        "observations": len(gap), "mean_return_3d": gap["future_return_3d"].mean(),
        "mean_return_7d": gap["future_return_7d"].mean(),
        "probability_down_3d": gap["future_return_3d"].lt(0).mean(),
    }])
    return {"index_zones": _probability_table(data, zones), "weekday": weekday,
            "patterns": headline, "gap": gap_summary}


def pattern_signals(data: pd.DataFrame, opposite_window: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect index turning-point setups on one row per calendar day.

    The private indices are daily, so calculating these signals on intraday rows
    would over-count the same observation. A sharp drop means an index rose by
    at least 12 points over the previous three days and then fell by at least 12.
    """
    daily = data.dropna(subset=["index_BTC", "index_me"]).copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values("timestamp").drop_duplicates("date", keep="last").set_index("date")
    for column in ["index_BTC", "index_me"]:
        daily[f"{column}_rise_3d"] = daily[column].shift(1) - daily[column].shift(4)
        daily[f"{column}_drop_1d"] = daily[column].diff()
        daily[f"{column}_drop_after_rise"] = (
            (daily[f"{column}_rise_3d"] >= 12) & (daily[f"{column}_drop_1d"] <= -12)
        )
    btc_event = daily["index_BTC_drop_after_rise"]
    me_event = daily["index_me_drop_after_rise"]
    aligned = pd.Series(False, index=daily.index)
    lead_lag = pd.Series(np.nan, index=daily.index, dtype=float)
    for lag in [0, -1, 1, -2, 2]:
        match = btc_event & me_event.shift(lag, fill_value=False) & ~aligned
        aligned |= match
        lead_lag.loc[match] = -lag
    daily["aligned_drop_after_rise_pm2d"] = aligned
    daily["aligned_drop_lead_lag_days"] = lead_lag
    btc_direction = np.sign(daily["index_BTC"].diff())
    me_direction = np.sign(daily["index_me"].diff())
    daily["opposite_phase_day"] = (btc_direction * me_direction < 0)
    daily["opposite_phase_ratio_5d"] = daily["opposite_phase_day"].rolling(opposite_window).mean()
    daily["opposite_phase_regime"] = daily["opposite_phase_ratio_5d"] >= 0.6
    daily["opposite_phase_start"] = daily["opposite_phase_regime"] & ~daily["opposite_phase_regime"].shift(
        fill_value=False
    )
    daily["index_BTC_peak_price_not_peak"] = (
        daily["index_BTC"].eq(daily["index_BTC"].rolling(7).max())
        & daily["close"].lt(daily["close"].rolling(7).max())
    )
    daily["late_index_me_trap"] = (daily["index_me"].diff() >= 15) & (daily["close"].pct_change(3) > 0.03)

    definitions = [
        ("index_BTC drop after 3-day rise", "index_BTC_drop_after_rise"),
        ("index_me drop after 3-day rise", "index_me_drop_after_rise"),
        ("both indices drop after rise within +/-2 days", "aligned_drop_after_rise_pm2d"),
        ("sustained opposite phase starts", "opposite_phase_start"),
        ("index_BTC peaks while price is not at a 7-day high", "index_BTC_peak_price_not_peak"),
        ("index_me jumps >= 15 after BTC rises > 3% in 3 days", "late_index_me_trap"),
    ]
    rows = []
    for label, column in definitions:
        sample = daily[daily[column]]
        rows.append({
            "pattern": label, "signal_column": column, "observations": len(sample),
            "probability_down_3d": sample["future_return_3d"].lt(0).mean(),
            "mean_return_3d": sample["future_return_3d"].mean(),
            "mean_return_7d": sample["future_return_7d"].mean(),
            "probability_pivot_high_next_N": sample["target_pivot_high_next_N"].mean(),
        })
    summary = pd.DataFrame(rows)

    episodes = []
    regime_group = (daily["opposite_phase_regime"] != daily["opposite_phase_regime"].shift()).cumsum()
    for _, episode in daily[daily["opposite_phase_regime"]].groupby(regime_group):
        episodes.append({
            "start": episode.index.min(), "end": episode.index.max(), "days": len(episode),
            "index_BTC_change": episode["index_BTC"].iloc[-1] - episode["index_BTC"].iloc[0],
            "index_me_change": episode["index_me"].iloc[-1] - episode["index_me"].iloc[0],
            "BTC_return_during_regime": episode["close"].iloc[-1] / episode["close"].iloc[0] - 1,
            "BTC_return_next_3d": episode["future_return_3d"].iloc[-1],
            "BTC_return_next_7d": episode["future_return_7d"].iloc[-1],
        })
    return daily.reset_index(), summary, pd.DataFrame(episodes)


def future_pattern_forecasts(
    indices: pd.DataFrame, dataset: pd.DataFrame, min_similarity: float = 0.72, top_matches: int = 8,
) -> pd.DataFrame:
    """Compare future index shapes with historical 3-day index patterns that have known price outcomes."""
    idx = indices[["date", "index_BTC", "index_me"]].dropna().copy().sort_values("date")
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    known = dataset.dropna(subset=["future_return_7d", "index_BTC", "index_me"]).copy()
    known["date"] = pd.to_datetime(known["date"]).dt.normalize()
    known = known.sort_values("timestamp").drop_duplicates("date", keep="last").set_index("date")
    price_end = pd.to_datetime(dataset["timestamp"]).max().normalize()
    if len(known) < 10:
        return pd.DataFrame()

    def signature(frame: pd.DataFrame) -> np.ndarray:
        values = frame[["index_BTC", "index_me"]].to_numpy(dtype=float)
        return np.concatenate([values[-1] / 100, np.diff(values, axis=0).reshape(-1) / 100])

    historical = []
    for date in known.index:
        history = idx[idx["date"] <= date].tail(3)
        if len(history) < 3:
            continue
        historical.append((date, signature(history), known.loc[date]))
    forecasts = []
    for _, current in idx[idx["date"] > price_end].iterrows():
        history = idx[idx["date"] <= current["date"]].tail(3)
        if len(history) < 3:
            continue
        vector = signature(history)
        matches = []
        for past_date, past_vector, outcome in historical:
            distance = float(np.linalg.norm(vector - past_vector))
            similarity = max(0.0, 1 - distance / 1.25)
            if similarity >= min_similarity:
                matches.append((similarity, past_date, outcome))
        matches = sorted(matches, key=lambda item: item[0], reverse=True)[:top_matches]
        if not matches:
            continue
        returns_1d = pd.Series([float(item[2]["close"] / item[2]["open"] - 1) for item in matches], dtype=float)
        returns_3d = pd.Series([item[2]["future_return_3d"] for item in matches], dtype=float)
        returns_7d = pd.Series([item[2]["future_return_7d"] for item in matches], dtype=float)
        median_3d = returns_3d.median()
        up = (returns_1d >= UP_1D_THRESHOLD).mean()
        down = (returns_1d <= DOWN_1D_THRESHOLD).mean()
        sideway = (returns_1d.abs() <= SIDEWAY_1D_THRESHOLD).mean()
        direction = max({"up": up, "down": down, "sideway": sideway}, key={"up": up, "down": down, "sideway": sideway}.get)
        confidence = max(up, down, sideway)
        trap = bool(current["index_me"] - current["index_BTC"] >= 20)
        forecasts.append({
            "date": current["date"], "forecast": direction, "confidence": confidence,
            "matches": len(matches), "mean_similarity": np.mean([item[0] for item in matches]),
            "median_return_1d": returns_1d.median(), "median_return_3d": median_3d, "median_return_7d": returns_7d.median(),
            "trap_warning": trap, "similar_dates": ", ".join(item[1].strftime("%d.%m.%y") for item in matches),
        })
    return pd.DataFrame(forecasts)


def scenario_pattern_forecasts(
    indices: pd.DataFrame, dataset: pd.DataFrame, min_similarity: float = 0.72, top_matches: int = 8,
) -> pd.DataFrame:
    """Build walk-forward historical and future scenarios with realized grading.

    Outcome convention:
    - up: same-day open-to-close return >= +3%
    - down: same-day open-to-close return <= -3%
    - sideway: same-day open-to-close return within +/-1%
    - mixed: none of the above
    """
    idx = indices[["date", "index_BTC", "index_me"]].dropna().copy().sort_values("date")
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    market = dataset.copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market.sort_values("timestamp").drop_duplicates("date", keep="last").set_index("date")
    price_end = market.index.max()

    def signature(frame: pd.DataFrame) -> np.ndarray:
        values = frame[["index_BTC", "index_me"]].to_numpy(dtype=float)
        return np.concatenate([values[-1] / 100, np.diff(values, axis=0).reshape(-1) / 100])

    def realized(date: pd.Timestamp) -> tuple[str | None, float | None, float | None, int, bool, float | None]:
        return realized_btc_outcome(market, date)

    candidates = []
    for _, current in idx.iterrows():
        date = current["date"]
        history = idx[idx["date"] <= date].tail(3)
        if len(history) < 3:
            continue
        outcome, return_3d, range_2d, horizon, is_final, day_return = realized(date)
        actual_final_for_outcome = outcome in {"up", "down", "sideway"}
        candidates.append({
            "date": date, "vector": signature(history), "actual": outcome,
            "actual_return_3d": return_3d, "actual_range_2d": range_2d,
            "actual_day_return": day_return, "actual_horizon": horizon, "actual_is_final": is_final,
            "actual_final_for_outcome": actual_final_for_outcome,
        })
    if not candidates:
        return pd.DataFrame()

    candidate_dates = pd.Series([item["date"] for item in candidates])
    candidate_date_values = candidate_dates.to_numpy(dtype="datetime64[ns]")
    vectors = np.vstack([item["vector"] for item in candidates]).astype(float)
    actuals = np.array([item["actual"] for item in candidates], dtype=object)
    actual_final_flags = np.array([item["actual_final_for_outcome"] for item in candidates], dtype=bool)
    valid_actual = np.isin(actuals, ["up", "down", "sideway"]) & actual_final_flags
    capped_dates = np.minimum(
        candidate_date_values,
        np.datetime64(price_end + pd.Timedelta(days=1), "ns"),
    )
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    similarities = np.maximum(0.0, 1 - distances / 1.25)

    rows = []
    for current_pos, current in enumerate(candidates):
        eligible_mask = (candidate_date_values < capped_dates[current_pos]) & valid_actual
        eligible_positions = np.flatnonzero(eligible_mask & (similarities[current_pos] >= min_similarity))
        if len(eligible_positions) == 0:
            continue
        order = np.argsort(similarities[current_pos, eligible_positions])[::-1][:top_matches]
        match_positions = eligible_positions[order]
        match_scores = similarities[current_pos, match_positions]
        votes = pd.Series(actuals[match_positions]).value_counts(normalize=True)
        forecast = str(votes.index[0])
        confidence = float(votes.iloc[0])
        actual = current["actual"]
        status = "pending"
        if actual is not None:
            is_final_for_forecast = True
            if not is_final_for_forecast:
                status = "pending"
            elif actual == forecast:
                status = "correct"
            elif actual == "mixed":
                status = "partial" if partial_score_for_forecast(
                    forecast, current["actual_return_3d"], current["actual_day_return"],
                ) > 0 else "wrong"
            else:
                status = "wrong"
            if status == "wrong" and is_final_for_forecast and forecast in {"up", "down", "sideway"}:
                delayed_scores = []
                for lag in [1, 2, 3]:
                    delayed_actual, delayed_return_3d, _, _, delayed_is_final, delayed_day_return = realized(current["date"] + pd.Timedelta(days=lag))
                    if delayed_actual is not None:
                        delayed_scores.append(
                            1.0 if delayed_actual == forecast
                            else partial_score_for_forecast(forecast, delayed_return_3d, delayed_day_return)
                            if delayed_actual == "mixed" else 0.0
                        )
                if delayed_scores and max(delayed_scores) > 0:
                    status = "delayed"
        rows.append({
            "date": current["date"], "forecast": forecast, "confidence": confidence,
            "matches": len(match_positions), "mean_similarity": np.mean(match_scores),
            "actual": actual, "status": status, "actual_return_3d": current["actual_return_3d"],
            "actual_range_2d": current["actual_range_2d"],
            "trap_warning": bool(idx.loc[idx["date"] == current["date"], "index_me"].iloc[0]
                                 - idx.loc[idx["date"] == current["date"], "index_BTC"].iloc[0] >= 20),
            "similar_dates": ", ".join(candidates[pos]["date"].strftime("%d.%m.%y") for pos in match_positions),
        })
    return pd.DataFrame(rows)


def update_scenario_ledger(
    scenarios: pd.DataFrame, daily_ohlcv: pd.DataFrame, source_name: str,
    ledger_path: str | Path, minimum_matches: int = 5,
) -> pd.DataFrame:
    """Persist emitted forecasts and grade them when enough closed daily candles exist."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "source", "date", "created_at", "forecast", "confidence", "matches",
        "mean_similarity", "trap_warning", "similar_dates", "actual", "status",
        "actual_return_3d", "actual_range_2d", "evaluated_at",
    ]
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path, parse_dates=["date"])
    else:
        ledger = pd.DataFrame(columns=columns)
    for column in ["source", "created_at", "forecast", "similar_dates", "actual", "status", "evaluated_at"]:
        ledger[column] = ledger[column].astype("object")
    market = closed_ohlcv(daily_ohlcv, "1d").copy()
    market["date"] = market["timestamp"].dt.normalize()
    market = market.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    latest_closed = market.index.max()
    now_text = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    future = scenarios[
        (scenarios["date"] > latest_closed)
        & (scenarios["matches"] >= minimum_matches)
    ].copy()
    existing_keys = set(zip(ledger.get("source", []), pd.to_datetime(ledger.get("date", [])).dt.normalize()))
    additions = []
    for _, scenario in future.iterrows():
        key = (source_name, pd.Timestamp(scenario["date"]).normalize())
        if key not in existing_keys:
            additions.append({
                "source": source_name, "date": key[1], "created_at": now_text,
                "forecast": scenario["forecast"], "confidence": scenario["confidence"],
                "matches": scenario["matches"], "mean_similarity": scenario["mean_similarity"],
                "trap_warning": scenario["trap_warning"], "similar_dates": scenario["similar_dates"],
                "actual": None, "status": "pending", "actual_return_3d": None,
                "actual_range_2d": None, "evaluated_at": None,
            })
    if additions:
        ledger = pd.concat([ledger, pd.DataFrame(additions)], ignore_index=True)

    def realized(date):
        return realized_btc_outcome(market, date)

    for row_index, row in ledger[ledger["source"] == source_name].iterrows():
        actual, return_3d, range_2d, horizon, is_final, day_return = realized(row["date"])
        if actual is None:
            continue
        forecast = str(row["forecast"])
        is_final_for_forecast = True
        if not is_final_for_forecast:
            status = "pending"
        elif actual == forecast:
            status = "correct"
        elif actual == "mixed" and partial_score_for_forecast(forecast, return_3d, day_return) > 0:
            status = "partial"
        else:
            status = "wrong"
        if status == "wrong" and is_final_for_forecast:
            delayed_scores = []
            for lag in [1, 2, 3]:
                delayed_actual, delayed_return_3d, _, _, delayed_is_final, delayed_day_return = realized(pd.Timestamp(row["date"]) + pd.Timedelta(days=lag))
                if delayed_actual is not None:
                    delayed_scores.append(
                        1.0 if delayed_actual == forecast
                        else partial_score_for_forecast(forecast, delayed_return_3d, delayed_day_return)
                        if delayed_actual == "mixed" else 0.0
                    )
            if delayed_scores and max(delayed_scores) > 0:
                status = "delayed"
        ledger.loc[row_index, ["actual", "status", "actual_return_3d", "actual_range_2d", "evaluated_at"]] = [
            actual, status, return_3d, range_2d, now_text,
        ]
    ledger = ledger[columns].sort_values(["source", "date"]).reset_index(drop=True)
    ledger.to_csv(ledger_path, index=False)
    return ledger[ledger["source"] == source_name].copy()


def scenario_accuracy_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    realized = ledger[ledger["status"].isin(["correct", "wrong", "partial", "delayed"])].copy()
    if realized.empty:
        return pd.DataFrame()
    realized["score"] = realized["status"].map({"correct": 1.0, "partial": 0.5, "delayed": 0.75, "wrong": 0.0})
    return realized.groupby("forecast").agg(
        evaluated=("date", "size"), calibrated_accuracy=("score", "mean"),
        correct=("status", lambda series: (series == "correct").sum()),
        partial=("status", lambda series: (series == "partial").sum()),
        delayed=("status", lambda series: (series == "delayed").sum()),
        wrong=("status", lambda series: (series == "wrong").sum()),
    ).reset_index()


def build_advanced_daily_features(
    indices: pd.DataFrame, daily_ohlcv: pd.DataFrame, closed_only: bool = True,
) -> pd.DataFrame:
    """Daily UTC feature frame for independent pattern discovery and ranking."""
    candles = closed_ohlcv(daily_ohlcv, "1d") if closed_only else daily_ohlcv
    data = build_dataset(indices, candles, "1d").copy()
    data["btc_psychology"] = data["index_BTC"]
    data["trader_energy"] = data["index_me"]
    data["btc_slope_3"] = _slope(data["btc_psychology"], 3)
    data["trader_slope_3"] = _slope(data["trader_energy"], 3)
    data["btc_acceleration"] = data["btc_slope_3"].diff()
    data["trader_acceleration"] = data["trader_slope_3"].diff()
    data["energy_distance"] = data["trader_energy"] - data["btc_psychology"]
    data["energy_distance_abs"] = data["energy_distance"].abs()
    data["energy_cross"] = np.sign(data["energy_distance"]) != np.sign(data["energy_distance"].shift())
    btc_direction = np.sign(data["btc_psychology"].diff())
    trader_direction = np.sign(data["trader_energy"].diff())
    price_direction = np.sign(data["close"].pct_change())
    data["same_phase"] = (btc_direction == trader_direction) & (btc_direction != 0)
    data["opposite_phase"] = (btc_direction * trader_direction) < 0
    data["btc_shock"] = data["btc_psychology"].diff().abs() >= 18
    data["trader_shock"] = data["trader_energy"].diff().abs() >= 18
    data["btc_divergence"] = (btc_direction * price_direction) < 0
    data["trader_divergence"] = (trader_direction * price_direction) < 0
    for window in [2, 5, 7]:
        data[f"index_corr_{window}d"] = data["btc_psychology"].rolling(window).corr(data["trader_energy"])
        data[f"btc_price_corr_{window}d"] = data["btc_psychology"].rolling(window).corr(data["close"])
        data[f"trader_price_corr_{window}d"] = data["trader_energy"].rolling(window).corr(data["close"])
    data["btc_extreme_high"] = data["btc_psychology"] >= 77
    data["btc_extreme_low"] = data["btc_psychology"] <= 23
    data["trader_extreme_high"] = data["trader_energy"] >= 77
    data["trader_extreme_low"] = data["trader_energy"] <= 23
    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    body = (data["close"] - data["open"]).abs()
    data["long_wick"] = (((candle_range - body) / candle_range) >= 0.65) & (candle_range / data["open"] >= 0.04)
    pump_raw = (
        (data["close"].pct_change(1) > 0.04)
        | (data["close"].pct_change(3) > 0.08)
        | (data["close"].pct_change(5) > 0.12)
        | (data["long_wick"] & (data["close"] > data["open"]))
    )
    dump_raw = (
        (data["close"].pct_change(1) < -0.04)
        | (data["close"].pct_change(3) < -0.08)
        | (data["close"].pct_change(5) < -0.12)
        | (data["long_wick"] & (data["close"] < data["open"]))
    )
    data["pump_event"] = pump_raw & (~dump_raw | (data["close"] >= data["open"]))
    data["dump_event"] = dump_raw & (~pump_raw | (data["close"] < data["open"]))
    data["pivot_high_pm3"] = data["pivot_high"].fillna(0).rolling(7, center=True, min_periods=1).max() > 0
    data["pivot_low_pm3"] = data["pivot_low"].fillna(0).rolling(7, center=True, min_periods=1).max() > 0
    return data


def btc_move_research_summary(advanced: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "1D pump >4%": advanced["close"].pct_change(1) > 0.04,
        "1D dump <-4%": advanced["close"].pct_change(1) < -0.04,
        "3D pump >8%": advanced["close"].pct_change(3) > 0.08,
        "3D dump <-8%": advanced["close"].pct_change(3) < -0.08,
        "5D pump >12%": advanced["close"].pct_change(5) > 0.12,
        "5D dump <-12%": advanced["close"].pct_change(5) < -0.12,
        "Long wick >=4% range": advanced["long_wick"],
    }
    rows = []
    for label, mask in labels.items():
        sample = advanced[mask.fillna(False)]
        rows.append({
            "BTC move label": label, "observations": len(sample),
            "mean_index_BTC_change_1d": sample["btc_psychology"].diff().mean(),
            "mean_index_me_change_1d": sample["trader_energy"].diff().mean(),
            "same_phase_rate": sample["same_phase"].mean(),
            "opposite_phase_rate": sample["opposite_phase"].mean(),
            "shock_rate": (sample["btc_shock"] | sample["trader_shock"]).mean(),
            "pivot_context_rate": (sample["pivot_high_pm3"] | sample["pivot_low_pm3"]).mean(),
        })
    return pd.DataFrame(rows)


PATTERN_DEFINITIONS = {
    "btc_extreme_high_shock": ("btc_extreme_high & btc_shock", "BTC psychology extreme-high shock"),
    "btc_extreme_low_shock": ("btc_extreme_low & btc_shock", "BTC psychology extreme-low shock"),
    "trader_extreme_high_shock": ("trader_extreme_high & trader_shock", "Trader-energy extreme-high shock"),
    "trader_extreme_low_shock": ("trader_extreme_low & trader_shock", "Trader-energy extreme-low shock"),
    "wide_gap_high_trader": ("energy_distance >= 24", "Trader energy exceeds BTC psychology by >=24"),
    "wide_gap_high_btc": ("energy_distance <= -24", "BTC psychology exceeds trader energy by >=24"),
    "energy_cross_up": ("energy_cross & (energy_distance > 0)", "Trader energy crosses above BTC psychology"),
    "energy_cross_down": ("energy_cross & (energy_distance < 0)", "Trader energy crosses below BTC psychology"),
    "same_phase_accel_up": ("same_phase & (btc_acceleration > 3) & (trader_acceleration > 3)", "Same-phase acceleration up"),
    "same_phase_accel_down": ("same_phase & (btc_acceleration < -3) & (trader_acceleration < -3)", "Same-phase acceleration down"),
    "opposite_phase_shock": ("opposite_phase & (btc_shock | trader_shock)", "Opposite-phase shock"),
    "btc_divergence_extreme": ("btc_divergence & (btc_extreme_high | btc_extreme_low)", "BTC-index price divergence at extreme"),
    "trader_divergence_extreme": ("trader_divergence & (trader_extreme_high | trader_extreme_low)", "Trader-index price divergence at extreme"),
    "negative_corr_5d": ("index_corr_5d <= -0.5", "Two indices negatively correlated over 5 days"),
    "positive_corr_5d_shock": ("(index_corr_5d >= 0.5) & (btc_shock | trader_shock)", "Positive 5-day correlation with shock"),
}


def _relative_sequence_key(frame: pd.DataFrame, window: int) -> pd.Series:
    btc_move = frame["btc_psychology"].diff(window)
    trader_move = frame["trader_energy"].diff(window)
    same_phase = np.sign(btc_move) == np.sign(trader_move)
    dominant = np.where(btc_move.abs() >= trader_move.abs(), "btc", "trader")
    direction = np.where((btc_move + trader_move) >= 0, "up", "down")
    phase = np.where(same_phase, "same", "opposite")
    strength = np.where((btc_move.abs() + trader_move.abs()) >= 24, "strong", "normal")
    return pd.Series(
        [f"seq{window}d_{p}_{d}_{dom}_{s}" for p, d, dom, s in zip(phase, direction, dominant, strength)],
        index=frame.index,
    )


def _add_relative_sequence_patterns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    definitions: dict[str, tuple[str, str]] = {}
    for window in range(1, 7):
        column = f"relative_sequence_{window}d"
        data[column] = _relative_sequence_key(data, window)
        for key in sorted(data[column].dropna().unique()):
            pattern_id = str(key)
            expression = f"{column} == '{key}'"
            label = f"Relative index sequence: {str(key).replace('_', ' ')}"
            definitions[pattern_id] = (expression, label)
    return data, definitions


def _nearest_large_move_pm3(data: pd.DataFrame) -> pd.Series:
    """Assign the nearest mutually exclusive pump/dump move inside +/-3 days."""
    events = pd.Series("neutral", index=data.index, dtype="object")
    pump_positions = set(np.flatnonzero(data["pump_event"].fillna(False).to_numpy()))
    dump_positions = set(np.flatnonzero(data["dump_event"].fillna(False).to_numpy()))
    for position in range(len(data)):
        nearby_pump = min((abs(position - candidate) for candidate in pump_positions), default=99)
        nearby_dump = min((abs(position - candidate) for candidate in dump_positions), default=99)
        if nearby_pump <= 3 and nearby_pump < nearby_dump:
            events.iloc[position] = "pump"
        elif nearby_dump <= 3 and nearby_dump < nearby_pump:
            events.iloc[position] = "dump"
    return events


def build_pattern_registry(
    indices: pd.DataFrame, daily_ohlcv: pd.DataFrame, source_name: str,
    registry_path: str | Path, as_of=None, forecast_ledger: pd.DataFrame | None = None,
    active_per_direction: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank relaxed 1-6 day patterns daily and select the top 3 pump/dump calls."""
    data = build_advanced_daily_features(indices, daily_ohlcv)
    data, sequence_definitions = _add_relative_sequence_patterns(data)
    definitions = {**PATTERN_DEFINITIONS, **sequence_definitions}
    as_of = pd.Timestamp(as_of or data["timestamp"].max()).normalize()
    train = data[data["timestamp"] <= as_of].copy()
    nearest_move = _nearest_large_move_pm3(train)
    rows = []
    for pattern_id, (expression, label) in definitions.items():
        signal = train.eval(expression).fillna(False).astype(bool)
        sample = train[signal]
        if sample.empty:
            pump_hits = dump_hits = pivot_hits = 0
        else:
            pump_hits = int((nearest_move[signal] == "pump").sum())
            dump_hits = int((nearest_move[signal] == "dump").sum())
            pivot_hits = int((train.loc[signal, "pivot_high_pm3"] | train.loc[signal, "pivot_low_pm3"]).sum())
        occurrences = int(signal.sum())
        for direction, hits, opposite in [("pump", pump_hits, dump_hits), ("dump", dump_hits, pump_hits)]:
            hit_rate = hits / occurrences if occurrences else 0.0
            opposite_rate = opposite / occurrences if occurrences else 0.0
            acceptable_rate = 1 - opposite_rate
            rows.append({
                "source": source_name, "as_of": as_of, "pattern_id": pattern_id,
                "pattern": label, "expression": expression, "direction": direction,
                "occurrences": occurrences, "hits_pm3": hits, "opposite_moves": opposite,
                "hit_rate": hit_rate, "acceptable_rate": acceptable_rate,
                "pivot_rate": pivot_hits / occurrences if occurrences else 0.0,
            })
    registry = pd.DataFrame(rows)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    prior_active = set(previous.loc[previous.get("status", pd.Series(dtype=str)) == "active", "key"]) if not previous.empty else set()
    wrong_keys: set[str] = set()
    if forecast_ledger is not None and not forecast_ledger.empty and "evaluation" in forecast_ledger:
        ledger = forecast_ledger.copy()
        wrong_keys = set(
            ledger.loc[
                (ledger.get("source", source_name) == source_name)
                & (ledger["evaluation"] == "wrong"),
                "pattern_key",
            ].dropna()
        )
    registry["key"] = registry["pattern_id"] + ":" + registry["direction"]
    registry["eligible"] = (registry["occurrences"] >= 3) & (registry["hit_rate"] >= 0.36)
    registry["candidate"] = registry["eligible"]
    registry["score"] = registry["hit_rate"] * np.log1p(registry["occurrences"]) * (0.5 + 0.5 * registry["acceptable_rate"])
    registry["status"] = "unused"
    for direction in ["pump", "dump"]:
        directional = registry[registry["direction"] == direction].sort_values(
            ["eligible", "score", "occurrences"], ascending=False
        )
        active_keys = directional[
            directional["eligible"] & ~directional["key"].isin(wrong_keys)
        ].head(active_per_direction)["key"]
        registry.loc[registry["key"].isin(active_keys), "status"] = "active"
    registry.loc[registry["candidate"] & (registry["status"] != "active"), "status"] = "candidate"
    demoted = prior_active - set(registry.loc[registry["status"] == "active", "key"])
    registry.loc[registry["key"].isin(demoted), "status"] = "retired"
    registry.loc[registry["key"].isin(wrong_keys) & (registry["status"] != "active"), "status"] = "retired"
    registry["is_new"] = registry["key"].isin(set(registry.loc[registry["status"] == "active", "key"]) - prior_active)
    registry["rank"] = registry.groupby("direction")["score"].rank(method="first", ascending=False).astype(int)
    registry.sort_values(["direction", "rank"]).to_csv(registry_path, index=False)
    signal_columns = {
        pattern_id: data.eval(expression).fillna(False).astype(bool)
        for pattern_id, (expression, _) in definitions.items()
    }
    signal_frame = pd.concat(
        [data[["timestamp", "date", "index_BTC", "index_me"]].copy(), pd.DataFrame(signal_columns, index=data.index)],
        axis=1,
    )
    return registry.sort_values(["direction", "rank"]).reset_index(drop=True), signal_frame


def registry_future_forecasts(
    indices: pd.DataFrame, latest_closed_date, registry: pd.DataFrame, source_name: str,
    ledger_path: str | Path,
) -> pd.DataFrame:
    """Issue registry-based future calls and preserve lineage when patterns retire."""
    idx = indices[["date", "index_BTC", "index_me"]].copy()
    synthetic = pd.DataFrame({
        "timestamp": idx["date"], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0,
    })
    feature_frame = build_advanced_daily_features(indices, synthetic, closed_only=False)
    feature_frame, _ = _add_relative_sequence_patterns(feature_frame)
    active = registry[registry["status"] == "active"].copy()
    future = feature_frame[feature_frame["date"] > pd.Timestamp(latest_closed_date).normalize()].copy()
    rows = []
    for _, pattern in active.iterrows():
        signals = future.eval(pattern["expression"]).fillna(False).astype(bool)
        for _, signal in future[signals].iterrows():
            rows.append({
                "source": source_name, "date": signal["date"], "pattern_key": pattern["key"],
                "pattern": pattern["pattern"], "direction": pattern["direction"],
                "rank": pattern["rank"], "historical_probability": pattern["hit_rate"],
                "occurrences": pattern["occurrences"], "acceptable_rate": pattern["acceptable_rate"],
                "icon": "new" if pattern["is_new"] else "active", "status": "pending",
            })
    issued = pd.DataFrame(rows)
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path, parse_dates=["date"])
    else:
        ledger = pd.DataFrame()
    if not issued.empty:
        issued["date"] = pd.to_datetime(issued["date"]).dt.normalize()
        ledger = pd.concat([ledger, issued], ignore_index=True).drop_duplicates(
            ["source", "date", "pattern_key"], keep="first"
        )
    if ledger.empty:
        return ledger
    active_keys = set(active["key"])
    ledger.loc[(ledger["source"] == source_name) & ~ledger["pattern_key"].isin(active_keys), "icon"] = "warning"
    ledger.sort_values(["source", "date", "rank"]).to_csv(ledger_path, index=False)
    return ledger[ledger["source"] == source_name].copy()


def evaluate_registry_forecast_ledger(
    ledger: pd.DataFrame, daily_ohlcv: pd.DataFrame, source_name: str,
    ledger_path: str | Path,
) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    market = build_advanced_daily_features(
        pd.DataFrame({
            "date": closed_ohlcv(daily_ohlcv, "1d")["timestamp"].dt.normalize(),
            "index_BTC": 50.0, "index_me": 50.0,
        }),
        daily_ohlcv,
    )
    market = market.set_index("date")
    ledger = ledger.copy()
    for column in ["actual", "evaluation"]:
        if column not in ledger:
            ledger[column] = None
        ledger[column] = ledger[column].astype("object")
    for row_index, row in ledger[ledger["source"] == source_name].iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        future = market.loc[market.index > date].head(3)
        if len(future) < 3:
            continue
        pump = bool(future["pump_event"].any())
        dump = bool(future["dump_event"].any())
        actual = "pump" if pump and not dump else ("dump" if dump and not pump else "neutral")
        direction = row["direction"]
        evaluation = "correct" if actual == direction else ("acceptable" if actual == "neutral" else "wrong")
        ledger.loc[row_index, ["actual", "evaluation"]] = [actual, evaluation]
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    ledger.sort_values(["source", "date", "rank"]).to_csv(ledger_path, index=False)
    return ledger[ledger["source"] == source_name].copy()


def should_run_daily_learning(state_path: str | Path, now=None) -> bool:
    """Run once per UTC day after 03:00 UTC."""
    now = pd.Timestamp(now or pd.Timestamp.now("UTC"))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    if now.hour < 3:
        return False
    state_path = Path(state_path)
    if not state_path.exists():
        return True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("last_learning_date") != now.strftime("%Y-%m-%d")


def write_daily_learning_state(
    state_path: str | Path, registry: pd.DataFrame, scenario_ledger: pd.DataFrame,
    registry_ledger: pd.DataFrame, model_metrics: pd.DataFrame, now=None,
) -> dict:
    now = pd.Timestamp(now or pd.Timestamp.now("UTC"))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    state = {
        "last_learning_date": now.strftime("%Y-%m-%d"),
        "last_learning_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_patterns": int((registry["status"] == "active").sum()),
        "candidate_patterns": int((registry["status"] == "candidate").sum()),
        "scenario_evaluated": int(scenario_ledger["status"].isin(["correct", "wrong", "partial", "delayed"]).sum()),
        "registry_evaluated": int(registry_ledger.get("evaluation", pd.Series(dtype=str)).isin(["correct", "wrong", "acceptable"]).sum()),
        "model_metrics": model_metrics.to_dict("records"),
    }
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def read_daily_learning_state(state_path: str | Path) -> dict:
    state_path = Path(state_path)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


@dataclass
class ModelResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    importance: pd.DataFrame


def walk_forward_models(data: pd.DataFrame, target: str, min_train: int = 120) -> ModelResult:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run model validation.") from exc
    candidates = {
        "Logistic Regression": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(SimpleImputer(), RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42
        )),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = make_pipeline(SimpleImputer(), XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.04, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=42,
        ))
    except ImportError:
        pass
    model_data = data.dropna(subset=[target]).copy()
    x = model_data[FEATURE_COLUMNS]
    y = model_data[target].astype(int)
    if len(model_data) < min_train + 30 or y.nunique() < 2:
        raise ValueError("Not enough labeled rows for walk-forward validation.")
    test_size = max(20, len(model_data) // 6)
    split_points = list(range(min_train, len(model_data), test_size))
    all_predictions, metric_rows, importance_rows = [], [], []
    for name, model in candidates.items():
        model_predictions = []
        for fold, split in enumerate(split_points, start=1):
            end = min(split + test_size, len(model_data))
            if end <= split or y.iloc[:split].nunique() < 2:
                continue
            model.fit(x.iloc[:split], y.iloc[:split])
            probs = model.predict_proba(x.iloc[split:end])[:, 1]
            model_predictions.append(pd.DataFrame({
                "timestamp": model_data["timestamp"].iloc[split:end].values,
                "actual": y.iloc[split:end].values, "probability": probs,
                "model": name, "fold": fold,
            }))
        if not model_predictions:
            continue
        preds = pd.concat(model_predictions, ignore_index=True)
        auc = roc_auc_score(preds["actual"], preds["probability"]) if preds["actual"].nunique() > 1 else np.nan
        metric_rows.append({"model": name, "rows": len(preds), "auc": auc,
                            "accuracy_at_0.5": accuracy_score(preds["actual"], preds["probability"] >= 0.5)})
        all_predictions.append(preds)
        model.fit(x, y)
        fitted = model.steps[-1][1]
        values = getattr(fitted, "feature_importances_", None)
        if values is None and hasattr(fitted, "coef_"):
            values = np.abs(fitted.coef_[0])
        if values is not None:
            importance_rows.extend({"model": name, "feature": feature, "importance": float(value)}
                                   for feature, value in zip(FEATURE_COLUMNS, values))
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return ModelResult(predictions, pd.DataFrame(metric_rows), pd.DataFrame(importance_rows))


def false_high_confidence_signals(predictions: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    return predictions[(predictions["probability"] >= threshold) & (predictions["actual"] == 0)].sort_values(
        "probability", ascending=False
    )
