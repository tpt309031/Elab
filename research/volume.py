"""Fixed-venue spot flow. Never mix venue volumes or forward-fill missing candles."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

SOURCE = "Binance spot BTCUSDT"
COLUMNS = ["date", "base_volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


def validate_volume(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    result = frame[COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None)
    if result["date"].isna().any() or not result["date"].eq(result["date"].dt.normalize()).all():
        raise ValueError("Volume timestamps must be daily UTC opens")
    for column in COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Non-finite or negative exchange volume")
    if (result["taker_buy_base"] > result["base_volume"] + 1e-6).any():
        raise ValueError("Taker volume exceeds total BTC volume")
    if (result["taker_buy_quote"] > result["quote_volume"] + 1e-3).any():
        raise ValueError("Taker notional exceeds total quote volume")
    return result[result["date"] <= end].drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def refresh_volume(path: Path, end: pd.Timestamp, refresh: bool = True) -> tuple[pd.DataFrame, dict]:
    end = pd.Timestamp(end).normalize()
    cached = validate_volume(pd.read_csv(path), end) if path.exists() else pd.DataFrame(columns=COLUMNS)
    result = cached
    errors = []
    endpoint = None
    if refresh:
        start = max(pd.Timestamp("2017-08-17"), cached["date"].max() - pd.Timedelta(days=5)) if len(cached) else pd.Timestamp("2017-08-17")
        if len(cached):
            missing = pd.date_range("2017-08-17", cached["date"].max()).difference(pd.DatetimeIndex(cached["date"]))
            if len(missing):
                start = min(start, missing.min())
        for host in ("https://data-api.binance.vision", "https://api.binance.com"):
            try:
                rows = []
                cursor = int(start.timestamp() * 1000)
                stop = int((end + pd.Timedelta(days=1)).timestamp() * 1000)
                while cursor < stop:
                    query = urlencode(dict(symbol="BTCUSDT", interval="1d", startTime=cursor, endTime=stop - 1, limit=1000))
                    response = requests.get(f"{host}/api/v3/klines?{query}", timeout=25)
                    response.raise_for_status()
                    candles = response.json()
                    if not isinstance(candles, list) or not candles:
                        break
                    for candle in candles:
                        rows.append([pd.to_datetime(candle[0], unit="ms"), *[float(candle[i]) for i in (5, 7, 8, 9, 10)]])
                    next_cursor = int(candles[-1][0]) + 86400000
                    if next_cursor <= cursor:
                        raise ValueError("Volume pagination did not advance")
                    cursor = next_cursor
                if not rows:
                    raise ValueError("Exchange returned no volume")
                incoming = validate_volume(pd.DataFrame(rows, columns=COLUMNS), end)
                result = validate_volume(pd.concat([cached, incoming], ignore_index=True), end)
                if result["date"].max() < end:
                    raise ValueError("Volume feed did not reach the latest closed UTC day")
                endpoint = host
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                result.to_csv(temporary, index=False)
                temporary.replace(path)
                break
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                errors.append(f"{host}: {type(exc).__name__}: {exc}")
    latest = result["date"].max() if len(result) else None
    gaps = len(pd.date_range(result["date"].min(), end).difference(pd.DatetimeIndex(result["date"]))) if len(result) else 0
    health = dict(source=SOURCE, endpoint=endpoint, status="healthy" if latest == end and gaps == 0 else "degraded",
                  latest_closed_utc=latest.strftime("%Y-%m-%d") if latest is not None else None,
                  rows=len(result), missing_days=gaps, errors=errors, base_unit="BTC", quote_unit="USDT")
    return result, health


def volume_features(volume: pd.DataFrame) -> pd.DataFrame:
    data = volume.set_index("date").sort_index().asfreq("D")
    output = pd.DataFrame(index=data.index)
    base = data["base_volume"]
    quote = data["quote_volume"]
    log_volume = np.log1p(quote)
    output["spot_rvol20"] = base / base.shift(1).rolling(20).mean().replace(0, np.nan)
    output["spot_quote_z20"] = (log_volume - log_volume.shift(1).rolling(20).mean()) / log_volume.shift(1).rolling(20).std().replace(0, np.nan)
    output["spot_volume_trend"] = base.rolling(5).mean() / base.rolling(20).mean().replace(0, np.nan) - 1
    output["spot_taker_imbalance"] = 2 * data["taker_buy_base"] / base.replace(0, np.nan) - 1
    output["spot_imbalance_3"] = output["spot_taker_imbalance"].rolling(3).mean()
    output["spot_average_trade_usdt"] = np.log1p(quote / data["trades"].replace(0, np.nan))
    output["spot_volume_change"] = log_volume.diff()
    return output.replace([np.inf, -np.inf], np.nan).reset_index()
