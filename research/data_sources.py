from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PREPUBLISHED_AT = pd.Timestamp("1900-01-01")
INTRADAY_INTERVALS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}
EXTERNAL_SCHEMAS: dict[str, tuple[str, ...]] = {
    "derivatives.csv": (
        "funding_rate",
        "open_interest",
        "basis",
        "liquidations_long",
        "liquidations_short",
    ),
    "options.csv": ("options_iv", "options_skew_25d"),
    "macro.csv": ("etf_net_flow", "dxy", "us10y"),
    "onchain.csv": ("onchain_active_addresses", "exchange_netflow"),
}


def _request_json(url: str, timeout: int = 30) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "ELAB-Research/3.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_lineage(
    path: str | Path,
    source_name: str,
    rows: int,
    first_date: object,
    last_date: object,
    explicit_availability_rows: int = 0,
    availability_mode: str | None = None,
) -> dict[str, object]:
    source = Path(path)

    def date_text(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    return {
        "source": source_name,
        "file": source.name,
        "sha256": sha256_file(source),
        "bytes": int(source.stat().st_size),
        "rows": int(rows),
        "first_date": date_text(first_date),
        "last_date": date_text(last_date),
        "explicit_availability_rows": int(explicit_availability_rows),
        "availability_mode": availability_mode or (
            "explicit" if explicit_availability_rows == rows and rows > 0 else "prepublished-imputed"
        ),
    }


def availability_series(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return UTC-naive availability timestamps and an imputation flag."""
    if "available_at" not in frame:
        return (
            pd.Series(PREPUBLISHED_AT, index=frame.index, dtype="datetime64[ns]"),
            pd.Series(True, index=frame.index, dtype=bool),
        )
    parsed = pd.to_datetime(frame["available_at"], errors="coerce", utc=True).dt.tz_localize(None)
    imputed = parsed.isna()
    return parsed.fillna(PREPUBLISHED_AT), imputed


def load_external_features(directory: str | Path) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Load optional point-in-time feature CSVs without converting missing feeds to zero."""
    root = Path(directory)
    frames: list[pd.DataFrame] = []
    health: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for filename, expected_columns in EXTERNAL_SCHEMAS.items():
        path = root / filename
        if not path.exists():
            health.append({
                "source": filename.removesuffix(".csv"),
                "available": False,
                "status": "missing optional feed",
                "rows": 0,
                "usable_rows": 0,
            })
            continue
        source = pd.read_csv(path)
        if "date" not in source:
            health.append({
                "source": filename.removesuffix(".csv"),
                "available": False,
                "status": "invalid: date column missing",
                "rows": int(len(source)),
                "usable_rows": 0,
            })
            continue
        source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        available_at, imputed = availability_series(source)
        source["available_at"] = available_at
        source["availability_imputed"] = imputed
        columns = [column for column in expected_columns if column in source]
        for column in columns:
            source[column] = pd.to_numeric(source[column], errors="coerce")
            source.loc[source["available_at"] > source["date"], column] = np.nan
        compact = source[["date", *columns]].dropna(subset=["date"])
        compact = compact.sort_values("date").drop_duplicates("date", keep="last")
        if columns:
            frames.append(compact)
        usable = int(compact[columns].notna().any(axis=1).sum()) if columns else 0
        status = "healthy" if columns and usable else "present but no usable point-in-time values"
        health.append({
            "source": filename.removesuffix(".csv"),
            "available": bool(columns and usable),
            "status": status,
            "rows": int(len(source)),
            "usable_rows": usable,
            "features": columns,
            "availability_imputed_rows": int(imputed.sum()),
        })
        lineage.append(source_lineage(
            path,
            filename.removesuffix(".csv"),
            len(source),
            source["date"].min(),
            source["date"].max(),
            int((~imputed).sum()),
        ))
    if not frames:
        return pd.DataFrame(columns=["date"]), health, lineage
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True), health, lineage


def _fetch_binance_intraday_from_base(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    base_url: str,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    cursor = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    interval_ms = int(INTRADAY_INTERVALS[timeframe].total_seconds() * 1000)
    for _ in range(1000):
        query = urllib.parse.urlencode({
            "symbol": "BTCUSDT",
            "interval": timeframe,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        })
        payload = _request_json(f"{base_url}/api/v3/klines?{query}")
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + interval_ms
        if next_cursor <= cursor or next_cursor > end_ms:
            break
        cursor = next_cursor
    if not rows:
        raise RuntimeError(f"Binance returned no {timeframe} candles")
    return pd.DataFrame({
        "timestamp": pd.to_datetime([int(row[0]) for row in rows], unit="ms", utc=True).tz_localize(None),
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [float(row[5]) for row in rows],
    })


def _prepare_intraday(frame: pd.DataFrame, expected_open: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"{timeframe} market data is missing columns: {', '.join(missing)}")
    output = frame[required].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="raise", utc=True).dt.tz_localize(None)
    for column in required[1:]:
        output[column] = pd.to_numeric(output[column], errors="raise")
    output = output[output["timestamp"] <= expected_open]
    output = output.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    invalid = (
        (output[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (output["volume"] < 0)
        | (output["high"] < output[["open", "close"]].max(axis=1))
        | (output["low"] > output[["open", "close"]].min(axis=1))
    )
    if output.empty or invalid.any():
        raise RuntimeError(f"Invalid or empty {timeframe} OHLCV data")
    return output


def refresh_intraday_market(
    cache_path: str | Path,
    timeframe: str,
    start: str = "2020-01-01",
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if timeframe not in INTRADAY_INTERVALS:
        raise ValueError(f"Unsupported intraday timeframe: {timeframe}")
    path = Path(cache_path)
    cached = pd.read_csv(path, parse_dates=["timestamp"]) if path.exists() else pd.DataFrame()
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is not None:
        current = current.tz_convert("UTC").tz_localize(None)
    interval = INTRADAY_INTERVALS[timeframe]
    expected_open = current.floor(interval) - interval
    requested_start = pd.Timestamp(start)
    cached_latest: pd.Timestamp | None = None
    fetch_start = requested_start
    if not cached.empty:
        cached_latest = pd.to_datetime(cached["timestamp"]).max()
        fetch_start = max(requested_start, cached_latest - interval * 12)
    attempts: list[dict[str, object]] = []
    fresh = pd.DataFrame()
    provider = "cache-current"
    for base_url in (
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api.binance.com",
    ):
        started = time.perf_counter()
        try:
            candidate = _prepare_intraday(
                _fetch_binance_intraday_from_base(fetch_start, expected_open + interval, timeframe, base_url),
                expected_open,
                timeframe,
            )
            latest = candidate["timestamp"].max()
            if latest < expected_open:
                raise RuntimeError(f"latest={latest}, expected={expected_open}")
            fresh = candidate
            provider = base_url
            attempts.append({
                "provider": base_url,
                "status": "healthy",
                "latest_open_utc": latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": None,
            })
            break
        except Exception as exc:
            attempts.append({
                "provider": base_url,
                "status": "failed",
                "latest_open_utc": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            })
    if fresh.empty and (cached_latest is None or cached_latest < expected_open):
        raise RuntimeError(
            f"Closed {timeframe} BTC bar {expected_open} unavailable; cache latest={cached_latest}",
        )
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    combined = _prepare_intraday(combined, expected_open, timeframe)
    actual_latest = combined["timestamp"].max()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    health = {
        "timeframe": timeframe,
        "status": "healthy" if fresh.shape[0] else "degraded-cache-current",
        "provider": provider,
        "expected_open_utc": expected_open.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actual_open_utc": actual_latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cache_latest_before_refresh": cached_latest.strftime("%Y-%m-%dT%H:%M:%SZ") if cached_latest is not None else None,
        "stale": bool(actual_latest < expected_open),
        "rows": int(len(combined)),
        "attempts": attempts,
    }
    return combined, health


def _daily_intraday_aggregate(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    data = frame.copy().sort_values("timestamp")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.normalize()
    data["bar_return"] = data["close"] / data["open"] - 1
    data["signed_volume"] = np.sign(data["close"] - data["open"]) * data["volume"]
    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    data["wick_imbalance"] = (
        (data[["open", "close"]].min(axis=1) - data["low"])
        - (data["high"] - data[["open", "close"]].max(axis=1))
    ) / candle_range
    expected_bars = int(pd.Timedelta(days=1) / INTRADAY_INTERVALS[timeframe])
    rows: list[dict[str, object]] = []
    suffix = timeframe.replace("h", "h")
    for date, group in data.groupby("date", sort=True):
        volume = float(group["volume"].sum())
        rows.append({
            "date": pd.Timestamp(date),
            f"intraday_realized_vol_{suffix}": float(math.sqrt(np.square(group["bar_return"]).sum())),
            f"intraday_trend_{suffix}": float(group["close"].iloc[-1] / group["open"].iloc[0] - 1),
            f"intraday_range_{suffix}": float(group["high"].max() / group["low"].min() - 1),
            f"intraday_signed_volume_{suffix}": float(group["signed_volume"].sum() / volume) if volume > 0 else math.nan,
            f"intraday_wick_imbalance_{suffix}": float(group["wick_imbalance"].mean()),
            f"intraday_volume_{suffix}": volume,
            f"intraday_coverage_{suffix}": float(min(1.0, len(group) / expected_bars)),
        })
    return pd.DataFrame(rows)


def build_intraday_daily_features(frames: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    aggregates = [_daily_intraday_aggregate(frame, timeframe) for timeframe, frame in frames if not frame.empty]
    if not aggregates:
        return pd.DataFrame(columns=["date"])
    output = aggregates[0]
    for frame in aggregates[1:]:
        output = output.merge(frame, on="date", how="outer")
    return output.sort_values("date").reset_index(drop=True)
