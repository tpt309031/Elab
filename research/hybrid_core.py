from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from research.data_sources import PREPUBLISHED_AT, availability_series
from research.evaluation import (
    ProbabilityCalibrator,
    conservative_beta_lower_bound,
    moving_block_confidence_intervals,
    multiclass_diagnostics,
)
from research.model_candidates import candidate_model_specs, learn_simplex_weights


CLASS_NAMES = ("down", "sideway", "up")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
OOS_START = pd.Timestamp("2024-01-01")
UP_CORRECT = 0.03
DOWN_CORRECT = -0.03
SIDEWAY_LIMIT = 0.01
MAX_SIDEWAY_PER_MONTH = 8
UP_PARTIAL_MIN = 0.001
DOWN_PARTIAL_MAX = -0.001
TRADING_COST = 0.0005


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_index: np.ndarray
    calibration_index: np.ndarray
    test_index: np.ndarray


@dataclass
class CandidatePrediction:
    name: str
    calibration_probabilities: np.ndarray
    test_probabilities: np.ndarray
    calibration_score: float
    calibration_log_loss: float
    weight: float
    calibration_method: str = "uncalibrated"
    calibration_selection_log_loss: float = math.nan


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame
    model_predictions: pd.DataFrame
    model_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    registry: pd.DataFrame
    no_call_summary: pd.DataFrame


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def grade_forecast(direction: str, daily_return: float | None) -> tuple[str, float]:
    if daily_return is None or not np.isfinite(daily_return):
        return "pending", math.nan
    move = float(daily_return)
    if direction == "up":
        if move >= UP_CORRECT:
            return "correct", 1.0
        if UP_PARTIAL_MIN <= move < UP_CORRECT:
            return "partial", 0.5
        return "wrong", 0.0
    if direction == "down":
        if move <= DOWN_CORRECT:
            return "correct", 1.0
        if DOWN_CORRECT < move <= DOWN_PARTIAL_MAX:
            return "partial", 0.5
        return "wrong", 0.0
    if direction == "sideway":
        return ("correct", 1.0) if -SIDEWAY_LIMIT <= move <= SIDEWAY_LIMIT else ("wrong", 0.0)
    return "no-call", math.nan


def direction_label(daily_return: float | None) -> int | float:
    if daily_return is None or not np.isfinite(daily_return):
        return math.nan
    if daily_return > SIDEWAY_LIMIT:
        return CLASS_TO_INDEX["up"]
    if daily_return < -SIDEWAY_LIMIT:
        return CLASS_TO_INDEX["down"]
    return CLASS_TO_INDEX["sideway"]


def _request_json(url: str, timeout: int = 30) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "elab-hybrid-research/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_binance_daily_from_base(start: pd.Timestamp, end: pd.Timestamp, base_url: str) -> pd.DataFrame:
    rows: list[list[object]] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    day_ms = 86_400_000
    while cursor < end_ms:
        query = urllib.parse.urlencode({
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        })
        payload = _request_json(f"{base_url}/api/v3/klines?{query}")
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + day_ms
        if next_cursor <= cursor or len(payload) < 1000:
            break
        cursor = next_cursor
    if not rows:
        raise RuntimeError("Binance returned no daily candles")
    return pd.DataFrame({
        "timestamp": pd.to_datetime([row[0] for row in rows], unit="ms", utc=True).tz_localize(None).normalize(),
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [float(row[5]) for row in rows],
    })


def _fetch_binance_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    errors: list[str] = []
    # The market-data-only host is less likely to be geo-blocked on hosted CI runners.
    for base_url in (
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api.binance.com",
    ):
        try:
            return _fetch_binance_daily_from_base(start, end, base_url)
        except Exception as exc:
            errors.append(f"{base_url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _fetch_okx_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    collected: list[list[str]] = []
    cursor = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    for _ in range(80):
        query = urllib.parse.urlencode({
            "instId": "BTC-USDT",
            "bar": "1Dutc",
            "after": cursor,
            "limit": 300,
        })
        payload = _request_json(f"https://www.okx.com/api/v5/market/history-candles?{query}")
        chunk = payload.get("data", []) if isinstance(payload, dict) else []
        if not chunk:
            break
        collected.extend(chunk)
        oldest = min(int(row[0]) for row in chunk)
        if oldest <= start_ms:
            break
        cursor = oldest - 1
    if not collected:
        raise RuntimeError("OKX returned no daily candles")
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([int(row[0]) for row in collected], unit="ms", utc=True).tz_localize(None).normalize(),
        "open": [float(row[1]) for row in collected],
        "high": [float(row[2]) for row in collected],
        "low": [float(row[3]) for row in collected],
        "close": [float(row[4]) for row in collected],
        "volume": [float(row[5]) for row in collected],
    })
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def _fetch_coinbase_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    collected: list[dict[str, str]] = []
    cursor = pd.Timestamp(start).normalize()
    end_at = pd.Timestamp(end).normalize()
    while cursor < end_at:
        page_end = min(cursor + pd.Timedelta(days=299), end_at)
        query = urllib.parse.urlencode({
            "start": int(cursor.timestamp()),
            "end": int(page_end.timestamp()),
            "granularity": "ONE_DAY",
        })
        payload = _request_json(
            f"https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USDT/candles?{query}",
        )
        chunk = payload.get("candles", []) if isinstance(payload, dict) else []
        collected.extend(item for item in chunk if isinstance(item, dict))
        cursor = page_end
    if not collected:
        raise RuntimeError("Coinbase returned no daily candles")
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([int(row["start"]) for row in collected], unit="s", utc=True).tz_localize(None).normalize(),
        "open": [float(row["open"]) for row in collected],
        "high": [float(row["high"]) for row in collected],
        "low": [float(row["low"]) for row in collected],
        "close": [float(row["close"]) for row in collected],
        "volume": [float(row["volume"]) for row in collected],
    })
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def _prepare_daily_market(frame: pd.DataFrame, closed_end: pd.Timestamp) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Daily market data is missing columns: {', '.join(missing)}")
    output = frame[required].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="raise").dt.normalize()
    for column in required[1:]:
        output[column] = pd.to_numeric(output[column], errors="raise")
    output = output[output["timestamp"] <= closed_end]
    output = output.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    if output.empty:
        raise RuntimeError("Daily market data is empty after closed-candle filtering")
    invalid = (
        (output[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (output["volume"] < 0)
        | (output["high"] < output[["open", "close"]].max(axis=1))
        | (output["low"] > output[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        bad_date = output.loc[invalid, "timestamp"].iloc[0].strftime("%Y-%m-%d")
        raise RuntimeError(f"Invalid daily OHLCV candle: {bad_date}")
    return output


def refresh_daily_market(
    cache_path: str | Path,
    start: str = "2017-08-17",
    now: pd.Timestamp | None = None,
    include_health: bool = False,
) -> tuple[pd.DataFrame, str] | tuple[pd.DataFrame, str, dict[str, object]]:
    cache_path = Path(cache_path)
    cached = pd.read_csv(cache_path, parse_dates=["timestamp"]) if cache_path.exists() else pd.DataFrame()
    requested_start = pd.Timestamp(start)
    current = pd.Timestamp(now) if now is not None else utc_now()
    if current.tzinfo is not None:
        current = current.tz_convert("UTC").tz_localize(None)
    closed_end = current.normalize() - pd.Timedelta(days=1)
    fetch_start = requested_start
    cached_latest: pd.Timestamp | None = None
    if not cached.empty:
        earliest = pd.to_datetime(cached["timestamp"]).min().normalize()
        cached_latest = pd.to_datetime(cached["timestamp"]).max().normalize()
        if earliest <= requested_start:
            fetch_start = max(requested_start, cached_latest - pd.Timedelta(days=5))
    provider = "cache-current"
    fresh = pd.DataFrame()
    errors: list[str] = []
    attempts: list[dict[str, object]] = []
    valid_sources: list[tuple[str, pd.DataFrame]] = []
    providers = (
        ("Binance", _fetch_binance_daily),
        ("OKX", _fetch_okx_daily),
        ("Coinbase", _fetch_coinbase_daily),
    )
    for provider_name, fetcher in providers:
        started = time.perf_counter()
        try:
            candidate = _prepare_daily_market(
                fetcher(fetch_start, closed_end + pd.Timedelta(days=1)), closed_end,
            )
            candidate_latest = candidate["timestamp"].max().normalize()
            if candidate_latest < closed_end:
                raise RuntimeError(
                    f"latest candle {candidate_latest:%Y-%m-%d}, expected {closed_end:%Y-%m-%d}",
                )
            valid_sources.append((provider_name, candidate))
            if fresh.empty:
                fresh = candidate
                provider = provider_name
            latest_close = float(candidate.loc[candidate["timestamp"] == candidate_latest, "close"].iloc[-1])
            attempts.append({
                "provider": provider_name,
                "status": "healthy",
                "latest_closed_utc": candidate_latest.strftime("%Y-%m-%d"),
                "latest_close": latest_close,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": None,
            })
            print(f"[market] {provider_name} supplied closed candle {closed_end:%Y-%m-%d}")
            if not include_health:
                break
        except Exception as exc:
            message = f"{provider_name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            attempts.append({
                "provider": provider_name,
                "status": "failed",
                "latest_closed_utc": None,
                "latest_close": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[market] provider failed: {message}")
    if fresh.empty and (cached_latest is None or cached_latest < closed_end):
        detail = " | ".join(errors) or "no providers attempted"
        raise RuntimeError(
            f"Closed BTC candle {closed_end:%Y-%m-%d} is unavailable; "
            f"cache latest={cached_latest}; {detail}",
        )
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    combined = _prepare_daily_market(combined, closed_end)
    latest_closed = combined["timestamp"].max().normalize()
    if latest_closed < closed_end:
        raise RuntimeError(f"Market refresh remained stale at {latest_closed:%Y-%m-%d}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    combined.to_csv(temporary, index=False)
    temporary.replace(cache_path)
    if not include_health:
        return combined, provider
    source_closes = [
        float(candidate.loc[candidate["timestamp"] == closed_end, "close"].iloc[-1])
        for _, candidate in valid_sources
        if (candidate["timestamp"] == closed_end).any()
    ]
    median_close = float(np.median(source_closes)) if source_closes else math.nan
    discrepancy_bps = (
        float((max(source_closes) - min(source_closes)) / median_close * 10_000)
        if len(source_closes) > 1 and median_close > 0 else None
    )
    degraded = provider == "cache-current" or bool(errors)
    health: dict[str, object] = {
        "status": "degraded" if degraded else "healthy",
        "expected_closed_utc": closed_end.strftime("%Y-%m-%d"),
        "actual_closed_utc": latest_closed.strftime("%Y-%m-%d"),
        "cache_latest_before_refresh": cached_latest.strftime("%Y-%m-%d") if cached_latest is not None else None,
        "selected_provider": provider,
        "provider_count": len(valid_sources),
        "cross_exchange_close_discrepancy_bps": discrepancy_bps,
        "stale": bool(latest_closed < closed_end),
        "attempts": attempts,
    }
    return combined, provider, health


def load_indices(index_btc_path: str | Path, index_me_path: str | Path) -> pd.DataFrame:
    btc = pd.read_csv(index_btc_path)
    trader = pd.read_csv(index_me_path)
    btc = btc.rename(columns={"score_percent": "index_BTC"})
    trader = trader.rename(columns={"score_percent": "index_me"})
    for frame in (btc, trader):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    btc_available, btc_imputed = availability_series(btc)
    trader_available, trader_imputed = availability_series(trader)
    btc["index_BTC_available_at"] = btc_available
    btc["index_BTC_availability_imputed"] = btc_imputed
    trader["index_me_available_at"] = trader_available
    trader["index_me_availability_imputed"] = trader_imputed
    merged = btc[[
        "date", "index_BTC", "index_BTC_available_at", "index_BTC_availability_imputed",
    ]].merge(trader[[
        "date", "index_me", "index_me_available_at", "index_me_availability_imputed",
    ]], on="date", how="outer")
    merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("index_BTC", "index_me"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").clip(0, 100)
    for column in ("index_BTC_available_at", "index_me_available_at"):
        merged[column] = pd.to_datetime(merged[column], errors="coerce").fillna(PREPUBLISHED_AT)
    for column in ("index_BTC_availability_imputed", "index_me_availability_imputed"):
        merged[column] = merged[column].fillna(True).astype(bool)
    return merged


def load_astro(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for column in ("finance", "career", "volatility", "composite"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    available_at, availability_imputed = availability_series(frame)
    frame["astro_available_at"] = available_at
    frame["astro_availability_imputed"] = availability_imputed
    frame["event"] = frame.get("event", False).fillna(False).astype(bool)
    frame["regime"] = frame.get("regime", "normal").fillna("normal").astype(str)
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return math.nan
        return float(np.dot(values - values.mean(), x_centered) / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return (series - mean) / std


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _atr(market: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = market["close"].shift(1)
    true_range = pd.concat(
        [
            market["high"] - market["low"],
            (market["high"] - previous_close).abs(),
            (market["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_feature_frame(
    indices: pd.DataFrame,
    market: pd.DataFrame,
    astro: pd.DataFrame,
    intraday: pd.DataFrame | None = None,
    external: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    market = market.copy().sort_values("timestamp")
    market["date"] = pd.to_datetime(market["timestamp"]).dt.normalize()
    market["daily_return"] = market["close"] / market["open"] - 1
    market["market_return_1"] = market["close"].pct_change(1)
    market["market_return_3"] = market["close"].pct_change(3)
    market["market_return_7"] = market["close"].pct_change(7)
    market["market_return_14"] = market["close"].pct_change(14)
    market["volatility_7"] = market["market_return_1"].rolling(7).std()
    market["volatility_21"] = market["market_return_1"].rolling(21).std()
    market["distance_ma20"] = market["close"] / market["close"].rolling(20).mean() - 1
    market["distance_ma50"] = market["close"] / market["close"].rolling(50).mean() - 1
    market["ema_gap_12_26"] = market["close"].ewm(span=12, adjust=False).mean() / market["close"].ewm(span=26, adjust=False).mean() - 1
    market["rsi14"] = _rsi(market["close"])
    market["atr14_pct"] = _atr(market) / market["close"]
    market["volume_z20"] = _rolling_zscore(market["volume"], 20)
    candle_range = (market["high"] - market["low"]).replace(0, np.nan)
    market["body_pct"] = (market["close"] - market["open"]) / market["open"]
    market["upper_wick_ratio"] = (market["high"] - market[["open", "close"]].max(axis=1)) / candle_range
    market["lower_wick_ratio"] = (market[["open", "close"]].min(axis=1) - market["low"]) / candle_range
    market["range_pct"] = candle_range / market["open"]
    target_columns = ["date", "open", "high", "low", "close", "volume", "daily_return"]
    technical_columns = [
        "market_return_1", "market_return_3", "market_return_7", "market_return_14",
        "volatility_7", "volatility_21", "distance_ma20", "distance_ma50", "ema_gap_12_26",
        "rsi14", "atr14_pct", "volume_z20", "body_pct", "upper_wick_ratio", "lower_wick_ratio", "range_pct",
    ]
    market_features = market[target_columns + technical_columns].copy()
    market_features[technical_columns] = market_features[technical_columns].shift(1)

    frame = astro.copy()
    frame = frame.merge(indices, on="date", how="outer")
    frame = frame.merge(market_features, on="date", how="outer").sort_values("date").reset_index(drop=True)
    for availability_column, imputed_column in (
        ("index_BTC_available_at", "index_BTC_availability_imputed"),
        ("index_me_available_at", "index_me_availability_imputed"),
        ("astro_available_at", "astro_availability_imputed"),
    ):
        if availability_column not in frame:
            frame[availability_column] = PREPUBLISHED_AT
        if imputed_column not in frame:
            frame[imputed_column] = True
    intraday_columns: list[str] = []
    if intraday is not None and not intraday.empty:
        intraday_frame = intraday.copy()
        intraday_frame["date"] = pd.to_datetime(intraday_frame["date"]).dt.normalize()
        intraday_columns = [column for column in intraday_frame if column != "date"]
        frame = frame.merge(intraday_frame, on="date", how="outer").sort_values("date").reset_index(drop=True)
        # Intraday aggregates from UTC day D become available to the forecast for D+1.
        frame[intraday_columns] = frame[intraday_columns].shift(1)
        for column in [item for item in intraday_columns if "volume_" in item and "signed" not in item]:
            frame[f"{column}_z20"] = _rolling_zscore(frame[column], 20)
            intraday_columns.append(f"{column}_z20")
    external_columns: list[str] = []
    if external is not None and not external.empty:
        external_frame = external.copy()
        external_frame["date"] = pd.to_datetime(external_frame["date"]).dt.normalize()
        external_columns = [column for column in external_frame if column != "date"]
        frame = frame.merge(external_frame, on="date", how="outer").sort_values("date").reset_index(drop=True)
        for column in list(external_columns):
            frame[f"{column}_change_1"] = frame[column].diff()
            frame[f"{column}_z30"] = _rolling_zscore(frame[column], 30)
            external_columns.extend([f"{column}_change_1", f"{column}_z30"])
    frame = frame[(frame["date"] >= pd.Timestamp("2017-08-17"))].copy()

    forecast_cutoff = frame["date"]
    for value_column, availability_column, imputed_column in (
        ("index_BTC", "index_BTC_available_at", "index_BTC_availability_imputed"),
        ("index_me", "index_me_available_at", "index_me_availability_imputed"),
    ):
        if availability_column in frame:
            explicit = ~frame[imputed_column].fillna(True).astype(bool)
            unavailable = explicit & (pd.to_datetime(frame[availability_column]) > forecast_cutoff)
            frame.loc[unavailable, value_column] = np.nan
    if "astro_available_at" in frame:
        explicit_astro = ~frame["astro_availability_imputed"].fillna(True).astype(bool)
        unavailable_astro = explicit_astro & (pd.to_datetime(frame["astro_available_at"]) > forecast_cutoff)
        frame.loc[unavailable_astro, ["finance", "career", "volatility", "composite"]] = np.nan
        frame.loc[unavailable_astro, "event"] = False

    return_median = frame["market_return_1"].rolling(63, min_periods=30).median()
    return_mad = (frame["market_return_1"] - return_median).abs().rolling(63, min_periods=30).median()
    volatility_median = frame["volatility_7"].rolling(63, min_periods=30).median()
    volatility_mad = (frame["volatility_7"] - volatility_median).abs().rolling(63, min_periods=30).median()
    frame["volatility_regime_z"] = _rolling_zscore(frame["volatility_7"], 63).clip(-5, 5)
    trend_scale = (frame["volatility_21"] * np.sqrt(20)).replace(0, np.nan)
    frame["trend_regime_score"] = (frame["distance_ma20"] / trend_scale).clip(-5, 5)
    frame["trend_regime"] = np.select(
        [frame["trend_regime_score"] >= 0.5, frame["trend_regime_score"] <= -0.5],
        [1.0, -1.0],
        default=0.0,
    )
    frame["return_change_point"] = (
        (frame["market_return_1"] - return_median).abs() > (3.0 * 1.4826 * return_mad).clip(lower=0.008)
    ).astype(float)
    frame["volatility_change_point"] = (
        (frame["volatility_7"] - volatility_median).abs() > (2.5 * 1.4826 * volatility_mad).clip(lower=0.002)
    ).astype(float)
    frame["regime_transition"] = (
        frame["trend_regime"].ne(frame["trend_regime"].shift(1))
        | frame["return_change_point"].gt(0)
        | frame["volatility_change_point"].gt(0)
    ).astype(float)
    technical_columns.extend([
        "volatility_regime_z", "trend_regime_score", "trend_regime",
        "return_change_point", "volatility_change_point", "regime_transition",
    ])

    frame["index_available"] = frame[["index_BTC", "index_me"]].notna().all(axis=1).astype(float)
    frame["index_availability_imputed"] = frame[[
        "index_BTC_availability_imputed", "index_me_availability_imputed",
    ]].fillna(True).any(axis=1).astype(float)
    frame["astro_availability_imputed"] = frame["astro_availability_imputed"].fillna(True).astype(float)
    frame["index_btc_change_1"] = frame["index_BTC"].diff(1)
    frame["index_btc_change_3"] = frame["index_BTC"].diff(3)
    frame["index_btc_slope_3"] = _rolling_slope(frame["index_BTC"], 3)
    frame["index_btc_slope_7"] = _rolling_slope(frame["index_BTC"], 7)
    frame["index_btc_acceleration"] = frame["index_btc_slope_3"].diff()
    frame["index_btc_z30"] = _rolling_zscore(frame["index_BTC"], 30)
    frame["index_me_change_1"] = frame["index_me"].diff(1)
    frame["index_me_change_3"] = frame["index_me"].diff(3)
    frame["index_me_slope_3"] = _rolling_slope(frame["index_me"], 3)
    frame["index_me_slope_7"] = _rolling_slope(frame["index_me"], 7)
    frame["index_me_acceleration"] = frame["index_me_slope_3"].diff()
    frame["index_me_z30"] = _rolling_zscore(frame["index_me"], 30)
    frame["gap_index"] = frame["index_BTC"] - frame["index_me"]
    frame["gap_index_abs"] = frame["gap_index"].abs()
    frame["gap_index_change"] = frame["gap_index"].diff()
    frame["index_cross"] = (np.sign(frame["gap_index"]) != np.sign(frame["gap_index"].shift(1))).astype(float)
    btc_direction = np.sign(frame["index_btc_change_1"])
    trader_direction = np.sign(frame["index_me_change_1"])
    frame["same_phase"] = ((btc_direction == trader_direction) & (btc_direction != 0)).astype(float)
    frame["opposite_phase"] = ((btc_direction * trader_direction) < 0).astype(float)
    frame["index_btc_shock"] = (frame["index_btc_change_1"].abs() >= 18).astype(float)
    frame["index_me_shock"] = (frame["index_me_change_1"].abs() >= 18).astype(float)
    for window in (2, 5, 7):
        frame[f"index_corr_{window}"] = frame["index_BTC"].rolling(window).corr(frame["index_me"])
    previous_price_direction = np.sign(frame["market_return_1"])
    frame["btc_price_divergence"] = ((btc_direction * previous_price_direction) < 0).astype(float)
    frame["trader_price_divergence"] = ((trader_direction * previous_price_direction) < 0).astype(float)
    frame["btc_extreme_high"] = (frame["index_BTC"] >= 80).astype(float)
    frame["btc_extreme_low"] = (frame["index_BTC"] <= 20).astype(float)
    frame["trader_extreme_high"] = (frame["index_me"] >= 80).astype(float)
    frame["trader_extreme_low"] = (frame["index_me"] <= 20).astype(float)

    frame["astro_finance_z30"] = _rolling_zscore(frame["finance"], 30)
    frame["astro_volatility_z30"] = _rolling_zscore(frame["volatility"], 30)
    frame["astro_composite_change_1"] = frame["composite"].diff()
    frame["astro_composite_slope_3"] = _rolling_slope(frame["composite"], 3)
    frame["astro_event"] = frame["event"].fillna(False).astype(float)
    frame["astro_watch"] = frame["regime"].str.lower().eq("watch").astype(float)
    frame["astro_index_alignment"] = (
        np.sign(frame["astro_composite_change_1"]) == np.sign(frame["index_btc_change_1"])
    ).astype(float)
    frame["astro_index_tension"] = (
        np.sign(frame["astro_composite_change_1"]) * np.sign(frame["index_btc_change_1"]) < 0
    ).astype(float)
    frame["shock_confluence"] = (
        (frame["index_btc_shock"] > 0) | (frame["index_me_shock"] > 0)
    ).astype(float) * frame["astro_event"]
    frame["volatility_confluence"] = frame["astro_volatility_z30"].clip(-3, 3) * frame["atr14_pct"].fillna(0)

    day_of_week = frame["date"].dt.dayofweek
    day_of_year = frame["date"].dt.dayofyear
    frame["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["target"] = frame["daily_return"].map(direction_label)
    technical_columns = technical_columns + intraday_columns

    index_columns = [
        "index_available", "index_availability_imputed", "index_BTC", "index_me", "index_btc_change_1", "index_btc_change_3",
        "index_btc_slope_3", "index_btc_slope_7", "index_btc_acceleration", "index_btc_z30",
        "index_me_change_1", "index_me_change_3", "index_me_slope_3", "index_me_slope_7",
        "index_me_acceleration", "index_me_z30", "gap_index", "gap_index_abs", "gap_index_change",
        "index_cross", "same_phase", "opposite_phase", "index_btc_shock", "index_me_shock",
        "index_corr_2", "index_corr_5", "index_corr_7", "btc_price_divergence",
        "trader_price_divergence", "btc_extreme_high", "btc_extreme_low", "trader_extreme_high",
        "trader_extreme_low",
    ]
    astro_columns = [
        "finance", "career", "volatility", "composite", "astro_availability_imputed", "astro_finance_z30", "astro_volatility_z30",
        "astro_composite_change_1", "astro_composite_slope_3", "astro_event", "astro_watch",
    ]
    interaction_columns = [
        "astro_index_alignment", "astro_index_tension", "shock_confluence", "volatility_confluence",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    ]
    groups = {
        "technical": technical_columns,
        "index": index_columns,
        "astro": astro_columns,
        "interaction": interaction_columns,
        "external": external_columns,
    }
    groups["calendar"] = index_columns + astro_columns + interaction_columns
    groups["full"] = technical_columns + index_columns + astro_columns + interaction_columns + external_columns
    sequence_calendar_base = [
        "index_BTC", "index_me", "gap_index", "index_btc_change_1", "index_me_change_1",
        "index_btc_slope_3", "index_me_slope_3", "same_phase", "opposite_phase",
        "finance", "volatility", "composite", "astro_composite_change_1", "astro_event",
    ]
    sequence_full_base = sequence_calendar_base + [
        "market_return_1", "market_return_3", "volatility_7", "distance_ma20", "rsi14", "atr14_pct",
    ]
    sequence_full_base += [
        column for column in (
            "intraday_realized_vol_1h", "intraday_trend_1h", "intraday_signed_volume_1h",
            "intraday_realized_vol_4h", "intraday_trend_4h", "intraday_signed_volume_4h",
        ) if column in frame
    ]
    sequence_features = {
        f"sequence__{column}__lag_{lag}": frame[column].shift(lag)
        for column in dict.fromkeys(sequence_full_base)
        for lag in range(13, -1, -1)
    }
    frame = pd.concat([frame, pd.DataFrame(sequence_features, index=frame.index)], axis=1)
    groups["sequence_calendar"] = [
        f"sequence__{column}__lag_{lag}"
        for lag in range(13, -1, -1)
        for column in sequence_calendar_base
    ]
    groups["sequence_full"] = [
        f"sequence__{column}__lag_{lag}"
        for lag in range(13, -1, -1)
        for column in sequence_full_base
    ]
    groups["sequence_calendar_base"] = sequence_calendar_base
    groups["sequence_full_base"] = sequence_full_base
    return frame.reset_index(drop=True), groups


PATTERN_DEFINITIONS: dict[str, tuple[str, str]] = {
    "positive_return_change_point": ("(return_change_point > 0) & (market_return_1 > 0)", "Positive price change point before the target session"),
    "negative_return_change_point": ("(return_change_point > 0) & (market_return_1 < 0)", "Negative price change point before the target session"),
    "high_volatility_transition": ("(volatility_change_point > 0) & (volatility_regime_z > 0)", "Transition into a high-volatility regime"),
    "bull_trend_regime": ("trend_regime > 0", "Lagged market trend is in a positive regime"),
    "bear_trend_regime": ("trend_regime < 0", "Lagged market trend is in a negative regime"),
    "btc_extreme_high": ("index_BTC >= 80", "BTC psychology in extreme-high zone"),
    "btc_extreme_low": ("index_BTC <= 20", "BTC psychology in extreme-low zone"),
    "trader_extreme_high": ("index_me >= 80", "Trader energy in extreme-high zone"),
    "trader_extreme_low": ("index_me <= 20", "Trader energy in extreme-low zone"),
    "wide_gap_btc": ("gap_index >= 24", "BTC psychology leads trader energy by at least 24"),
    "wide_gap_trader": ("gap_index <= -24", "Trader energy leads BTC psychology by at least 24"),
    "cross_btc_above": ("(index_cross > 0) & (gap_index > 0)", "BTC psychology crosses above trader energy"),
    "cross_trader_above": ("(index_cross > 0) & (gap_index < 0)", "Trader energy crosses above BTC psychology"),
    "same_phase_accel_up": ("(same_phase > 0) & (index_btc_acceleration > 3) & (index_me_acceleration > 3)", "Both indices accelerate upward"),
    "same_phase_accel_down": ("(same_phase > 0) & (index_btc_acceleration < -3) & (index_me_acceleration < -3)", "Both indices accelerate downward"),
    "opposite_phase_shock": ("(opposite_phase > 0) & ((index_btc_shock > 0) | (index_me_shock > 0))", "Opposite-phase index shock"),
    "btc_shock_up": ("(index_btc_shock > 0) & (index_btc_change_1 > 0)", "BTC psychology positive shock"),
    "btc_shock_down": ("(index_btc_shock > 0) & (index_btc_change_1 < 0)", "BTC psychology negative shock"),
    "trader_shock_up": ("(index_me_shock > 0) & (index_me_change_1 > 0)", "Trader energy positive shock"),
    "trader_shock_down": ("(index_me_shock > 0) & (index_me_change_1 < 0)", "Trader energy negative shock"),
    "negative_corr": ("index_corr_5 <= -0.5", "Indices negatively correlated for five days"),
    "positive_corr_shock": ("(index_corr_5 >= 0.5) & ((index_btc_shock > 0) | (index_me_shock > 0))", "Positive index correlation with shock"),
    "btc_price_divergence": ("btc_price_divergence > 0", "BTC psychology diverges from prior price move"),
    "trader_price_divergence": ("trader_price_divergence > 0", "Trader energy diverges from prior price move"),
    "astro_event_high_vol": ("(astro_event > 0) & (astro_volatility_z30 >= 1)", "Astro event with elevated volatility score"),
    "astro_finance_high": ("astro_finance_z30 >= 1.25", "Astro finance score at rolling extreme high"),
    "astro_finance_low": ("astro_finance_z30 <= -1.25", "Astro finance score at rolling extreme low"),
    "astro_index_alignment": ("(astro_index_alignment > 0) & (index_btc_change_1 != 0)", "Astro composite and BTC psychology align"),
    "astro_index_tension": ("astro_index_tension > 0", "Astro composite and BTC psychology conflict"),
    "shock_confluence": ("shock_confluence > 0", "Index shock coincides with astro event"),
}


def reward_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for action_index, action in enumerate(CLASS_NAMES):
        for actual_index in range(3):
            sample = frame.loc[frame["target"] == actual_index, "daily_return"].dropna()
            if sample.empty:
                matrix[action_index, actual_index] = 1 / 3
            else:
                matrix[action_index, actual_index] = float(np.mean([grade_forecast(action, value)[1] for value in sample]))
    return matrix


def choose_direction(
    probabilities: np.ndarray,
    matrix: np.ndarray,
    policy_mode: str = "reward",
    sideway_penalty: float = 1.0,
) -> tuple[str, float, float]:
    utilities = probabilities.copy() if policy_mode == "probability" else matrix @ probabilities
    utilities[CLASS_TO_INDEX["sideway"]] *= sideway_penalty
    order = np.argsort(utilities)
    best = int(order[-1])
    margin = float(utilities[order[-1]] - utilities[order[-2]])
    return CLASS_NAMES[best], float(utilities[best]), margin


def allocate_monthly_directions(
    dates: Sequence[pd.Timestamp],
    probabilities: np.ndarray,
    matrix: np.ndarray,
    policy_mode: str = "reward",
    sideway_penalty: float = 1.0,
    max_sideway_per_month: int = MAX_SIDEWAY_PER_MONTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the decision policy while retaining only the strongest monthly sideway calls."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape (n_rows, 3)")
    if len(dates) != len(probabilities):
        raise ValueError("dates and probabilities must have the same length")

    utilities = probabilities.copy() if policy_mode == "probability" else probabilities @ matrix.T
    utilities[:, CLASS_TO_INDEX["sideway"]] *= sideway_penalty
    selected = np.argmax(utilities, axis=1).astype(int)
    overridden = np.zeros(len(selected), dtype=bool)
    month_keys = pd.to_datetime(pd.Series(list(dates))).dt.strftime("%Y-%m").to_numpy()
    sideway_index = CLASS_TO_INDEX["sideway"]

    for month_key in pd.unique(month_keys):
        candidates = np.flatnonzero((month_keys == month_key) & (selected == sideway_index))
        if len(candidates) <= max_sideway_per_month:
            continue
        directional_best = np.maximum(utilities[candidates, 0], utilities[candidates, 2])
        sideway_advantage = utilities[candidates, sideway_index] - directional_best
        order = np.argsort(-sideway_advantage, kind="stable")
        rejected = candidates[order[max_sideway_per_month:]]
        selected[rejected] = np.where(
            utilities[rejected, CLASS_TO_INDEX["up"]] > utilities[rejected, CLASS_TO_INDEX["down"]],
            CLASS_TO_INDEX["up"],
            CLASS_TO_INDEX["down"],
        )
        overridden[rejected] = True

    expected_scores = utilities[np.arange(len(selected)), selected]
    margins = np.empty(len(selected), dtype=float)
    for row_index, direction_index in enumerate(selected):
        alternatives = np.delete(utilities[row_index], direction_index)
        margins[row_index] = expected_scores[row_index] - float(np.max(alternatives))
    directions = np.asarray(CLASS_NAMES, dtype=object)[selected]
    return directions, expected_scores, margins, overridden


def select_decision_policy(
    probabilities: np.ndarray,
    daily_returns: Sequence[float],
    matrix: np.ndarray,
    dates: Sequence[pd.Timestamp] | None = None,
) -> tuple[str, float, dict[str, float]]:
    returns = np.asarray(daily_returns, dtype=float)
    policy_dates = dates if dates is not None else pd.date_range("2000-01-01", periods=len(probabilities), freq="D")
    candidates = [
        (mode, penalty)
        for mode in ("probability", "reward")
        for penalty in (0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
    ]
    rows: list[tuple[float, str, float, dict[str, float]]] = []
    for mode, penalty in candidates:
        directions, _, _, _ = allocate_monthly_directions(
            policy_dates, probabilities, matrix, mode, penalty,
        )
        grades = [grade_forecast(direction, move) for direction, move in zip(directions, returns)]
        exact = float(np.mean([status == "correct" for status, _ in grades]))
        weighted = float(np.mean([score for _, score in grades]))
        directional = float(np.mean(
            ((directions == "up") & (returns > 0))
            | ((directions == "down") & (returns < 0))
            | ((directions == "sideway") & (np.abs(returns) <= SIDEWAY_LIMIT))
        ))
        strategy = np.where(
            directions == "up",
            returns - TRADING_COST,
            np.where(directions == "down", -returns - TRADING_COST, 0.0),
        )
        expectancy = float(np.mean(strategy))
        largest_share = float(pd.Series(directions).value_counts(normalize=True).max())
        diversity = 1 - largest_share
        expectancy_scaled = float(np.clip(expectancy, -0.002, 0.002) / 0.004 + 0.5)
        objective = directional * 0.44 + weighted * 0.25 + exact * 0.10 + expectancy_scaled * 0.16 + diversity * 0.05
        diagnostics = {
            "objective": objective,
            "directional_accuracy": directional,
            "weighted_accuracy": weighted,
            "exact_accuracy": exact,
            "expectancy": expectancy,
            "largest_direction_share": largest_share,
        }
        rows.append((objective, mode, penalty, diagnostics))
    positive_expectancy = [row for row in rows if row[3]["expectancy"] > 0]
    _, mode, penalty, diagnostics = max(positive_expectancy or rows, key=lambda row: row[0])
    return mode, penalty, diagnostics


def build_pattern_registry(train: pd.DataFrame, minimum_occurrences: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    direction_baselines = {
        direction: float(np.mean([grade_forecast(direction, value)[1] for value in train["daily_return"].dropna()]))
        for direction in CLASS_NAMES
    }
    for pattern_id, (expression, label) in PATTERN_DEFINITIONS.items():
        try:
            mask = train.eval(expression, engine="python").fillna(False).astype(bool)
        except Exception:
            continue
        sample = train.loc[mask & train["daily_return"].notna()]
        if sample.empty:
            continue
        for direction in CLASS_NAMES:
            grades = [grade_forecast(direction, value) for value in sample["daily_return"]]
            scores = np.array([grade[1] for grade in grades], dtype=float)
            statuses = [grade[0] for grade in grades]
            smoothed_score = float((scores.sum() + 2.0) / (len(scores) + 4.0))
            exact_accuracy = float((np.array(statuses) == "correct").mean())
            baseline = direction_baselines[direction]
            standard_error = math.sqrt(max(1e-8, baseline * (1 - baseline) / max(1, len(scores))))
            z_score = (float(scores.mean()) - baseline) / standard_error
            p_value = float(0.5 * math.erfc(z_score / math.sqrt(2)))
            strategy = np.where(
                direction == "up",
                sample["daily_return"].to_numpy() - TRADING_COST,
                np.where(direction == "down", -sample["daily_return"].to_numpy() - TRADING_COST, 0.0),
            )
            rows.append({
                "pattern_id": pattern_id,
                "pattern": label,
                "expression": expression,
                "direction": direction,
                "occurrences": int(len(sample)),
                "weighted_accuracy": smoothed_score,
                "weighted_lcb": conservative_beta_lower_bound(
                    scores,
                    baseline,
                    prior_strength=12.0,
                ),
                "exact_accuracy": exact_accuracy,
                "baseline_score": baseline,
                "p_value": p_value,
                "expectancy": float(np.mean(strategy)),
                "last_seen": sample["date"].max(),
                "examples": sample.sort_values("date").tail(6)["date"].dt.strftime("%Y-%m-%d").tolist(),
            })
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    order = registry["p_value"].sort_values().index
    ranked_p = registry.loc[order, "p_value"].to_numpy(dtype=float)
    adjusted = np.minimum.accumulate(
        (ranked_p * len(ranked_p) / np.arange(1, len(ranked_p) + 1))[::-1],
    )[::-1]
    registry["false_discovery_q"] = 1.0
    registry.loc[order, "false_discovery_q"] = np.clip(adjusted, 0, 1)
    registry["eligible"] = (
        (registry["occurrences"] >= minimum_occurrences)
        & (registry["weighted_accuracy"] >= 0.36)
        & ((registry["false_discovery_q"] <= 0.25) | (registry["occurrences"] >= 40))
    )
    registry["rank_score"] = (
        registry["weighted_lcb"] * np.log1p(registry["occurrences"])
        + np.clip(registry["expectancy"], -0.03, 0.03) * 8
    )
    registry = registry.sort_values(["eligible", "rank_score", "occurrences"], ascending=False).reset_index(drop=True)
    registry["rank"] = np.arange(1, len(registry) + 1)
    registry["status"] = np.where(registry["eligible"] & (registry["rank"] <= 16), "active", "standby")
    return registry


def pattern_probabilities(frame: pd.DataFrame, registry: pd.DataFrame) -> np.ndarray:
    probabilities = np.full((len(frame), 3), 1 / 3, dtype=float)
    if registry.empty or frame.empty:
        return probabilities
    active_mask = registry["eligible"].astype(bool)
    if "status" in registry:
        active_mask &= registry["status"].eq("active")
    active = registry[active_mask].copy()
    action_votes = np.zeros((len(frame), 3), dtype=float)
    for _, rule in active.iterrows():
        try:
            mask = frame.eval(str(rule["expression"]), engine="python").fillna(False).to_numpy(dtype=bool)
        except Exception:
            continue
        if not mask.any():
            continue
        action_index = CLASS_TO_INDEX[str(rule["direction"])]
        conservative_accuracy = float(rule.get("weighted_lcb", rule["weighted_accuracy"]))
        baseline = float(rule.get("baseline_score", 0.25))
        weight = max(0.01, conservative_accuracy - baseline + 0.05) * math.log1p(float(rule["occurrences"]))
        action_votes[mask, action_index] += weight
    totals = action_votes.sum(axis=1)
    valid = totals > 0
    probabilities[valid] = (action_votes[valid] + 0.25) / (totals[valid, None] + 0.75)
    return probabilities


def _align_probabilities(model: object, probabilities: np.ndarray) -> np.ndarray:
    output = np.zeros((len(probabilities), 3), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is None:
        return probabilities
    for source_index, class_value in enumerate(classes):
        output[:, int(class_value)] = probabilities[:, source_index]
    return output


def candidate_models(feature_columns: Sequence[str], random_state: int = 42) -> dict[str, object]:
    """Compatibility view used by explainability; return classifiers with categorical targets."""
    return {
        spec.name: spec.estimator
        for spec in candidate_model_specs(feature_columns, random_state)
        if spec.target_column == "target"
    }


def monthly_purged_folds(
    frame: pd.DataFrame,
    oos_start: pd.Timestamp = OOS_START,
    purge_days: int = 5,
    calibration_days: int = 90,
    rolling_days: int | None = 1460,
    horizon_days: int = 1,
    embargo_days: int = 0,
) -> list[Fold]:
    folds: list[Fold] = []
    labeled = frame[frame["target"].notna()].copy()
    latest = labeled["date"].max()
    month_starts = pd.date_range(oos_start.to_period("M").start_time, latest.to_period("M").start_time, freq="MS")
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(1)
        test_mask = labeled["date"].between(month_start, month_end)
        test_index = labeled.index[test_mask].to_numpy()
        if len(test_index) == 0:
            continue
        effective_gap = max(purge_days, horizon_days, embargo_days)
        train_cutoff = month_start - pd.Timedelta(days=effective_gap)
        train_start = (
            train_cutoff - pd.Timedelta(days=rolling_days)
            if rolling_days is not None else labeled["date"].min()
        )
        eligible = labeled[labeled["date"].between(train_start, train_cutoff, inclusive="left")]
        calibration_start = train_cutoff - pd.Timedelta(days=calibration_days)
        calibration_index = eligible.index[eligible["date"] >= calibration_start].to_numpy()
        train_index = eligible.index[eligible["date"] < calibration_start].to_numpy()
        if len(train_index) < 365 or len(calibration_index) < 45:
            continue
        folds.append(Fold(month_start.strftime("%Y-%m"), train_index, calibration_index, test_index))
    return folds


def _safe_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(log_loss(labels, probabilities, labels=[0, 1, 2]))
    except ValueError:
        return math.nan


def _decision_rows(
    dates: Sequence[pd.Timestamp],
    probabilities: np.ndarray,
    returns: Sequence[float],
    matrix: np.ndarray,
    model: str,
    fold_id: str,
    dynamic_no_call: bool,
    volatility: Sequence[float] | None = None,
    policy_mode: str = "reward",
    sideway_penalty: float = 1.0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    monthly_abstentions: defaultdict[str, int] = defaultdict(int)
    volatility_array = np.asarray(volatility if volatility is not None else np.full(len(dates), np.nan), dtype=float)
    finite_volatility = volatility_array[np.isfinite(volatility_array)]
    median_volatility = float(np.nanmedian(finite_volatility)) if finite_volatility.size else 0.03
    directions, expected_scores, margins, sideway_overrides = allocate_monthly_directions(
        dates, probabilities, matrix, policy_mode, sideway_penalty,
    )
    for index, (date, probability, daily_return) in enumerate(zip(dates, probabilities, returns)):
        direction = str(directions[index])
        expected_score = float(expected_scores[index])
        margin = float(margins[index])
        entropy = float(-np.sum(np.clip(probability, 1e-9, 1) * np.log(np.clip(probability, 1e-9, 1))) / np.log(3))
        volatility_ratio = volatility_array[index] / median_volatility if median_volatility > 0 else 1.0
        uncertainty_threshold = 0.965 - 0.015 * np.clip(volatility_ratio - 1, -1, 1)
        month_key = pd.Timestamp(date).strftime("%Y-%m")
        no_call = bool(dynamic_no_call and entropy >= uncertainty_threshold and margin < 0.025 and monthly_abstentions[month_key] < 6)
        if no_call:
            monthly_abstentions[month_key] += 1
            status, score = "no-call", math.nan
            strategy_return = 0.0
            chosen = "no-call"
        else:
            status, score = grade_forecast(direction, float(daily_return))
            chosen = direction
            if direction == "up":
                strategy_return = float(daily_return) - TRADING_COST
            elif direction == "down":
                strategy_return = -float(daily_return) - TRADING_COST
            else:
                strategy_return = 0.0
        directional_hit = bool(
            (direction == "up" and daily_return > 0)
            or (direction == "down" and daily_return < 0)
            or (direction == "sideway" and abs(float(daily_return)) <= SIDEWAY_LIMIT)
        )
        records.append({
            "date": pd.Timestamp(date),
            "model": model,
            "fold": fold_id,
            "forecast": chosen,
            "status": status,
            "score": score,
            "daily_return": float(daily_return),
            "actual_class": int(direction_label(float(daily_return))),
            "directional_hit": directional_hit,
            "strategy_return": strategy_return,
            "prob_down": float(probability[0]),
            "prob_sideway": float(probability[1]),
            "prob_up": float(probability[2]),
            "expected_score": expected_score,
            "decision_margin": margin,
            "entropy": entropy,
            "policy_mode": policy_mode,
            "sideway_penalty": sideway_penalty,
            "sideway_cap_override": bool(sideway_overrides[index]),
        })
    return records


def _fit_analog_probabilities(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    neighbors: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train[list(feature_columns)]))
    calibration_values = scaler.transform(imputer.transform(calibration[list(feature_columns)]))
    test_values = scaler.transform(imputer.transform(test[list(feature_columns)]))
    count = min(neighbors, len(train_values))
    model = NearestNeighbors(n_neighbors=count, metric="euclidean")
    model.fit(train_values)
    labels = train["target"].astype(int).to_numpy()

    def predict(values: np.ndarray) -> np.ndarray:
        distances, indices = model.kneighbors(values)
        output = np.zeros((len(values), 3), dtype=float)
        for row_index in range(len(values)):
            weights = 1 / (distances[row_index] + 0.25)
            for neighbor_index, weight in zip(indices[row_index], weights):
                output[row_index, labels[neighbor_index]] += weight
            output[row_index] += 0.35
            output[row_index] /= output[row_index].sum()
        return output

    return predict(calibration_values), predict(test_values)


def _metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in predictions.groupby("model"):
        calls = group[group["forecast"] != "no-call"].copy()
        if calls.empty:
            continue
        exact_accuracy = float((calls["status"] == "correct").mean())
        weighted_accuracy = float(calls["score"].mean())
        sign_hit = calls["directional_hit"].astype(bool)
        strategy = calls["strategy_return"].fillna(0).to_numpy(dtype=float)
        equity = np.cumprod(1 + strategy)
        running_peak = np.maximum.accumulate(equity)
        drawdown = equity / running_peak - 1
        positive = strategy[strategy > 0].sum()
        negative = -strategy[strategy < 0].sum()
        sharpe = float(np.mean(strategy) / np.std(strategy, ddof=1) * np.sqrt(365)) if len(strategy) > 2 and np.std(strategy, ddof=1) > 0 else 0.0
        probability_columns = calls[["prob_down", "prob_sideway", "prob_up"]].to_numpy()
        actual = calls["actual_class"].astype(int).to_numpy()
        brier = float(np.mean(np.sum((probability_columns - np.eye(3)[actual]) ** 2, axis=1)))
        probability_diagnostics = multiclass_diagnostics(probability_columns, actual)
        positions = calls["forecast"].map({"down": -1.0, "sideway": 0.0, "up": 1.0}).to_numpy(dtype=float)
        turnover = float(np.abs(np.diff(positions)).mean()) if len(positions) > 1 else 0.0
        intervals = moving_block_confidence_intervals(
            calls,
            random_state=42 + sum(ord(character) for character in str(model)),
        )
        row = {
            "model": model,
            "observations": int(len(group)),
            "calls": int(len(calls)),
            "coverage": float(len(calls) / len(group)),
            "no_calls": int((group["forecast"] == "no-call").sum()),
            "exact_accuracy": exact_accuracy,
            "weighted_accuracy": weighted_accuracy,
            "directional_accuracy": float(sign_hit.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(actual, probability_columns.argmax(axis=1))),
            "mcc": probability_diagnostics["mcc"],
            "brier": brier,
            "log_loss": probability_diagnostics["log_loss"],
            "ece": probability_diagnostics["ece"],
            "sharpe": sharpe,
            "profit_factor": float(positive / negative) if negative > 0 else math.inf,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "expectancy": float(strategy.mean()) if len(strategy) else 0.0,
            "net_return": float(equity[-1] - 1) if len(equity) else 0.0,
            "turnover": turnover,
        }
        row.update(intervals)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    sharpe_scaled = metrics["sharpe"].clip(-2, 3).add(2).div(5)
    expectancy_scaled = metrics["expectancy"].clip(-0.01, 0.01).add(0.01).div(0.02)
    metrics["rank_score"] = (
        metrics["weighted_accuracy"] * 0.32
        + metrics["directional_accuracy"] * 0.28
        + sharpe_scaled * 0.18
        + expectancy_scaled * 0.12
        + metrics["coverage"] * 0.10
    )
    metrics = metrics.sort_values("rank_score", ascending=False).reset_index(drop=True)
    metrics["rank"] = np.arange(1, len(metrics) + 1)
    metrics["status"] = np.where(
        (metrics["expectancy"] > 0) & (metrics["rank"] <= max(3, len(metrics) // 2)),
        "active",
        "standby",
    )
    return metrics


def run_walk_forward(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    analog_columns: Sequence[str],
    lane: str,
    sequence_columns: Sequence[str] | None = None,
    sequence_base_count: int | None = None,
    include_deep: bool = False,
) -> BacktestResult:
    folds = monthly_purged_folds(frame)
    ensemble_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    latest_registry = pd.DataFrame()
    prior_policy_probabilities: list[np.ndarray] = []
    prior_policy_returns: list[np.ndarray] = []
    prior_policy_dates: list[np.ndarray] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = frame.loc[fold.train_index].copy()
        calibration = frame.loc[fold.calibration_index].copy()
        test = frame.loc[fold.test_index].copy()
        calibration_split = max(30, min(len(calibration) - 20, int(len(calibration) * 0.67)))
        calibration_fit = calibration.iloc[:calibration_split].copy()
        policy_validation = calibration.iloc[calibration_split:].copy()
        if len(calibration_fit) < 30 or len(policy_validation) < 15:
            continue
        matrix = reward_matrix(train)
        candidates: list[CandidatePrediction] = []
        model_specs: list[tuple[str, object, Sequence[str], str]] = [
            (spec.name, spec.estimator, spec.columns, spec.target_column)
            for spec in candidate_model_specs(feature_columns, random_state=42 + fold_number)
        ]
        if include_deep and sequence_columns and sequence_base_count:
            try:
                from research.deep_models import TorchSequenceClassifier

                model_specs.extend([
                    (
                        "LSTM",
                        TorchSequenceClassifier(
                            architecture="lstm",
                            input_features=sequence_base_count,
                            random_state=42 + fold_number,
                        ),
                        sequence_columns,
                        "target",
                    ),
                    (
                        "Transformer",
                        TorchSequenceClassifier(
                            architecture="transformer",
                            input_features=sequence_base_count,
                            random_state=84 + fold_number,
                        ),
                        sequence_columns,
                        "target",
                    ),
                ])
            except ImportError:
                pass
        for name, model, model_columns, target_column in model_specs:
            fitted = clone(model)
            try:
                target_values = (
                    train[target_column].astype(int)
                    if target_column == "target"
                    else train[target_column].astype(float)
                )
                fitted.fit(train[list(model_columns)], target_values)
                calibration_fit_raw = _align_probabilities(
                    fitted, fitted.predict_proba(calibration_fit[list(model_columns)]),
                )
                policy_raw = _align_probabilities(
                    fitted, fitted.predict_proba(policy_validation[list(model_columns)]),
                )
                test_raw = _align_probabilities(fitted, fitted.predict_proba(test[list(model_columns)]))
                calibrator = ProbabilityCalibrator().fit(
                    calibration_fit_raw, calibration_fit["target"].astype(int).to_numpy(),
                )
                calibration_probabilities = calibrator.transform(policy_raw)
                test_probabilities = calibrator.transform(test_raw)
            except Exception:
                continue
            calibration_rows = _decision_rows(
                policy_validation["date"], calibration_probabilities, policy_validation["daily_return"], matrix,
                name, fold.fold_id, False,
            )
            calibration_score = float(pd.DataFrame(calibration_rows)["score"].mean())
            calibration_loss = _safe_log_loss(
                policy_validation["target"].astype(int).to_numpy(), calibration_probabilities,
            )
            quality = max(0.015, calibration_score - 0.25) * math.exp(-max(0.0, calibration_loss - 0.8))
            candidates.append(CandidatePrediction(
                name, calibration_probabilities, test_probabilities, calibration_score, calibration_loss, quality,
                calibrator.diagnostics_.method, calibrator.diagnostics_.selection_log_loss,
            ))

        training_registry = build_pattern_registry(train)
        latest_registry = build_pattern_registry(pd.concat([train, calibration], ignore_index=True))
        pattern_calibration = pattern_probabilities(policy_validation, training_registry)
        pattern_test = pattern_probabilities(test, latest_registry)
        pattern_rows = _decision_rows(
            policy_validation["date"], pattern_calibration, policy_validation["daily_return"], matrix,
            "Pattern Registry", fold.fold_id, False,
        )
        pattern_score = float(pd.DataFrame(pattern_rows)["score"].mean())
        candidates.append(CandidatePrediction(
            "Pattern Registry", pattern_calibration, pattern_test, pattern_score,
            _safe_log_loss(policy_validation["target"].astype(int).to_numpy(), pattern_calibration),
            max(0.01, pattern_score - 0.25),
        ))

        try:
            analog_calibration, analog_test = _fit_analog_probabilities(
                train, policy_validation, test, analog_columns,
            )
            analog_rows = _decision_rows(
                policy_validation["date"], analog_calibration, policy_validation["daily_return"], matrix,
                "Historical Analog", fold.fold_id, False,
            )
            analog_score = float(pd.DataFrame(analog_rows)["score"].mean())
            candidates.append(CandidatePrediction(
                "Historical Analog", analog_calibration, analog_test, analog_score,
                _safe_log_loss(policy_validation["target"].astype(int).to_numpy(), analog_calibration),
                max(0.01, analog_score - 0.25),
            ))
        except Exception:
            pass

        if not candidates:
            continue
        gated = candidates
        weights, stacking_diagnostics = learn_simplex_weights(
            [candidate.calibration_probabilities for candidate in gated],
            policy_validation["target"].astype(int).to_numpy(),
        )
        ensemble_calibration = sum(
            weight * candidate.calibration_probabilities for weight, candidate in zip(weights, gated)
        )
        ensemble_probabilities = sum(weight * candidate.test_probabilities for weight, candidate in zip(weights, gated))
        if sum(len(values) for values in prior_policy_returns) >= 120:
            past_probabilities = np.concatenate(prior_policy_probabilities, axis=0)[-365:]
            past_returns = np.concatenate(prior_policy_returns, axis=0)[-365:]
            past_dates = np.concatenate(prior_policy_dates, axis=0)[-365:]
            policy_probabilities = np.concatenate([past_probabilities, ensemble_calibration], axis=0)
            policy_returns = np.concatenate([past_returns, policy_validation["daily_return"].to_numpy(dtype=float)])
            policy_dates = np.concatenate([past_dates, policy_validation["date"].to_numpy()], axis=0)
            policy_mode, sideway_penalty, policy_diagnostics = select_decision_policy(
                policy_probabilities, policy_returns, matrix, policy_dates,
            )
        else:
            policy_mode, sideway_penalty = "probability", 1.0
            _, _, policy_diagnostics = select_decision_policy(
                ensemble_calibration, policy_validation["daily_return"], matrix, policy_validation["date"],
            )
        ensemble_rows = _decision_rows(
            test["date"], ensemble_probabilities, test["daily_return"], matrix,
            f"{lane} Ensemble", fold.fold_id, True, test["volatility_7"], policy_mode, sideway_penalty,
        )
        ensemble_records.extend(ensemble_rows)
        prior_policy_probabilities.append(ensemble_probabilities)
        prior_policy_returns.append(test["daily_return"].to_numpy(dtype=float))
        prior_policy_dates.append(test["date"].to_numpy())

        for candidate in candidates:
            model_records.extend(_decision_rows(
                test["date"], candidate.test_probabilities, test["daily_return"], matrix,
                candidate.name, fold.fold_id, False, test["volatility_7"], policy_mode, sideway_penalty,
            ))
        model_records.extend(ensemble_rows)
        fold_frame = pd.DataFrame(ensemble_rows)
        calls = fold_frame[fold_frame["forecast"] != "no-call"]
        fold_rows.append({
            "lane": lane,
            "fold": fold.fold_id,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "calibration_fit_end": calibration_fit["date"].max(),
            "policy_start": policy_validation["date"].min(),
            "calibration_end": policy_validation["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
            "calls": int(len(calls)),
            "no_calls": int((fold_frame["forecast"] == "no-call").sum()),
            "exact_accuracy": float((calls["status"] == "correct").mean()) if len(calls) else math.nan,
            "weighted_accuracy": float(calls["score"].mean()) if len(calls) else math.nan,
            "expectancy": float(calls["strategy_return"].mean()) if len(calls) else math.nan,
            "members": [candidate.name for candidate in gated],
            "weights": [float(value) for value in weights],
            "stacking_method": stacking_diagnostics["method"],
            "stacking_log_loss": stacking_diagnostics.get("loss"),
            "uniform_ensemble_log_loss": stacking_diagnostics.get("uniform_loss"),
            "calibration_methods": {
                candidate.name: candidate.calibration_method for candidate in candidates
            },
            "policy_mode": policy_mode,
            "sideway_penalty": sideway_penalty,
            "policy_calibration_directional_accuracy": policy_diagnostics["directional_accuracy"],
            "policy_calibration_expectancy": policy_diagnostics["expectancy"],
        })

    forecasts = pd.DataFrame(ensemble_records).sort_values("date").reset_index(drop=True)
    model_predictions = pd.DataFrame(model_records).sort_values(["date", "model"]).reset_index(drop=True)
    metrics = _metric_summary(model_predictions)
    fold_metrics = pd.DataFrame(fold_rows)
    no_calls = (
        forecasts.assign(month=forecasts["date"].dt.strftime("%Y-%m"))
        .groupby("month")
        .agg(days=("date", "size"), no_calls=("forecast", lambda values: (values == "no-call").sum()))
        .reset_index()
    ) if not forecasts.empty else pd.DataFrame()
    importance = fit_permutation_importance(frame, feature_columns)
    return BacktestResult(forecasts, model_predictions, metrics, fold_metrics, importance, latest_registry, no_calls)


def fit_permutation_importance(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    labeled = frame[frame["target"].notna()].copy()
    if len(labeled) < 500:
        return pd.DataFrame()
    train = labeled.iloc[:-180]
    validation = labeled.iloc[-180:]
    candidates = candidate_models(feature_columns)
    preferred = "XGBoost" if "XGBoost" in candidates else "Random Forest"
    model = candidates[preferred]
    try:
        model.fit(train[list(feature_columns)], train["target"].astype(int))
        if preferred in {"XGBoost", "LightGBM", "Random Forest"}:
            try:
                import shap

                transformed = model.named_steps["preprocess"].transform(validation[list(feature_columns)].tail(140))
                estimator = model.named_steps["model"]
                values = np.asarray(shap.TreeExplainer(estimator).shap_values(transformed))
                if values.ndim == 3 and values.shape[1] == len(feature_columns):
                    importance_values = np.abs(values).mean(axis=(0, 2))
                elif values.ndim == 3 and values.shape[2] == len(feature_columns):
                    importance_values = np.abs(values).mean(axis=(0, 1))
                elif values.ndim == 2:
                    importance_values = np.abs(values).mean(axis=0)
                else:
                    raise ValueError("Unsupported SHAP output shape")
                output = pd.DataFrame({
                    "feature": list(feature_columns),
                    "importance": importance_values,
                    "importance_std": 0.0,
                    "model": preferred,
                    "method": "SHAP TreeExplainer on purged holdout",
                })
                return output.sort_values("importance", ascending=False).head(24).reset_index(drop=True)
            except Exception:
                pass
        result = permutation_importance(
            model,
            validation[list(feature_columns)],
            validation["target"].astype(int),
            n_repeats=3,
            random_state=42,
            scoring="neg_log_loss",
            n_jobs=1,
        )
    except Exception:
        return pd.DataFrame()
    output = pd.DataFrame({
        "feature": list(feature_columns),
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
        "model": preferred,
        "method": "purged holdout permutation",
    })
    return output.sort_values("importance", ascending=False).head(24).reset_index(drop=True)


def _analog_forecast_bundle(
    history: pd.DataFrame,
    calibration: pd.DataFrame,
    future: pd.DataFrame,
    feature_columns: Sequence[str],
    neighbors: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, object]]]]:
    history_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    history_scaler = StandardScaler()
    history_values = history_scaler.fit_transform(history_imputer.fit_transform(history[list(feature_columns)]))
    calibration_values = history_scaler.transform(history_imputer.transform(calibration[list(feature_columns)]))
    history_count = min(neighbors, len(history_values))
    history_nearest = NearestNeighbors(n_neighbors=history_count, metric="euclidean").fit(history_values)
    calibration_distances, calibration_indices = history_nearest.kneighbors(calibration_values)
    history_labels = history["target"].astype(int).to_numpy()
    calibration_probabilities = np.zeros((len(calibration), 3), dtype=float)
    for row_index in range(len(calibration)):
        weights = 1 / (calibration_distances[row_index] + 0.25)
        for neighbor_index, weight in zip(calibration_indices[row_index], weights):
            calibration_probabilities[row_index, history_labels[neighbor_index]] += weight
        calibration_probabilities[row_index] += 0.35
        calibration_probabilities[row_index] /= calibration_probabilities[row_index].sum()
    development = pd.concat([history, calibration], ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    history_values = scaler.fit_transform(imputer.fit_transform(development[list(feature_columns)]))
    future_values = scaler.transform(imputer.transform(future[list(feature_columns)]))
    count = min(neighbors, len(history_values))
    nearest = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(history_values)
    distances, indices = nearest.kneighbors(future_values)
    labels = development["target"].astype(int).to_numpy()
    probabilities = np.zeros((len(future), 3), dtype=float)
    details: list[list[dict[str, object]]] = []
    for row_index in range(len(future)):
        weights = 1 / (distances[row_index] + 0.25)
        cases: list[dict[str, object]] = []
        for neighbor_index, weight, distance in zip(indices[row_index], weights, distances[row_index]):
            probabilities[row_index, labels[neighbor_index]] += weight
            neighbor = development.iloc[int(neighbor_index)]
            cases.append({
                "date": pd.Timestamp(neighbor["date"]).strftime("%Y-%m-%d"),
                "move": float(neighbor["daily_return"]),
                "outcome": CLASS_NAMES[int(neighbor["target"])],
                "similarity": float(1 / (1 + distance)),
            })
        probabilities[row_index] += 0.35
        probabilities[row_index] /= probabilities[row_index].sum()
        details.append(cases[:8])
    return calibration_probabilities, probabilities, details


def _matching_pattern_sets(frame: pd.DataFrame, registry: pd.DataFrame) -> list[list[dict[str, object]]]:
    matches: list[list[dict[str, object]]] = [[] for _ in range(len(frame))]
    if registry.empty:
        return matches
    active_mask = registry["eligible"].astype(bool)
    if "status" in registry:
        active_mask &= registry["status"].eq("active")
    score_column = "adaptive_rank_score" if "adaptive_rank_score" in registry else "rank_score"
    accuracy_column = "adjusted_weighted_accuracy" if "adjusted_weighted_accuracy" in registry else "weighted_accuracy"
    eligible = registry[active_mask].sort_values(score_column, ascending=False)
    for _, rule in eligible.iterrows():
        try:
            mask = frame.eval(str(rule["expression"]), engine="python").fillna(False).to_numpy(dtype=bool)
        except Exception:
            continue
        for row_index in np.flatnonzero(mask):
            matches[int(row_index)].append({
                "id": str(rule["pattern_id"]),
                "name": str(rule["pattern"]),
                "direction": str(rule["direction"]),
                "occurrences": int(rule["occurrences"]),
                "weighted_accuracy": float(rule[accuracy_column]),
                "rank": int(rule["rank"]),
            })
    return matches


def fit_latest_forecasts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    analog_columns: Sequence[str],
    lane: str,
    sequence_columns: Sequence[str] | None = None,
    sequence_base_count: int | None = None,
    include_deep: bool = False,
    max_future_days: int | None = None,
    policy_history: pd.DataFrame | None = None,
    preferred_models: Sequence[str] | None = None,
    pattern_adjuster: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = frame[frame["target"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    latest_market_date = labeled["date"].max()
    future = frame[frame["date"] > latest_market_date].copy()
    if max_future_days is not None:
        future = future.head(max_future_days)
    if future.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    calibration_start = latest_market_date - pd.Timedelta(days=90)
    training_end = calibration_start - pd.Timedelta(days=5)
    training_start = training_end - pd.Timedelta(days=1460)
    train = labeled[labeled["date"].between(training_start, training_end, inclusive="left")].copy()
    calibration = labeled[labeled["date"].between(calibration_start, latest_market_date)].copy()
    calibration_split = max(30, min(len(calibration) - 20, int(len(calibration) * 0.67)))
    calibration_fit = calibration.iloc[:calibration_split].copy()
    policy_validation = calibration.iloc[calibration_split:].copy()
    if len(calibration_fit) < 30 or len(policy_validation) < 15:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    development = pd.concat([train, calibration], ignore_index=True)
    matrix = reward_matrix(train)
    candidates: list[CandidatePrediction] = []
    future_probabilities_by_model: dict[str, np.ndarray] = {}
    model_specs: list[tuple[str, object, Sequence[str], str]] = [
        (spec.name, spec.estimator, spec.columns, spec.target_column)
        for spec in candidate_model_specs(feature_columns)
    ]
    if include_deep and sequence_columns and sequence_base_count:
        try:
            from research.deep_models import TorchSequenceClassifier

            model_specs.extend([
                (
                    "LSTM",
                    TorchSequenceClassifier(architecture="lstm", input_features=sequence_base_count),
                    sequence_columns,
                    "target",
                ),
                (
                    "Transformer",
                    TorchSequenceClassifier(
                        architecture="transformer", input_features=sequence_base_count, random_state=84,
                    ),
                    sequence_columns,
                    "target",
                ),
            ])
        except ImportError:
            pass
    for name, model, model_columns, target_column in model_specs:
        initial = clone(model)
        try:
            train_target = (
                train[target_column].astype(int)
                if target_column == "target"
                else train[target_column].astype(float)
            )
            initial.fit(train[list(model_columns)], train_target)
            calibration_fit_raw = _align_probabilities(
                initial, initial.predict_proba(calibration_fit[list(model_columns)]),
            )
            policy_raw = _align_probabilities(
                initial, initial.predict_proba(policy_validation[list(model_columns)]),
            )
            calibrator = ProbabilityCalibrator().fit(
                calibration_fit_raw, calibration_fit["target"].astype(int).to_numpy(),
            )
            calibration_probabilities = calibrator.transform(policy_raw)
            final_model = clone(model)
            development_target = (
                development[target_column].astype(int)
                if target_column == "target"
                else development[target_column].astype(float)
            )
            final_model.fit(development[list(model_columns)], development_target)
            future_raw = _align_probabilities(final_model, final_model.predict_proba(future[list(model_columns)]))
            future_probabilities = calibrator.transform(future_raw)
        except Exception:
            continue
        calibration_rows = _decision_rows(
            policy_validation["date"], calibration_probabilities, policy_validation["daily_return"], matrix,
            name, "latest", False,
        )
        calibration_score = float(pd.DataFrame(calibration_rows)["score"].mean())
        calibration_loss = _safe_log_loss(
            policy_validation["target"].astype(int).to_numpy(), calibration_probabilities,
        )
        quality = max(0.015, calibration_score - 0.25) * math.exp(-max(0.0, calibration_loss - 0.8))
        candidates.append(CandidatePrediction(
            name, calibration_probabilities, future_probabilities, calibration_score, calibration_loss, quality,
            calibrator.diagnostics_.method, calibrator.diagnostics_.selection_log_loss,
        ))
        future_probabilities_by_model[name] = future_probabilities

    analog_history = pd.concat([train, calibration_fit], ignore_index=True)
    training_registry = build_pattern_registry(analog_history)
    final_registry = build_pattern_registry(development)
    if pattern_adjuster is not None:
        final_registry = pattern_adjuster(final_registry)
    pattern_calibration = pattern_probabilities(policy_validation, training_registry)
    pattern_future = pattern_probabilities(future, final_registry)
    pattern_rows = _decision_rows(
        policy_validation["date"], pattern_calibration, policy_validation["daily_return"], matrix,
        "Pattern Registry", "latest", False,
    )
    pattern_score = float(pd.DataFrame(pattern_rows)["score"].mean())
    candidates.append(CandidatePrediction(
        "Pattern Registry", pattern_calibration, pattern_future, pattern_score,
        _safe_log_loss(policy_validation["target"].astype(int).to_numpy(), pattern_calibration),
        max(0.01, pattern_score - 0.25),
    ))
    future_probabilities_by_model["Pattern Registry"] = pattern_future

    analog_details: list[list[dict[str, object]]] = [[] for _ in range(len(future))]
    try:
        analog_calibration, analog_future, analog_details = _analog_forecast_bundle(
            analog_history, policy_validation, future, analog_columns,
        )
        analog_rows = _decision_rows(
            policy_validation["date"], analog_calibration, policy_validation["daily_return"], matrix,
            "Historical Analog", "latest", False,
        )
        analog_score = float(pd.DataFrame(analog_rows)["score"].mean())
        candidates.append(CandidatePrediction(
            "Historical Analog", analog_calibration, analog_future, analog_score,
            _safe_log_loss(policy_validation["target"].astype(int).to_numpy(), analog_calibration),
            max(0.01, analog_score - 0.25),
        ))
        future_probabilities_by_model["Historical Analog"] = analog_future
    except Exception:
        pass
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), final_registry
    preferred = set(preferred_models or [])
    selected = [candidate for candidate in candidates if not preferred or candidate.name in preferred]
    if not selected:
        selected = candidates
    weights, stacking_diagnostics = learn_simplex_weights(
        [candidate.calibration_probabilities for candidate in selected],
        policy_validation["target"].astype(int).to_numpy(),
    )
    ensemble_calibration = sum(
        weight * candidate.calibration_probabilities for weight, candidate in zip(weights, selected)
    )
    ensemble = sum(weight * future_probabilities_by_model[candidate.name] for weight, candidate in zip(weights, selected))
    effective_members = [
        (candidate, float(weight))
        for candidate, weight in zip(selected, weights)
        if weight >= 0.005
    ]
    if not effective_members:
        best_index = int(np.argmax(weights))
        effective_members = [(selected[best_index], float(weights[best_index]))]
    policy_probabilities = ensemble_calibration
    policy_returns = policy_validation["daily_return"].to_numpy(dtype=float)
    policy_dates = policy_validation["date"].to_numpy()
    if policy_history is not None and not policy_history.empty:
        history = policy_history.dropna(subset=["daily_return"]).tail(365)
        if not history.empty:
            historical_probabilities = history[["prob_down", "prob_sideway", "prob_up"]].to_numpy(dtype=float)
            policy_probabilities = np.concatenate([historical_probabilities, policy_probabilities], axis=0)
            policy_returns = np.concatenate([history["daily_return"].to_numpy(dtype=float), policy_returns])
            policy_dates = np.concatenate([history["date"].to_numpy(), policy_dates], axis=0)
    policy_mode, sideway_penalty, _ = select_decision_policy(
        policy_probabilities, policy_returns, matrix, policy_dates,
    )
    matching_pattern_sets = _matching_pattern_sets(future, final_registry)
    monthly_abstentions: defaultdict[str, int] = defaultdict(int)
    forecast_rows: list[dict[str, object]] = []
    finite_volatility = development["volatility_7"].dropna()
    median_volatility = float(finite_volatility.median()) if len(finite_volatility) else 0.03
    directions, expected_scores, margins, sideway_overrides = allocate_monthly_directions(
        future["date"], ensemble, matrix, policy_mode, sideway_penalty,
    )
    for row_index, (_, row) in enumerate(future.iterrows()):
        probability = ensemble[row_index]
        direction = str(directions[row_index])
        expected_score = float(expected_scores[row_index])
        margin = float(margins[row_index])
        entropy = float(-np.sum(np.clip(probability, 1e-9, 1) * np.log(np.clip(probability, 1e-9, 1))) / np.log(3))
        current_volatility = row.get("volatility_7", math.nan)
        volatility_ratio = float(current_volatility / median_volatility) if np.isfinite(current_volatility) and median_volatility > 0 else 1.0
        uncertainty_threshold = 0.965 - 0.015 * np.clip(volatility_ratio - 1, -1, 1)
        month_key = pd.Timestamp(row["date"]).strftime("%Y-%m")
        no_call = bool(entropy >= uncertainty_threshold and margin < 0.025 and monthly_abstentions[month_key] < 6)
        if no_call:
            monthly_abstentions[month_key] += 1
        forecast_rows.append({
            "date": pd.Timestamp(row["date"]),
            "lane": lane,
            "forecast": "no-call" if no_call else direction,
            "status": "no-call" if no_call else "pending",
            "confidence": float(probability[CLASS_TO_INDEX[direction]]),
            "prob_down": float(probability[0]),
            "prob_sideway": float(probability[1]),
            "prob_up": float(probability[2]),
            "expected_score": expected_score,
            "decision_margin": margin,
            "entropy": entropy,
            "top_pattern": matching_pattern_sets[row_index][0] if matching_pattern_sets[row_index] else None,
            # Only the next publishable session needs complete live-learning provenance.
            "matching_patterns": matching_pattern_sets[row_index] if row_index == 0 else [],
            "similar_cases": analog_details[row_index],
            "model_members": [candidate.name for candidate, _ in effective_members],
            "model_weights": [weight for _, weight in effective_members],
            "stacking_method": stacking_diagnostics["method"],
            "stacking_log_loss": stacking_diagnostics.get("loss"),
            "policy_mode": policy_mode,
            "sideway_penalty": sideway_penalty,
            "sideway_cap_override": bool(sideway_overrides[row_index]),
        })
    selected_weights = {candidate.name: float(weight) for candidate, weight in zip(selected, weights)}
    selection_rows: list[dict[str, object]] = []
    for candidate in candidates:
        next_probability = candidate.test_probabilities[0]
        next_direction, next_expected_score, next_margin = choose_direction(
            next_probability, matrix, policy_mode, sideway_penalty,
        )
        selection_rows.append({
            "model": candidate.name,
            "lane": lane,
            "calibration_score": candidate.calibration_score,
            "calibration_log_loss": candidate.calibration_log_loss,
            "calibration_method": candidate.calibration_method,
            "calibration_selection_log_loss": candidate.calibration_selection_log_loss,
            "weight": selected_weights.get(candidate.name, 0.0),
            "status": "active" if selected_weights.get(candidate.name, 0.0) >= 0.005 else "standby",
            "next_forecast": next_direction,
            "next_prob_down": float(next_probability[0]),
            "next_prob_sideway": float(next_probability[1]),
            "next_prob_up": float(next_probability[2]),
            "next_expected_score": next_expected_score,
            "next_decision_margin": next_margin,
        })
    return pd.DataFrame(forecast_rows), pd.DataFrame(selection_rows), final_registry


def reliability_bins(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    probabilities = predictions[["prob_down", "prob_sideway", "prob_up"]].to_numpy()
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    actual = predictions["daily_return"].map(direction_label).astype(int).to_numpy()
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for bucket_index in range(bins):
        mask = bucket == bucket_index
        if not mask.any():
            continue
        rows.append({
            "bucket": bucket_index,
            "confidence": float(confidence[mask].mean()),
            "observed_accuracy": float((predicted[mask] == actual[mask]).mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def equity_curve(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    output = forecasts[["date", "strategy_return"]].copy()
    output["equity"] = (1 + output["strategy_return"].fillna(0)).cumprod()
    output["benchmark"] = (1 + forecasts["daily_return"].fillna(0)).cumprod()
    output["drawdown"] = output["equity"] / output["equity"].cummax() - 1
    return output


def feature_heatmap(frame: pd.DataFrame, feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    selected = [
        "daily_return", "index_BTC", "index_me", "gap_index", "index_btc_slope_3",
        "index_me_slope_3", "composite", "finance", "volatility", "market_return_1",
        "volatility_7", "rsi14", "atr14_pct",
    ]
    selected = [column for column in selected if column in frame]
    realized = frame[frame["daily_return"].notna()][selected].tail(730)
    correlation = realized.corr(method="spearman")
    return correlation.reset_index().rename(columns={"index": "feature"})


def serialize_frame(
    frame: pd.DataFrame,
    date_columns: Iterable[str] = (
        "date", "train_start", "train_end", "calibration_fit_end", "policy_start", "calibration_end",
        "test_start", "test_end", "last_seen",
        "index_BTC_available_at", "index_me_available_at", "astro_available_at", "available_at",
    ),
) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    output = frame.copy()
    for column in date_columns:
        if column in output:
            parsed = pd.to_datetime(output[column], errors="coerce")
            output[column] = parsed.dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ" if "available_at" in column else "%Y-%m-%d",
            )
    records: list[dict[str, object]] = []
    for row in output.to_dict("records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if not np.isfinite(value) else float(value)
            elif isinstance(value, np.bool_):
                clean[key] = bool(value)
            elif isinstance(value, pd.Timestamp):
                clean[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, np.ndarray):
                clean[key] = value.tolist()
            elif pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records
