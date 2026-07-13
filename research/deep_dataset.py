from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntradaySequenceDataset:
    values: np.ndarray
    labels: np.ndarray
    returns: np.ndarray
    dates: np.ndarray
    last_bar_closed_at: np.ndarray
    feature_names: tuple[str, ...]
    lookback: int


@dataclass(frozen=True)
class ChronologicalFold:
    fold_id: str
    train_index: np.ndarray
    calibration_index: np.ndarray
    test_index: np.ndarray


def _prepare_bars(frame: pd.DataFrame, timeframe_hours: int) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Intraday bars are missing columns: {', '.join(missing)}")
    output = frame[required].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="raise").dt.tz_localize(None)
    for column in required[1:]:
        output[column] = pd.to_numeric(output[column], errors="raise")
    output = output.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    output["closed_at"] = output["timestamp"] + pd.Timedelta(hours=timeframe_hours)
    return output


def build_intraday_feature_table(one_hour: pd.DataFrame, four_hour: pd.DataFrame) -> pd.DataFrame:
    hourly = _prepare_bars(one_hour, 1)
    four = _prepare_bars(four_hour, 4)
    hourly["bucket"] = hourly["timestamp"].dt.floor("4h")
    hourly["log_return_1h"] = np.log(hourly["close"] / hourly["open"])
    hourly["range_1h"] = (hourly["high"] - hourly["low"]) / hourly["open"]
    hourly["signed_volume"] = np.sign(hourly["log_return_1h"]) * hourly["volume"]
    hourly_aggregates = hourly.groupby("bucket", as_index=False).agg(
        realized_vol_1h=("log_return_1h", lambda values: float(np.sqrt(np.square(values).sum()))),
        return_1h_sum=("log_return_1h", "sum"),
        range_1h_max=("range_1h", "max"),
        signed_volume_1h=("signed_volume", "sum"),
        volume_1h=("volume", "sum"),
        hourly_count=("timestamp", "size"),
    )
    hourly_aggregates["signed_volume_1h"] /= hourly_aggregates["volume_1h"].replace(0, np.nan)
    four["log_return_4h"] = np.log(four["close"] / four["open"])
    four["body_4h"] = (four["close"] - four["open"]) / four["open"]
    four["range_4h"] = (four["high"] - four["low"]) / four["open"]
    candle_range = (four["high"] - four["low"]).replace(0, np.nan)
    four["upper_wick_4h"] = (four["high"] - four[["open", "close"]].max(axis=1)) / candle_range
    four["lower_wick_4h"] = (four[["open", "close"]].min(axis=1) - four["low"]) / candle_range
    log_volume = np.log1p(four["volume"])
    rolling_mean = log_volume.rolling(126, min_periods=30).mean()
    rolling_std = log_volume.rolling(126, min_periods=30).std().replace(0, np.nan)
    four["volume_z_4h"] = (log_volume - rolling_mean) / rolling_std
    features = four.merge(hourly_aggregates, left_on="timestamp", right_on="bucket", how="left")
    features = features[features["hourly_count"].fillna(0).eq(4)].copy()
    feature_columns = [
        "log_return_4h", "body_4h", "range_4h", "upper_wick_4h", "lower_wick_4h",
        "volume_z_4h", "realized_vol_1h", "return_1h_sum", "range_1h_max", "signed_volume_1h",
    ]
    return features[["timestamp", "closed_at"] + feature_columns].sort_values("closed_at").reset_index(drop=True)


def _direction_label(daily_return: float) -> int:
    if daily_return < -0.01:
        return 0
    if daily_return > 0.01:
        return 2
    return 1


def sequence_for_target(
    intraday_features: pd.DataFrame,
    target_date: pd.Timestamp,
    context_row: pd.Series | None,
    context_columns: tuple[str, ...],
    lookback: int,
) -> tuple[np.ndarray, pd.Timestamp] | None:
    target = pd.Timestamp(target_date).tz_localize(None).normalize()
    eligible = intraday_features[intraday_features["closed_at"] <= target].tail(lookback)
    if len(eligible) != lookback or pd.Timestamp(eligible["closed_at"].iloc[-1]) != target:
        return None
    expected = pd.date_range(target - pd.Timedelta(hours=4 * (lookback - 1)), target, freq="4h")
    actual = pd.DatetimeIndex(eligible["closed_at"])
    if not actual.equals(expected):
        return None
    intraday_columns = [column for column in eligible if column not in {"timestamp", "closed_at"}]
    values = eligible[intraday_columns].to_numpy(dtype=np.float32)
    if context_columns:
        if context_row is None:
            context_values = np.full(len(context_columns), np.nan, dtype=np.float32)
        else:
            context_values = pd.to_numeric(context_row[list(context_columns)], errors="coerce").to_numpy(dtype=np.float32)
        repeated_context = np.repeat(context_values[None, :], lookback, axis=0)
        values = np.concatenate([values, repeated_context], axis=1)
    return values, pd.Timestamp(eligible["closed_at"].iloc[-1])


def build_intraday_sequence_dataset(
    one_hour: pd.DataFrame,
    four_hour: pd.DataFrame,
    daily_market: pd.DataFrame,
    context: pd.DataFrame | None = None,
    context_columns: tuple[str, ...] = (),
    lookback: int = 42,
    start_date: str | pd.Timestamp = "2021-01-01",
) -> IntradaySequenceDataset:
    intraday_features = build_intraday_feature_table(one_hour, four_hour)
    market = daily_market.copy()
    timestamp_column = "timestamp" if "timestamp" in market else "date"
    market["date"] = pd.to_datetime(market[timestamp_column], errors="raise").dt.tz_localize(None).dt.normalize()
    market["daily_return"] = pd.to_numeric(market["close"], errors="raise") / pd.to_numeric(
        market["open"], errors="raise",
    ) - 1
    market = market[market["date"] >= pd.Timestamp(start_date)].sort_values("date")
    context_lookup: dict[pd.Timestamp, pd.Series] = {}
    if context is not None and not context.empty:
        context_frame = context.copy()
        context_frame["date"] = pd.to_datetime(context_frame["date"]).dt.tz_localize(None).dt.normalize()
        context_lookup = {pd.Timestamp(row["date"]): row for _, row in context_frame.iterrows()}

    values: list[np.ndarray] = []
    labels: list[int] = []
    returns: list[float] = []
    dates: list[pd.Timestamp] = []
    last_closes: list[pd.Timestamp] = []
    for _, market_row in market.iterrows():
        date = pd.Timestamp(market_row["date"])
        sequence = sequence_for_target(
            intraday_features,
            date,
            context_lookup.get(date),
            context_columns,
            lookback,
        )
        if sequence is None:
            continue
        sequence_values, last_close = sequence
        daily_return = float(market_row["daily_return"])
        values.append(sequence_values)
        labels.append(_direction_label(daily_return))
        returns.append(daily_return)
        dates.append(date)
        last_closes.append(last_close)
    if not values:
        raise ValueError("No complete leakage-safe intraday sequences were produced")
    intraday_feature_names = tuple(
        column for column in intraday_features if column not in {"timestamp", "closed_at"}
    )
    return IntradaySequenceDataset(
        values=np.stack(values),
        labels=np.asarray(labels, dtype=int),
        returns=np.asarray(returns, dtype=float),
        dates=np.asarray(dates, dtype="datetime64[ns]"),
        last_bar_closed_at=np.asarray(last_closes, dtype="datetime64[ns]"),
        feature_names=intraday_feature_names + context_columns,
        lookback=lookback,
    )


def chronological_folds(
    dates: np.ndarray,
    n_folds: int = 2,
    calibration_days: int = 90,
    test_days: int = 180,
    purge_days: int = 1,
    minimum_train_days: int = 730,
) -> list[ChronologicalFold]:
    timestamps = pd.DatetimeIndex(dates)
    folds: list[ChronologicalFold] = []
    for offset in reversed(range(n_folds)):
        test_end_position = len(timestamps) - offset * test_days
        test_start_position = test_end_position - test_days
        calibration_start_position = test_start_position - calibration_days
        train_end_position = calibration_start_position - purge_days
        if train_end_position < minimum_train_days or test_start_position < 0:
            continue
        train_index = np.arange(0, train_end_position)
        calibration_index = np.arange(calibration_start_position, test_start_position)
        test_index = np.arange(test_start_position, test_end_position)
        folds.append(ChronologicalFold(
            fold_id=pd.Timestamp(timestamps[test_index[0]]).strftime("%Y-%m"),
            train_index=train_index,
            calibration_index=calibration_index,
            test_index=test_index,
        ))
    return folds
