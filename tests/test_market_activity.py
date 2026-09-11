from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.hybrid_core import allocate_monthly_directions
from research.large_moves import _digest, event_labels, event_metrics, grade_event_forecasts
from research.volume import COLUMNS, validate_volume, volume_features
from research.run_pipeline import _apply_trade_gate, _reserve_next_session
from research.learning import grade_learning_state
from threadpoolctl import threadpool_limits


def test_quota_prefix_does_not_depend_on_later_signals():
    dates = pd.date_range("2026-01-01", periods=62)
    p = np.random.default_rng(9).dirichlet([1, 4, 1], len(dates))
    complete = allocate_monthly_directions(dates, p, np.eye(3), allow_no_call=True)
    for stop in (1, 7, 9, 15, 30, 31, 48):
        prefix = allocate_monthly_directions(dates[:stop], p[:stop], np.eye(3), allow_no_call=True)
        for full, partial in zip(complete, prefix):
            np.testing.assert_equal(full[:stop], partial)


def test_volume_units_gaps_and_prior_baseline():
    data = pd.DataFrame([[day, 10, 1000, 20, 6, 600] for day in pd.date_range("2024-01-01", periods=25)], columns=COLUMNS)
    data.loc[24, ["base_volume", "quote_volume"]] = [100, 10000]
    validated = validate_volume(data, pd.Timestamp("2024-01-25"))
    features = volume_features(validated)
    assert features.iloc[-1]["spot_rvol20"] == 10
    assert np.isclose(features.iloc[0]["spot_taker_imbalance"], .2)
    gap = volume_features(validated.drop(index=15))
    assert pd.isna(gap.iloc[15]["spot_taker_imbalance"])
    assert pd.isna(gap.iloc[-1]["spot_rvol20"])
    data.loc[0, "taker_buy_base"] = 11
    with pytest.raises(ValueError, match="exceeds"):
        validate_volume(data, pd.Timestamp("2024-01-25"))


def candles():
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10), "open": 100.,
                         "high": 115., "low": 99., "close": [105, 101, 109, 102, 113, 100, 100, 100, 100, 100]})


def test_event_labels_require_full_forward_horizon_and_do_not_count_wicks_as_hits():
    market = candles()
    for horizon in (1, 3, 5):
        labels = event_labels(market, horizon)
        assert labels.iloc[0]["event"] == 1
        assert np.isclose(labels.iloc[0]["move"], market.iloc[horizon-1]["close"] / 100 - 1)
        if horizon > 1:
            assert labels.tail(horizon-1)["event"].isna().all()
    assert event_labels(market, 1).iloc[1]["event"] == 0
    assert event_labels(market, 3).iloc[5]["event"] == 0
    gap = event_labels(market.drop(index=1), 3)
    assert pd.isna(gap.iloc[0]["event"])


def test_event_ledger_waits_for_utc_maturity_and_is_idempotent():
    record = dict(id="x", date="2026-01-01", lane="Hybrid", horizon=3, probability=.7, alert=True,
                  alert_threshold=.5, model="Logistic", candidates={"Logistic": .7}, version="test",
                  information_cutoff="2025-12-31T00:00:00Z", issued_at="2025-12-31T03:20:00Z")
    record["digest"] = _digest(record)
    state = {"large_move_forecasts": [record]}
    assert grade_event_forecasts(state, candles(), "2026-01-04T02:59:59Z") == 0
    assert grade_event_forecasts(state, candles(), "2026-01-04T03:00:00Z") == 1
    assert state["large_move_forecasts"][0]["status"] == "hit"
    assert grade_event_forecasts(state, candles(), "2026-01-05T03:00:00Z") == 0
    changed = copy.deepcopy(state)
    changed["large_move_forecasts"][0]["probability"] = .9
    with pytest.raises(ValueError, match="integrity"):
        grade_event_forecasts(changed, candles(), "2026-01-05T03:00:00Z")


def test_event_metrics_count_false_alarms_and_misses_separately():
    rows = [dict(source="walk-forward", lane="Hybrid", horizon=1, date=f"2024-01-0{i+1}",
                 actual_event=y, probability=p, candidates={"Logistic": p}, base_probability=.5, alert_threshold=.5)
            for i, (y, p) in enumerate([(1, .9), (0, .8), (1, .2), (0, .1)])]
    selected = next(row for row in event_metrics(rows) if row["model"] == "Selected")
    assert selected["precision"] == .5
    assert selected["recall"] == .5
    assert selected["false_alarms"] == 1
    assert selected["missed"] == 1
    assert selected["precision_lcb"] < .5


def test_execution_cannot_reuse_next_session_signal_for_later_dates():
    forecasts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "forecast": ["up"]*3})
    metrics = pd.DataFrame([dict(model="Champion", rank=1, status="active", expectancy=.01, expectancy_lcb=.002)])
    selection = pd.DataFrame([dict(model="Champion", next_forecast="up")])
    result = _apply_trade_gate(forecasts, metrics, selection)
    assert result["trade_eligible"].tolist() == [True, False, False]


def test_daily_grading_waits_for_three_utc_and_scores_every_candidate():
    state = {"forecasts": [dict(target_date="2026-01-01", forecast="up",
                               model_predictions=[dict(model="A", forecast="up"), dict(model="B", forecast="sideway")]) ]}
    assert grade_learning_state(state, candles(), pd.Timestamp("2026-01-01"), "2026-01-02T02:59:59Z") == 0
    assert grade_learning_state(state, candles(), pd.Timestamp("2026-01-01"), "2026-01-02T03:00:00Z") == 1
    assert [p["status"] for p in state["forecasts"][0]["model_predictions"]] == ["correct", "wrong"]


def test_refit_cannot_release_an_immutable_no_call_reservation():
    locked = pd.DataFrame({"date": pd.date_range("2026-09-04", periods=4), "forecast": ["no-call"]*4})
    proposed = pd.DataFrame({"date": [pd.Timestamp("2026-09-07")], "forecast": ["up"]})
    reservation = _reserve_next_session(locked, proposed)
    assert (reservation["forecast"] == "no-call").sum() == 4
    np.testing.assert_array_equal(reservation["forecast"], locked["forecast"])


def test_magnitude_target_is_not_shadowed_by_astro_event(monkeypatch):
    import research.large_moves as module

    market = candles()
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=12), "event": False, "astro_event": 0.})
    seen = []
    def capture(target, columns, history, lane, horizon, cutoff, provenance):
        if len(history):
            assert history.iloc[0]["event"] == 1
            assert history.iloc[0]["astro_event"] == 0
            seen.append(horizon)
        return []
    monkeypatch.setattr(module, "_records", capture)
    result = module.build_large_move_research(frame, {"calendar": ["astro_event"], "full": ["astro_event"]}, market, {}, "2026-01-11T03:20:00Z")
    assert set(seen) == {1, 3, 5}
    assert result["official"] == []


def test_event_models_do_not_use_target_outcome_for_selection():
    from research.large_moves import _records

    dates = pd.date_range("2020-01-01", periods=800)
    rng = np.random.default_rng(42)
    history = pd.DataFrame({"date": dates, "feature": rng.normal(size=800),
                            "event": (rng.random(800) < .2).astype(float), "move": .05, "range": .07})
    target = history.tail(1).copy()
    with threadpool_limits(limits=2):
        first = _records(target, ["feature"], history, "Hybrid", 3, dates[-1], "walk-forward")[0]
        history.loc[history["date"] >= dates[-1], "event"] = 1 - target.iloc[0]["event"]
        target["event"] = 1 - target["event"]
        second = _records(target, ["feature"], history, "Hybrid", 3, dates[-1], "walk-forward")[0]
    assert first["candidates"] == second["candidates"]
    assert first["model"] == second["model"]
    assert pd.Timestamp(first["selection_end"]) + pd.Timedelta(days=2) <= dates[-1] - pd.Timedelta(days=2)


def test_checkpoint_roundtrip_and_changed_inputs_force_refit(tmp_path, monkeypatch):
    import research.hybrid_core as core
    from research.checkpoints import checkpointed_walk_forward

    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "target": [0, 1, 2], "daily_return": [-.01, 0, .01], "x": [1., 2., 3.]})
    calls = []
    def fit(*args):
        calls.append(True)
        return core.BacktestResult(*[frame.copy() for _ in range(7)])
    monkeypatch.setattr(core, "run_walk_forward", fit)
    arguments = (tmp_path, frame, ["x"], [], "Calendar", [], 0, False)
    first = checkpointed_walk_forward(*arguments)
    second = checkpointed_walk_forward(*arguments)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts, check_dtype=False)
    assert len(calls) == 1
    frame.loc[2, "x"] = 99
    checkpointed_walk_forward(*arguments)
    assert len(calls) == 2
