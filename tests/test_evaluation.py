from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.evaluation import (
    ProbabilityCalibrator,
    expected_calibration_error,
    moving_block_confidence_intervals,
    page_hinkley_alarm,
)


def test_page_hinkley_only_suspends_on_recent_material_deterioration() -> None:
    stable = page_hinkley_alarm([0.3] * 300)
    degraded = page_hinkley_alarm([0.25] * 260 + [0.85] * 60)

    assert stable["alarm"] is False
    assert stable["action"] == "monitor"
    assert degraded["alarm"] is True
    assert degraded["action"] == "suspend-execution"
    assert degraded["recent_loss"] > degraded["baseline_loss"]
from research.learning import (
    append_official_forecast,
    apply_live_model_ranking,
    empty_learning_state,
    grade_event_state,
    grade_learning_state,
)


def test_calibrator_uses_chronological_selection_and_returns_simplex() -> None:
    labels = np.array(([0, 1, 2] * 40), dtype=int)
    probabilities = np.full((len(labels), 3), 0.05)
    probabilities[np.arange(len(labels)), labels] = 0.90
    probabilities = probabilities ** 0.45
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    calibrator = ProbabilityCalibrator().fit(probabilities, labels)
    transformed = calibrator.transform(probabilities[-12:])

    assert calibrator.diagnostics_.fit_rows > calibrator.diagnostics_.validation_rows
    assert calibrator.diagnostics_.method in {"identity", "sigmoid", "temperature", "isotonic"}
    assert np.allclose(transformed.sum(axis=1), 1.0)
    assert expected_calibration_error(transformed, labels[-12:]) <= 1.0


def test_moving_block_intervals_contain_point_estimates() -> None:
    count = 100
    predictions = pd.DataFrame({
        "forecast": ["up"] * count,
        "status": ["correct"] * 60 + ["wrong"] * 40,
        "score": [1.0] * 60 + [0.0] * 40,
        "directional_hit": [True] * 70 + [False] * 30,
        "strategy_return": np.linspace(-0.02, 0.03, count),
    })
    intervals = moving_block_confidence_intervals(predictions, samples=80, block_size=10)
    assert intervals["weighted_lcb"] <= 0.60 <= intervals["weighted_ucb"]
    assert intervals["directional_lcb"] <= 0.70 <= intervals["directional_ucb"]


def test_live_challenger_needs_twenty_grades_before_monthly_promotion() -> None:
    metrics = pd.DataFrame([
        _model_metric("Challenger", 0.60),
        _model_metric("Champion", 0.50),
    ])
    state = empty_learning_state()
    state["selection_history"] = [{
        "as_of_closed": "2026-06-30",
        "lanes": {"Calendar": {"active_models": ["Champion"], "active_patterns": []}},
    }]
    state["forecasts"] = _live_model_rows(10)
    ranked = apply_live_model_ranking(
        metrics, state, "Calendar", active_limit=1, as_of_closed="2026-07-01",
    )
    assert ranked.loc[ranked["model"] == "Champion", "status"].iloc[0] == "active"
    assert "20 live grades" in ranked.loc[ranked["model"] == "Challenger", "replacement_reason"].iloc[0]

    state["forecasts"] = _live_model_rows(22)
    ranked = apply_live_model_ranking(
        metrics, state, "Calendar", active_limit=1, as_of_closed="2026-07-01",
    )
    assert ranked.loc[ranked["model"] == "Challenger", "status"].iloc[0] == "active"


def test_event_grading_is_separate_from_daily_grade() -> None:
    state = empty_learning_state()
    append_official_forecast(
        state,
        {
            "date": "2026-07-05",
            "forecast": "up",
            "prob_down": 0.2,
            "prob_sideway": 0.3,
            "prob_up": 0.5,
            "matching_patterns": [{"id": "lead", "name": "Lead", "direction": "up", "rank": 1}],
        },
        "Full Hybrid",
        "2026-07-04T03:20:00Z",
        "2026-07-03",
        [{"model": "Challenger", "next_forecast": "up", "status": "standby"}],
    )
    dates = pd.date_range("2026-07-01", periods=14, freq="D")
    close = np.full(len(dates), 100.0)
    close[5] = 104.5
    market = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": np.maximum(close, 100.0) + 1,
        "low": np.minimum(close, 100.0) - 1,
        "close": close,
        "volume": 1.0,
    })
    grade_learning_state(state, market, pd.Timestamp("2026-07-14"), "2026-07-14T03:20:00Z")
    daily_before = (state["forecasts"][0]["status"], state["forecasts"][0]["score"])
    evaluated = grade_event_state(state, market, pd.Timestamp("2026-07-14"), "2026-07-14T03:20:00Z")

    assert daily_before == ("wrong", 0.0)
    assert evaluated >= 1
    assert any(row["status"] == "matched" and row["lead_lag_days"] == 1 for row in state["event_evaluations"])
    assert (state["forecasts"][0]["status"], state["forecasts"][0]["score"]) == daily_before


def _model_metric(model: str, rank_score: float) -> dict[str, float | int | str]:
    return {
        "model": model,
        "observations": 500,
        "calls": 500,
        "coverage": 1.0,
        "no_calls": 0,
        "exact_accuracy": 0.4,
        "weighted_accuracy": 0.5,
        "weighted_lcb": 0.42,
        "directional_accuracy": 0.55,
        "balanced_accuracy": 0.5,
        "brier": 0.6,
        "sharpe": 0.2,
        "profit_factor": 1.05,
        "max_drawdown": -0.2,
        "expectancy": 0.001,
        "expectancy_lcb": 0.0001,
        "net_return": 0.0,
        "rank_score": rank_score,
        "rank": 1,
        "status": "standby",
    }


def _live_model_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "lane": "Calendar",
            "evaluated_at": f"2026-06-{(index % 28) + 1:02d}T03:20:00Z",
            "model_predictions": [
                {
                    "model": "Challenger",
                    "status": "correct",
                    "score": 1.0,
                    "directional_hit": True,
                    "strategy_return": 0.01,
                },
                {
                    "model": "Champion",
                    "status": "wrong",
                    "score": 0.0,
                    "directional_hit": False,
                    "strategy_return": -0.01,
                },
            ],
        }
        for index in range(count)
    ]
