from __future__ import annotations

import pytest
import pandas as pd

from research.learning import (
    append_official_forecast,
    apply_live_model_ranking,
    apply_live_pattern_ranking,
    empty_learning_state,
    grade_learning_state,
    load_learning_state,
    record_selection_snapshot,
    record_source_revision,
    serialize_learning_state,
)


def test_source_revision_history_is_append_only_and_content_addressed() -> None:
    state = empty_learning_state()
    first = [{"source": "private-index-btc", "sha256": "a", "rows": 10, "last_date": "2026-01-01"}]
    second = [{"source": "private-index-btc", "sha256": "b", "rows": 11, "last_date": "2026-01-02"}]

    assert record_source_revision(state, first, "2026-01-02T00:00:00Z")
    assert not record_source_revision(state, first, "2026-01-02T01:00:00Z")
    assert record_source_revision(state, second, "2026-01-03T00:00:00Z")
    assert len(state["source_revisions"]) == 2
    assert state["source_revisions"][-1]["changed_sources"] == ["private-index-btc"]


def test_official_forecast_is_idempotent_and_immutable() -> None:
    state = empty_learning_state()
    forecast = {
        "date": "2026-07-13",
        "forecast": "up",
        "prob_down": 0.2,
        "prob_sideway": 0.3,
        "prob_up": 0.5,
        "top_pattern": {"id": "shock", "name": "Shock", "direction": "up", "rank": 1},
    }
    selection = [{"model": "Logistic", "next_forecast": "up", "status": "active"}]
    assert append_official_forecast(state, forecast, "Calendar", "2026-07-12T03:20:00Z", "2026-07-11", selection)
    changed = {**forecast, "forecast": "down"}
    assert not append_official_forecast(state, changed, "Calendar", "2026-07-12T04:00:00Z", "2026-07-11", selection)
    assert len(state["forecasts"]) == 1
    assert state["forecasts"][0]["forecast"] == "up"
    assert state["forecasts"][0]["issued_at"] == "2026-07-12T03:20:00Z"
    assert state["forecasts"][0]["contract_version"] == 2


def test_official_forecast_rejects_a_target_that_has_already_opened() -> None:
    state = empty_learning_state()
    with pytest.raises(ValueError, match="issued before target opens"):
        append_official_forecast(
            state,
            {"date": "2026-07-12", "forecast": "up"},
            "Calendar",
            "2026-07-12T03:20:00Z",
            "2026-07-11",
        )


def test_only_closed_sessions_are_graded_with_model_and_pattern_provenance() -> None:
    state = empty_learning_state()
    base = {
        "forecast": "up",
        "prob_down": 0.1,
        "prob_sideway": 0.2,
        "prob_up": 0.7,
        "top_pattern": {"id": "breakout", "name": "Breakout", "direction": "up", "rank": 2},
        "matching_patterns": [
            {"id": "breakout", "name": "Breakout", "direction": "up", "rank": 2},
            {"id": "shock", "name": "Shock", "direction": "down", "rank": 4},
        ],
    }
    append_official_forecast(
        state,
        {**base, "date": "2026-07-10"},
        "Full Hybrid",
        "2026-07-09T03:20:00Z",
        "2026-07-08",
        [{"model": "Logistic", "next_forecast": "down", "status": "active"}],
    )
    append_official_forecast(
        state,
        {**base, "date": "2026-07-12"},
        "Full Hybrid",
        "2026-07-11T03:20:00Z",
        "2026-07-10",
        [],
    )
    market = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-10", "2026-07-11"]),
        "open": [100.0, 104.0],
        "high": [105.0, 105.0],
        "low": [99.0, 101.0],
        "close": [104.0, 102.0],
        "volume": [1.0, 1.0],
    })
    count = grade_learning_state(state, market, pd.Timestamp("2026-07-11"), "2026-07-12T03:20:00Z")
    assert count == 1
    evaluated, future = state["forecasts"]
    assert evaluated["status"] == "correct"
    assert evaluated["score"] == 1.0
    assert evaluated["model_predictions"][1]["status"] == "wrong"
    assert evaluated["pattern_evaluation"]["status"] == "correct"
    assert [item["status"] for item in evaluated["pattern_predictions"]] == ["correct", "wrong"]
    assert future["status"] == "pending"
    assert future["evaluated_at"] is None


def test_live_results_promote_better_model_without_erasing_oos_prior() -> None:
    metrics = pd.DataFrame([
        _model_metric("Model A"),
        _model_metric("Model B"),
    ])
    state = empty_learning_state()
    state["forecasts"] = [
        {
            "lane": "Calendar",
            "evaluated_at": f"2026-07-{day:02d}T03:20:00Z",
            "model_predictions": [
                {"model": "Model A", "status": "correct", "score": 1.0, "directional_hit": True, "strategy_return": 0.01},
                {"model": "Model B", "status": "wrong", "score": 0.0, "directional_hit": False, "strategy_return": -0.01},
            ],
        }
        for day in range(1, 11)
    ]
    ranked = apply_live_model_ranking(metrics, state, "Calendar", active_limit=1)
    assert ranked.iloc[0]["model"] == "Model A"
    assert ranked.iloc[0]["status"] == "active"
    assert ranked.iloc[0]["live_samples"] == 10
    assert ranked.iloc[0]["adjusted_weighted_accuracy"] > ranked.iloc[1]["adjusted_weighted_accuracy"]


def test_model_with_non_positive_expectancy_lower_bound_cannot_trade() -> None:
    metric = _model_metric("Optimistic model")
    metric["expectancy_lcb"] = -0.0001
    ranked = apply_live_model_ranking(
        pd.DataFrame([metric]),
        empty_learning_state(),
        "Calendar",
    )

    assert ranked.iloc[0]["status"] == "standby"
    assert not bool(ranked.iloc[0]["trade_eligible"])
    assert "expectancy lower bound" in ranked.iloc[0]["replacement_reason"]


def test_live_pattern_results_control_active_registry_and_snapshot_is_daily() -> None:
    registry = pd.DataFrame([
        _pattern_metric("pattern-a"),
        _pattern_metric("pattern-b"),
    ])
    state = empty_learning_state()
    state["forecasts"] = [
        {
            "lane": "Full Hybrid",
            "evaluated_at": f"2026-07-{day:02d}T03:20:00Z",
            "pattern_evaluation": {
                "pattern_id": pattern,
                "direction": "up",
                "status": status,
                "score": score,
                "strategy_return": strategy_return,
            },
        }
        for day in range(1, 9)
        for pattern, status, score, strategy_return in [
            ("pattern-a", "correct", 1.0, 0.01),
            ("pattern-b", "wrong", 0.0, -0.01),
        ]
    ]
    ranked = apply_live_pattern_ranking(registry, state, "Full Hybrid", active_limit=1)
    assert ranked.iloc[0]["pattern_id"] == "pattern-a"
    assert ranked.iloc[0]["status"] == "active"
    assert ranked.iloc[0]["selection_change"] == "promoted"
    model_metrics = apply_live_model_ranking(pd.DataFrame([_model_metric("Model A")]), state, "Full Hybrid")
    assert record_selection_snapshot(
        state,
        "2026-07-11",
        "2026-07-12T03:20:00Z",
        {"Full Hybrid": model_metrics},
        {"Full Hybrid": ranked},
    )
    assert not record_selection_snapshot(
        state,
        "2026-07-11",
        "2026-07-12T04:20:00Z",
        {"Full Hybrid": model_metrics},
        {"Full Hybrid": ranked},
    )
    assert len(state["selection_history"]) == 1
    assert state["selection_history"][0]["lanes"]["Full Hybrid"]["active_patterns"] == ["pattern-a|up"]


def test_published_digest_detects_mutation_and_corrupt_state_fails_closed(tmp_path) -> None:
    state = empty_learning_state()
    forecast = {
        "date": "2026-07-13",
        "forecast": "up",
        "prob_down": 0.2,
        "prob_sideway": 0.3,
        "prob_up": 0.5,
    }
    append_official_forecast(state, forecast, "Calendar", "2026-07-12T03:20:00Z", "2026-07-11")
    path = tmp_path / "learning.json"
    path.write_text(serialize_learning_state(state), encoding="utf-8")
    assert load_learning_state(path)["forecasts"][0]["forecast"] == "up"

    state["forecasts"][0]["forecast"] = "down"
    path.write_text(serialize_learning_state(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        load_learning_state(path)

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot load"):
        load_learning_state(path)


def _model_metric(model: str) -> dict[str, float | int | str]:
    return {
        "model": model,
        "observations": 500,
        "calls": 450,
        "coverage": 0.9,
        "no_calls": 50,
        "exact_accuracy": 0.4,
        "weighted_accuracy": 0.5,
        "directional_accuracy": 0.55,
        "balanced_accuracy": 0.5,
        "brier": 0.6,
        "sharpe": 0.5,
        "profit_factor": 1.1,
        "max_drawdown": -0.2,
        "expectancy": 0.001,
        "expectancy_lcb": 0.0001,
        "net_return": 0.1,
        "rank_score": 0.5,
        "rank": 1,
        "status": "active",
    }


def _pattern_metric(pattern_id: str) -> dict[str, float | int | str | bool | list[str]]:
    return {
        "pattern_id": pattern_id,
        "pattern": pattern_id,
        "expression": "index_BTC > 50",
        "direction": "up",
        "occurrences": 20,
        "weighted_accuracy": 0.5,
        "exact_accuracy": 0.4,
        "expectancy": 0.001,
        "last_seen": "2026-07-01",
        "examples": [],
        "eligible": True,
        "rank_score": 1.5,
        "rank": 1,
        "status": "active",
    }
