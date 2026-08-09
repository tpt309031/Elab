from __future__ import annotations

import pandas as pd

from research.run_pipeline import _apply_trade_gate


def _forecast(direction: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-08-10"),
        "forecast": direction,
        "prob_down": 0.2,
        "prob_sideway": 0.2,
        "prob_up": 0.6,
    }])


def _metrics(expectancy_lcb: float = 0.004) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": "Champion",
        "status": "active",
        "rank": 1,
        "expectancy": 0.008,
        "expectancy_lcb": expectancy_lcb,
    }])


def test_trade_gate_requires_champion_direction_agreement() -> None:
    selection = pd.DataFrame([{"model": "Champion", "next_forecast": "down"}])

    result = _apply_trade_gate(_forecast("up"), _metrics(), selection).iloc[0]

    assert result["trade_action"] == "flat"
    assert not bool(result["trade_eligible"])
    assert pd.isna(result["expectancy_lcb"])
    assert "does not confirm" in result["trade_gate_reason"]


def test_trade_gate_executes_only_positive_aligned_direction() -> None:
    selection = pd.DataFrame([{"model": "Champion", "next_forecast": "up"}])

    result = _apply_trade_gate(_forecast("up"), _metrics(), selection).iloc[0]

    assert result["trade_action"] == "up"
    assert bool(result["trade_eligible"])
    assert result["expectancy_lcb"] == 0.004
    assert result["execution_model"] == "Champion"
    assert result["execution_model_forecast"] == "up"

    rejected = _apply_trade_gate(_forecast("up"), _metrics(-0.001), selection).iloc[0]
    assert rejected["trade_action"] == "flat"
    assert not bool(rejected["trade_eligible"])
    assert pd.isna(rejected["expectancy_lcb"])
