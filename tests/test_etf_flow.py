from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.etf_flow import ETF_FEATURES, load_macro, macro_features, refresh_etf_flow, validate_macro
from research.etf_research import history_digest
from research.hybrid_core import build_feature_frame
from research.learning import official_forecast_digest
from research.model_candidates import candidate_model_specs


def publications() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=45)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
        "available_at": (dates.tz_localize("UTC") + pd.Timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "etf_net_flow": np.arange(45, dtype=float) * 1e6, "etf_aum": 1e10})


def test_late_publication_and_revision_are_visible_only_after_cutoff() -> None:
    raw = pd.DataFrame([
        dict(date="2024-01-05", available_at="2024-01-06T14:00:00Z", etf_net_flow=100.),
        dict(date="2024-01-05", available_at="2024-01-09T20:00:00Z", etf_net_flow=-999.),
    ])
    dates = pd.date_range("2024-01-05", "2024-01-12")
    result = macro_features(raw, dates).set_index("date")
    assert result.loc[:"2024-01-07", "etf_net_flow"].isna().all()
    assert result.loc["2024-01-08":"2024-01-10", "etf_net_flow"].eq(100).all()
    assert result.loc["2024-01-11", "etf_net_flow"] == -999
    original = macro_features(raw.iloc[:1], dates).set_index("date")
    pd.testing.assert_frame_equal(result.loc[:"2024-01-10"], original.loc[:"2024-01-10"])


def test_exact_boundary_timezone_weekends_and_missing_days() -> None:
    raw = pd.DataFrame([
        dict(date="2024-01-03", available_at="2024-01-04T00:00:00Z", etf_net_flow=10.),
        dict(date="2024-01-04", available_at="2024-01-05T00:00:00Z", etf_net_flow=20.),
        dict(date="2024-01-05", available_at="2024-01-06T03:00:00+03:00", etf_net_flow=30.),
    ])
    frame = macro_features(raw, pd.date_range("2024-01-05", "2024-01-15")).set_index("date")
    assert frame.loc["2024-01-05", "etf_net_flow"] == 10
    assert frame.loc["2024-01-06", "etf_net_flow"] == 20
    assert frame.loc["2024-01-07":"2024-01-09", "etf_flow_cum_3"].eq(60).all()
    assert frame["etf_flow_cum_7"].isna().all()
    assert pd.isna(frame.loc["2024-01-13", "etf_net_flow"])


@pytest.mark.parametrize("timestamp", [None, "2024-01-02 12:00", "invalid", "2023-12-31T23:00:00Z"])
def test_unknown_or_impossible_publication_fails_closed(tmp_path, timestamp) -> None:
    path = tmp_path / "macro.csv"
    pd.DataFrame([dict(date="2024-01-01", available_at=timestamp, etf_net_flow=123)]).to_csv(path, index=False)
    frame, health = load_macro(path, now=pd.Timestamp("2024-01-10T12:00:00Z"))
    assert frame.empty
    assert not health["available"]
    assert health["status"] == "invalid publication-time feed"


def test_refresh_append_only_idempotent_and_failed_update_preserves_cache(tmp_path) -> None:
    path = tmp_path / "macro.csv"
    rows = publications().to_dict("records")
    now = pd.Timestamp("2024-04-01T00:00:00Z")
    refresh_etf_flow(path, records=rows, now=now)
    assert refresh_etf_flow(path, records=rows, now=now)["added"] == 0
    original = path.read_bytes()
    bad = {**rows[0], "etf_net_flow": 99999}
    with pytest.raises(ValueError, match="Conflicting"):
        refresh_etf_flow(path, records=[bad], now=now)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="future"):
        refresh_etf_flow(path, records=[{**bad, "available_at": "2025-01-01T00:00:00Z"}], now=now)
    assert path.read_bytes() == original


def test_sums_intensity_extremes_and_missing_aum() -> None:
    raw = publications()
    raw.loc[44, "etf_net_flow"] = 1e9
    target = pd.to_datetime(raw.iloc[-1]["date"]) + pd.Timedelta(days=2)
    row = macro_features(raw, pd.Series([target])).iloc[0]
    for window in (3, 7, 14, 30):
        assert row[f"etf_flow_cum_{window}"] == raw["etf_net_flow"].tail(window).sum()
    assert row["etf_flow_intensity"] == .1
    assert row["etf_extreme_inflow"] == 1
    assert row["etf_extreme_outflow"] == 0
    assert pd.isna(macro_features(raw.drop(columns="etf_aum"), pd.Series([target])).iloc[0]["etf_flow_intensity"])


def test_build_frame_groups_interactions_and_future_mutation() -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    market = pd.DataFrame(dict(timestamp=dates, open=100., high=110., low=90.,
        close=100 + np.sin(np.arange(100)), volume=1000.))
    indices = pd.DataFrame(dict(date=dates, index_BTC=np.arange(100), index_me=np.arange(100) / 2))
    astro = pd.DataFrame(dict(date=dates, finance=50., career=50., volatility=50., composite=50., event=False, regime="normal"))
    raw = publications()
    frame, groups = build_feature_frame(indices, market, astro, macro=raw)
    assert set(ETF_FEATURES).issubset(groups["external"])
    assert set(groups["etf"]).issubset(groups["full"])
    assert not set(groups["etf"]).intersection(groups["calendar"])
    assert "sequence__etf_net_flow__lag_0" in groups["sequence_full"]
    usable = frame["etf_net_flow"].notna()
    assert np.allclose(frame.loc[usable, "etf_x_index_BTC"], frame.loc[usable, "etf_net_flow"] * frame.loc[usable, "index_BTC"])
    changed = raw.copy()
    changed.loc[44, "etf_net_flow"] = -1e12
    altered, _ = build_feature_frame(indices, market, astro, macro=changed)
    past = frame["date"] <= pd.Timestamp(raw.loc[44, "date"]) + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(frame.loc[past], altered.loc[past])
    for spec in candidate_model_specs(groups["full"]):
        assert "etf_net_flow" in spec.columns
    assert len([s for s in candidate_model_specs(groups["full"]) if s.name.startswith("Flow ")]) == 2


def test_empty_feed_does_not_change_dimensions_or_fabricate_observations() -> None:
    empty = pd.DataFrame(columns=["date", "available_at", "etf_net_flow"])
    result = macro_features(empty, pd.date_range("2024-01-01", periods=3))
    assert result[list(ETF_FEATURES)].isna().all().all()
    assert not any(s.name.startswith("Flow ") for s in candidate_model_specs(["market_return_1"]))


def test_historical_revision_invalidates_cache_but_new_publications_do_not() -> None:
    raw = validate_macro(publications())
    old = history_digest(raw, "2024-01-20")
    future = raw.copy()
    future.loc[44, "etf_net_flow"] = 99
    assert history_digest(future, "2024-01-20") == old
    future.loc[0, "etf_net_flow"] = 99
    assert history_digest(future, "2024-01-20") != old


def test_additive_explanation_preserves_old_digests_and_protects_new_snapshots() -> None:
    old = {"contract_version": 2, "date": "2026-11-03", "forecast": "up"}
    digest = official_forecast_digest(old)
    assert official_forecast_digest({**old, "etf_evidence": None}) == digest
    new = {**old, "etf_evidence": {"model": "Logistic", "features": []}}
    assert official_forecast_digest(new) != digest
    changed = json.loads(json.dumps(new))
    changed["etf_evidence"]["model"] = "altered"
    assert official_forecast_digest(changed) != official_forecast_digest(new)


def test_refresh_preserves_subsecond_cutoff_precision(tmp_path) -> None:
    path = tmp_path / "macro.csv"
    record = dict(date="2024-01-05", available_at="2024-01-06T00:00:00.000000001Z", etf_net_flow=3.)
    refresh_etf_flow(path, records=[record], now=pd.Timestamp("2024-01-10T00:00:00Z"))
    observations, _ = load_macro(path, now=pd.Timestamp("2024-01-10T00:00:00Z"))
    result = macro_features(observations, pd.Series(pd.to_datetime(["2024-01-07", "2024-01-08"])))
    assert pd.isna(result.iloc[0]["etf_net_flow"])
    assert result.iloc[1]["etf_net_flow"] == 3


def test_specialists_fit_and_return_normalized_probabilities() -> None:
    features = pd.DataFrame({"market_return_1": np.sin(np.arange(120)) / 100,
        "etf_net_flow": np.cos(np.arange(120)) * 1e8, "etf_flow_cum_3": np.sin(np.arange(120) / 3) * 1e8})
    for spec in candidate_model_specs(list(features)):
        if not spec.name.startswith("Flow "):
            continue
        target = np.arange(120) % 3 if spec.target_column == "target" else np.sin(np.arange(120)) * .04
        fitted = spec.estimator.fit(features, target)
        probabilities = fitted.predict_proba(features.tail(5))
        assert probabilities.shape == (5, 3)
        assert np.allclose(probabilities.sum(axis=1), 1)


def test_flow_bayesian_evidence_is_independent_and_promotion_is_gated() -> None:
    from research.learning import apply_live_model_ranking, empty_learning_state
    name = "Flow Threshold Utility"
    metrics = pd.DataFrame([dict(model=name, observations=400, weighted_accuracy=.8,
        directional_accuracy=.8, expectancy=.01, rank_score=.9, coverage=1.,
        weighted_lcb=.7, expectancy_lcb=.005)])
    state = empty_learning_state()
    state["forecasts"] = [dict(lane="Full Hybrid", evaluated_at="2026-10-01T03:20:00Z",
        model_predictions=[dict(model="Logistic", status="correct", score=1.,
                                directional_hit=True, strategy_return=.01)]) for _ in range(20)]
    result = apply_live_model_ranking(metrics, state, "Full Hybrid").iloc[0]
    assert result["live_samples"] == 0
    assert not result["trade_eligible"]
    for row in state["forecasts"]:
        row["model_predictions"][0]["model"] = name
    result = apply_live_model_ranking(metrics, state, "Full Hybrid").iloc[0]
    assert result["live_samples"] == 20
    assert result["trade_eligible"]
    assert result["status"] == "active"


def test_missing_history_does_not_report_mock_uplift_or_touch_ledger(tmp_path) -> None:
    from research.evaluate_etf import evaluate
    ledger = tmp_path / "data/learning_state.json"
    ledger.parent.mkdir()
    ledger.write_text('{"forecasts": []}', encoding="utf-8")
    original = ledger.read_bytes()
    report = evaluate(tmp_path)
    assert report["status"] == "blocked_missing_publication_time_history"
    assert report["comparison"] == []
    assert ledger.read_bytes() == original
