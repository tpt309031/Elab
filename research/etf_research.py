"""Additive ETF diagnostics; never rewrites directional forecasts or their grades."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from research.etf_flow import ETF_FEATURES


def history_digest(observations: pd.DataFrame, target_end: str) -> str:
    end = pd.Timestamp(target_end)
    known = observations[(observations["available_at"] <= end.tz_localize("UTC") - pd.Timedelta(days=1))
                         & (observations["date"] <= end - pd.Timedelta(days=2))]
    return hashlib.sha256(known.to_json(orient="records", date_format="iso").encode()).hexdigest()


def etf_report(observations: pd.DataFrame, frame: pd.DataFrame, health: dict[str, Any],
               state: dict[str, Any], target_date: str,
               display_archive: dict[str, Any] | None = None) -> dict[str, Any]:
    from research.hybrid_core import serialize_frame

    cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    visible = observations[(observations["available_at"] <= cutoff.tz_localize("UTC"))
        & (observations["date"] < cutoff)].dropna(subset=["etf_net_flow"])
    visible = visible.sort_values(["date", "available_at"]).drop_duplicates("date", keep="last")
    visible = visible.copy()
    visible["cumulative_flow"] = visible["etf_net_flow"].cumsum()
    fields = [c for c in ETF_FEATURES if c in frame]
    timeline = frame.loc[frame["date"].between(pd.Timestamp("2024-01-01"), pd.Timestamp(target_date)), ["date", *fields]]
    correlation = []
    for feature in [c for c in frame if c.startswith("etf_")]:
        pairs = frame.loc[frame["date"] < cutoff, [feature, "daily_return"]].dropna()
        value = pairs[feature].corr(pairs["daily_return"]) if len(pairs) >= 30 and pairs[feature].nunique() > 1 else None
        correlation.append(dict(feature=feature, samples=len(pairs), correlation=float(value) if pd.notna(value) else None))
    published = next((r for r in state.get("forecasts", []) if r["lane"] == "Full Hybrid" and r["date"] == target_date), {})
    explanation = published.get("etf_evidence")
    last = timeline[timeline["date"] == pd.Timestamp(target_date)]
    return dict(health=health, history=serialize_frame(visible), timeline=serialize_frame(timeline),
        display_archive=display_archive,
        latest=serialize_frame(last), correlations=correlation, contribution=explanation,
        contribution_status="published coefficient snapshot" if explanation else "No ETF attribution in this immutable call",
        definitions={"unit": "USD; positive inflow, negative outflow; US spot Bitcoin ETF aggregate",
            "cutoff": "For target T: available_at <= T-1 00:00 UTC and observation date <= T-2",
            "rolling": "Last 3/7/14/30 reported trading observations, not calendar days; missing is not zero",
            "chart": "Daily and cumulative reflect the latest known vintage at the current cutoff. Markers use the first eligible forecast date, not the source date.",
            "correlation": "Descriptive lagged-feature/target-return correlation; not a causal effect or OOS uplift",
            "contribution": "Logistic reference class-logit terms, not calibrated ensemble probability contributions",
            "ranking": "Flow candidates use the same OOS and live Bayesian gates; no importance-based promotion bonus"})
