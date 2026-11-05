"""Publication-time vintages for US spot Bitcoin ETF net flow (USD)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

MACRO_VALUES = ("etf_net_flow", "etf_aum", "dxy", "us10y")
ETF_FEATURES = (
    "etf_net_flow", "etf_net_flow_change_1", "etf_net_flow_z30",
    "etf_flow_cum_3", "etf_flow_cum_7", "etf_flow_cum_14", "etf_flow_cum_30",
    "etf_flow_intensity", "etf_extreme_inflow", "etf_extreme_outflow", "etf_age_days",
)
MAX_AGE_DAYS = 7
MIN_FLOW_TRAINING_ROWS = 90


def validate_macro(source: pd.DataFrame) -> pd.DataFrame:
    """Retain distinct revisions; never infer publication time from trading date."""
    if not {"date", "available_at", "etf_net_flow"}.issubset(source):
        raise ValueError("macro.csv requires date, available_at, etf_net_flow (USD)")
    output = source[[c for c in ("date", "available_at", *MACRO_VALUES) if c in source]].copy()
    if output.empty:
        output["date"] = pd.to_datetime(output["date"])
        output["available_at"] = pd.to_datetime(output["available_at"], utc=True)
        return output
    if not output["date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("date must be an observation date in YYYY-MM-DD format")
    if not output["available_at"].astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("available_at requires an explicit timezone; missing timestamps are not imputed")
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    output["available_at"] = pd.to_datetime(output["available_at"], utc=True, errors="raise", format="mixed")
    if (output["available_at"].dt.tz_localize(None) < output["date"]).any():
        raise ValueError("publication cannot precede the observation date")
    for column in MACRO_VALUES:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="raise")
            if np.isinf(output[column].to_numpy(dtype=float)).any():
                raise ValueError(f"Non-finite {column}")
    if "etf_aum" in output and (output["etf_aum"].dropna() <= 0).any():
        raise ValueError("ETF AUM must be positive USD")
    output = output.drop_duplicates()
    if output.duplicated(["date", "available_at"]).any():
        raise ValueError("Conflicting values for the same observation/publication; append a later revision")
    return output.sort_values(["available_at", "date"]).reset_index(drop=True)


def load_macro(path: str | Path, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    now = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    health: dict[str, Any] = dict(source="macro", file="macro.csv", available=False,
        status="missing publication-time feed", rows=0, usable_rows=0, unit="USD",
        features=list(ETF_FEATURES), availability_mode="explicit-publication-required")
    empty = pd.DataFrame(columns=["date", "available_at", "etf_net_flow"])
    if not path.exists():
        return validate_macro(empty), health
    try:
        observations = validate_macro(pd.read_csv(path))
    except (ValueError, pd.errors.ParserError) as exc:
        health.update(status="invalid publication-time feed", error=str(exc))
        return validate_macro(empty), health
    known = observations[observations["available_at"] <= now]
    usable = known.dropna(subset=["etf_net_flow"])
    latest = usable["date"].max()
    health.update(rows=len(observations), usable_rows=len(usable), available=bool(len(usable)),
        status=("healthy" if (now.tz_convert("UTC").tz_localize(None).normalize() - latest).days <= MAX_AGE_DAYS
                else "stale") if len(usable) else "awaiting verified ETF publications",
        future_publications_excluded=int((observations["available_at"] > now).sum()),
        latest_observation=latest.strftime("%Y-%m-%d") if len(usable) else None,
        latest_publication=usable["available_at"].max().isoformat() if len(usable) else None,
        availability_imputed_rows=0)
    return known.reset_index(drop=True), health


def macro_features(observations: pd.DataFrame, target_dates: pd.Series,
                   lead_days: int = 2) -> pd.DataFrame:
    """Rebuild each vintage at T-(lead-1) 00:00 UTC, using observations <=T-lead.

    Sums and z-scores use distinct reported sessions, not forward-filled BTC days.
    Missing reports are not zero. Revisions can affect only subsequent cutoffs.
    """
    if lead_days < 2:
        raise ValueError("ETF features require at least a two-session forecast lead")
    observations = validate_macro(observations)
    columns = [*ETF_FEATURES, *[c for c in ("dxy", "us10y") if c in observations]]
    if observations.empty:
        output = pd.DataFrame({"date": pd.to_datetime(target_dates).sort_values().drop_duplicates()})
        return output.assign(**dict.fromkeys(columns, np.nan))
    records: list[dict[str, Any]] = []
    for date in pd.to_datetime(target_dates).sort_values().drop_duplicates():
        cutoff = date.tz_localize("UTC") - pd.Timedelta(days=lead_days - 1)
        row: dict[str, Any] = {"date": date, **dict.fromkeys(columns, np.nan)}
        vintage = observations[(observations["available_at"] <= cutoff)
            & (observations["date"] <= date - pd.Timedelta(days=lead_days))]
        for field in ("dxy", "us10y"):
            if field in vintage:
                values = vintage.dropna(subset=[field]).sort_values(["date", "available_at"])
                if len(values) and (date - values.iloc[-1]["date"]).days <= MAX_AGE_DAYS:
                    row[field] = float(values.iloc[-1][field])
        vintage = vintage.dropna(subset=["etf_net_flow"]).sort_values(["date", "available_at"])
        vintage = vintage.drop_duplicates("date", keep="last")
        if len(vintage):
            latest = vintage.iloc[-1]
            age = (date - latest["date"]).days
            if age <= MAX_AGE_DAYS:
                values = vintage["etf_net_flow"].astype(float)
                row.update(etf_net_flow=float(values.iloc[-1]), etf_age_days=float(age))
                if len(values) >= 2:
                    row["etf_net_flow_change_1"] = float(values.iloc[-1] - values.iloc[-2])
                for window in (3, 7, 14, 30):
                    if len(values) >= window:
                        row[f"etf_flow_cum_{window}"] = float(values.tail(window).sum())
                trailing = values.tail(30)
                if len(trailing) >= 10 and trailing.std() > 0:
                    z = float((values.iloc[-1] - trailing.mean()) / trailing.std())
                    row.update(etf_net_flow_z30=z, etf_extreme_inflow=float(z > 2),
                               etf_extreme_outflow=float(z < -2))
                aum = latest.get("etf_aum", np.nan)
                if pd.notna(aum) and aum > 0:
                    row["etf_flow_intensity"] = float(values.iloc[-1] / aum)
        records.append(row)
    return pd.DataFrame(records, columns=["date", *columns])


def refresh_etf_flow(path: str | Path, records: list[dict[str, Any]] | None = None,
                     endpoint: str | None = None, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Import trusted publication-stamped JSON; records is the offline/test adapter.

    Endpoint contract: a JSON list, or {records: [...]}, in USD. No invented lag,
    scraping-time backdating, credential logging or replacement of earlier vintages.
    """
    path = Path(path)
    if records is None:
        endpoint = endpoint or os.environ.get("ELAB_ETF_FLOW_URL")
        if not endpoint:
            return {"status": "not-configured", "added": 0}
        if not endpoint.startswith("https://"):
            raise ValueError("ETF feed endpoint must use HTTPS")
        headers = {"Accept": "application/json"}
        if os.environ.get("ELAB_ETF_FLOW_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['ELAB_ETF_FLOW_TOKEN']}"
        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("ETF endpoint must supply a list of publication-stamped records")
    if not records:
        return {"status": "no-new-publications", "added": 0}
    incoming = validate_macro(pd.DataFrame(records))
    clock = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if clock.tzinfo is None or (incoming["available_at"] > clock).any():
        raise ValueError("Cannot ingest future publications or use a naive current time")
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "available_at", "etf_net_flow"])
    validate_macro(old)
    # Re-validate the union before any write, including conflicting same-time revisions.
    merged = validate_macro(pd.concat([old, pd.DataFrame(records)], ignore_index=True))
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged["available_at"] = merged["available_at"].map(lambda timestamp: timestamp.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(path)
    return {"status": "updated", "added": len(merged) - len(old), "rows": len(merged)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Trusted JSON records with actual publication timestamps")
    parser.add_argument("--output", type=Path, default=Path("data/external/macro.csv"))
    args = parser.parse_args()
    imported = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    print(json.dumps(refresh_etf_flow(args.output, records=imported), indent=2))
