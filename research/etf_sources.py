"""Public ETF display archive, deliberately isolated from publication-time features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import requests

TFTC_URL = "https://www.tftc.io/bitcoin-etf-flows/data.json"
TFTC_METHOD = "https://www.tftc.io/bitcoin-etf-flows/methodology"
ATTRIBUTION = "The Commoner Bitcoin ETF Tracker by TFTC, tftc.io/bitcoin-etf-flows"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"


def parse_tftc(payload: dict[str, Any], observed_at: pd.Timestamp) -> list[dict[str, Any]]:
    """Validate USD observations; dateModified is NOT a publication timestamp."""
    if observed_at.tzinfo is None:
        raise ValueError("Capture time must be timezone-aware")
    if payload.get("units") != "USD" or not isinstance(payload.get("days"), list):
        raise ValueError("Unexpected TFTC schema or units")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload["days"]:
        date = item["date"]
        stamp = pd.Timestamp(date)
        if stamp.strftime("%Y-%m-%d") != date or stamp.tzinfo is not None:
            raise ValueError("ETF observation date must be YYYY-MM-DD")
        if stamp > observed_at.tz_convert("UTC").tz_localize(None).normalize() or date in seen:
            raise ValueError("Future or duplicate ETF observation")
        seen.add(date)
        value = item.get("netFlowUsd")
        if value is None:
            continue
        flow = float(value)
        aum = item.get("totalNetAssetsUsd")
        aum = float(aum) if aum is not None else None
        if not math.isfinite(flow) or (aum is not None and (not math.isfinite(aum) or aum <= 0)):
            raise ValueError("Invalid ETF net flow or AUM")
        # TFTC explicitly identifies zero/no-breakdown rows as market holidays.
        if flow == 0 and not item.get("perEtfUsd"):
            continue
        rows.append(dict(date=date, etf_net_flow=flow, etf_aum=aum, available_at=None))
    if not rows:
        raise ValueError("No usable ETF observations in public response")
    return sorted(rows, key=lambda row: row["date"])


def refresh_public_history(path: Path, payload: dict[str, Any] | None = None,
                           now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Cache validated history and immutable changed snapshots; never writes macro.csv."""
    if path.name == "macro.csv":
        raise ValueError("Public display history cannot replace the strict macro feed")
    if payload is None:
        response = requests.get(TFTC_URL, headers={"Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
    clock = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    rows = parse_tftc(payload, clock)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if old.get("history") and (rows[0]["date"] > old["history"][0]["date"]
                               or rows[-1]["date"] < old["history"][-1]["date"]
                               or not {r["date"] for r in old["history"]}.issubset({r["date"] for r in rows})):
        raise ValueError("Public ETF history regressed; keeping the last validated snapshot")
    captured = clock.tz_convert("UTC").isoformat()
    unchanged = digest == old.get("content_sha256")
    result = dict(source="TFTC", source_url=TFTC_URL, methodology_url=TFTC_METHOD,
        attribution=ATTRIBUTION, license=LICENSE, unit="USD", training_eligible=False,
        availability_mode="display-only: original publication times unknown",
        retrieved_at=old.get("retrieved_at", captured) if unchanged else captured,
        checked_at=captured, content_sha256=digest, history=rows,
        source_date_modified=payload.get("dateModified"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not unchanged:
        snapshots = path.parent / "etf_snapshots"
        snapshots.mkdir(exist_ok=True)
        snapshot = snapshots / f"{clock.tz_convert('UTC').strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:16]}.json"
        # Exclusive creation prevents a retry from replacing already captured evidence.
        with snapshot.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"), allow_nan=False)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return dict(status="unchanged" if unchanged else "updated", rows=len(rows),
                start=rows[0]["date"], end=rows[-1]["date"], training_eligible=False)


def load_public_history(path: Path, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Display diagnostics only. No caller may treat these rows as macro inputs."""
    empty: dict[str, Any] = dict(history=[], status="unavailable", training_eligible=False)
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["history"]
        expected = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if expected != data["content_sha256"] or data["training_eligible"] is not False or not rows:
            raise ValueError("Invalid display archive")
        clock = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
        if clock.tzinfo is None or pd.Timestamp(data["checked_at"]) > clock:
            raise ValueError("Invalid archive capture clock")
        data["status"] = "display-only" if (clock.tz_convert("UTC").tz_localize(None).normalize()
            - pd.Timestamp(rows[-1]["date"])).days <= 7 else "stale display archive"
        cumulative = 0.0
        for row in rows:
            cumulative += row["etf_net_flow"]
            row["cumulative_flow"] = cumulative
        return data
    except (ValueError, KeyError, TypeError, OSError):
        return {**empty, "status": "invalid display archive"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/etf_history.json"))
    args = parser.parse_args()
    print(json.dumps(refresh_public_history(args.output)))
