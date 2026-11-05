from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from research.data_sources import load_external_features
from research.etf_flow import load_macro, validate_macro
from research.etf_sources import load_public_history, parse_tftc, refresh_public_history


def sample() -> dict:
    return {"units": "USD", "dateModified": "2024-01-15T22:00:00Z", "days": [
        {"date": "2024-01-11", "netFlowUsd": 100, "totalNetAssetsUsd": 1000, "perEtfUsd": {"IBIT": 100}},
        {"date": "2024-01-12", "netFlowUsd": -20, "totalNetAssetsUsd": None, "perEtfUsd": {"IBIT": -20}},
        {"date": "2024-01-15", "netFlowUsd": 0, "perEtfUsd": None},
    ]}


def test_history_cannot_be_mistaken_for_publication_time_features(tmp_path) -> None:
    clock = pd.Timestamp("2024-01-16T12:00:00Z")
    path = tmp_path / "etf_history.json"
    result = refresh_public_history(path, sample(), clock)
    assert result["rows"] == 2  # No invented holiday observations.
    archive = load_public_history(path, clock)
    assert not archive["training_eligible"]
    assert archive["history"][-1]["cumulative_flow"] == 80
    assert all(row["available_at"] is None for row in archive["history"])
    with pytest.raises(ValueError, match="timezone"):
        validate_macro(pd.DataFrame(archive["history"]))
    macro, health = load_macro(tmp_path / "macro.csv", clock)
    assert macro.empty and not health["available"]
    features, _, _ = load_external_features(tmp_path)
    assert features.empty


def test_changed_snapshots_are_retained_but_repeat_fetch_is_not_a_new_vintage(tmp_path) -> None:
    clock = pd.Timestamp("2024-01-16T12:00:00Z")
    path = tmp_path / "etf_history.json"
    refresh_public_history(path, sample(), clock)
    first = next((tmp_path / "etf_snapshots").glob("*.json"))
    original = first.read_bytes()
    assert refresh_public_history(path, sample(), clock + pd.Timedelta(hours=1))["status"] == "unchanged"
    assert len(list(first.parent.glob("*.json"))) == 1
    assert load_public_history(path, clock + pd.Timedelta(hours=2))["retrieved_at"] == clock.isoformat()
    changed = sample()
    changed["days"][0]["netFlowUsd"] = 95
    refresh_public_history(path, changed, clock + pd.Timedelta(days=1))
    assert len(list(first.parent.glob("*.json"))) == 2
    assert first.read_bytes() == original


@pytest.mark.parametrize("kind", ["units", "duplicate", "future", "nonfinite", "negative_aum", "regressed"])
def test_bad_public_refresh_preserves_cache(tmp_path, kind) -> None:
    clock = pd.Timestamp("2024-01-16T12:00:00Z")
    path = tmp_path / "etf_history.json"
    refresh_public_history(path, sample(), clock)
    original = path.read_bytes()
    payload = copy.deepcopy(sample())
    if kind == "units":
        payload["units"] = "USD millions"
    elif kind == "duplicate":
        payload["days"].append(payload["days"][0])
    elif kind == "future":
        payload["days"][0]["date"] = "2025-01-01"
    elif kind == "nonfinite":
        payload["days"][0]["netFlowUsd"] = float("inf")
    elif kind == "negative_aum":
        payload["days"][0]["totalNetAssetsUsd"] = -1
    else:
        payload["days"] = payload["days"][1:]
    with pytest.raises(ValueError):
        refresh_public_history(path, payload, clock + pd.Timedelta(hours=1))
    assert path.read_bytes() == original


def test_archive_tampering_and_future_capture_fail_closed(tmp_path) -> None:
    clock = pd.Timestamp("2024-01-16T12:00:00Z")
    path = tmp_path / "etf_history.json"
    refresh_public_history(path, sample(), clock)
    assert load_public_history(path, clock - pd.Timedelta(seconds=1))["history"] == []
    assert load_public_history(path, clock + pd.Timedelta(days=20))["status"] == "stale display archive"
    data = json.loads(path.read_text())
    data["history"][0]["etf_net_flow"] = 9999
    path.write_text(json.dumps(data))
    assert load_public_history(path, clock)["history"] == []
    with pytest.raises(ValueError, match="strict macro"):
        refresh_public_history(tmp_path / "macro.csv", sample(), clock)
    with pytest.raises(ValueError, match="timezone"):
        parse_tftc(sample(), pd.Timestamp("2024-01-16"))
