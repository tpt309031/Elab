"""Content-addressed local checkpoints; JSON only, never executable pickles."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd


def checkpointed_walk_forward(root: Path, frame, columns, analog_columns, lane, sequence_columns,
                              sequence_count, include_deep):
    from research.hybrid_core import BacktestResult, RESEARCH_PROTOCOL, run_walk_forward

    digest = hashlib.sha256(f"{RESEARCH_PROTOCOL}|{lane}|{include_deep}".encode())
    for module in ("hybrid_core.py", "model_candidates.py", "evaluation.py", "deep_models.py"):
        path = root / "research" / module
        if path.exists():
            digest.update(path.read_bytes())
    inputs = list(dict.fromkeys(["date", "target", "daily_return", *columns, *analog_columns,
                                 *(sequence_columns if include_deep else [])]))
    digest.update(json.dumps(inputs).encode())
    observed = frame.loc[frame["target"].notna(), inputs]
    digest.update(pd.util.hash_pandas_object(observed, index=False).to_numpy().tobytes())
    key = digest.hexdigest()
    path = root / "data" / ".research-cache" / f"{lane.replace(' ', '-')}-{key}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["key"] != key:
                raise ValueError("Checkpoint key mismatch")
            result = BacktestResult(**{item.name: pd.read_json(io.StringIO(payload["frames"][item.name]), orient="table") for item in fields(BacktestResult)})
            print(f"[{lane}] restored completed input-matched checkpoint", flush=True)
            return result
        except (ValueError, KeyError, OSError):
            print(f"[{lane}] invalid checkpoint; recalculating", flush=True)
    result = run_walk_forward(frame, columns, analog_columns, lane, sequence_columns, sequence_count, include_deep)
    payload = {"key": key, "frames": {item.name: getattr(result, item.name).to_json(orient="table", date_format="iso", double_precision=15) for item in fields(BacktestResult)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    print(f"[{lane}] checkpoint saved", flush=True)
    return result
