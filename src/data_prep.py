# src/data_prep.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_sessions(path: str | Path, n: int = 50_000) -> List[Dict[str, Any]]:
    """
    Load first n sessions from OTTO train.jsonl (sampled reading to avoid OOM).
    Returns a list of dicts: {"session": int, "events": [{aid, ts, type}, ...]}
    """
    path = Path(path)
    sessions: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sessions.append(json.loads(line))
    return sessions


def flatten_events(sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten nested session/events into an event-level table:
    columns: session, aid, ts, type
    """
    rows = []
    for s in sessions:
        sid = s["session"]
        for e in s["events"]:
            rows.append(
                {
                    "session": sid,
                    "aid": e["aid"],
                    "ts": e["ts"],
                    "type": e["type"],
                }
            )
    event_df = pd.DataFrame(rows)
    return event_df


def build_session_features(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-level logs into session-level features.
    Returns DataFrame indexed by session with:
      total_events, click_cnt, cart_cnt, order_cnt, converted
    """
    if event_df.empty:
        raise ValueError("event_df is empty. Check input sessions or file path.")

    session_features = event_df.groupby("session").agg(
        total_events=("type", "count"),
        click_cnt=("type", lambda x: (x == "clicks").sum()),
        cart_cnt=("type", lambda x: (x == "carts").sum()),
        order_cnt=("type", lambda x: (x == "orders").sum()),
    )

    session_features["converted"] = session_features["order_cnt"] > 0
    return session_features


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe as parquet (optional helper)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)