"""Miscellaneous helper utilities."""
from __future__ import annotations

import pandas as pd


def popularity_recommend(events: pd.DataFrame, k: int = 5) -> list[int]:
    """Return top-k tracks by play count."""
    counts = events[events["event_type"] == "play"]["track_id"].value_counts()
    return list(counts.head(k).index)
