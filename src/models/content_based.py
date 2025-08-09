"""Content-based recommender using cosine similarity."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentRecommender:
    """Recommend tracks based on feature cosine similarity."""

    features: pd.DataFrame

    def __post_init__(self) -> None:
        self.features = self.features.set_index("track_id")
        self.matrix = self.features.values
        self.ids = list(self.features.index)
        self.sim_matrix = cosine_similarity(self.matrix)

    def recommend(self, track_id: int, k: int = 5) -> list[tuple[int, float]]:
        idx = self.ids.index(track_id)
        sims = self.sim_matrix[idx]
        pairs = [(tid, score) for tid, score in zip(self.ids, sims) if tid != track_id]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]
