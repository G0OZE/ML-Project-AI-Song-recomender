"""Hybrid recommender blending ALS and content similarity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .collab_als import ALSRecommender
from .content_based import ContentRecommender


@dataclass
class HybridRecommender:
    als: ALSRecommender
    content: ContentRecommender
    alpha: float = 0.6

    def recommend(self, user_id: str, k: int = 5) -> List[Tuple[int, float]]:
        cf = self.als.recommend(user_id, k)
        cf_scores = {tid: score for tid, score in cf}
        content_scores: dict[int, float] = {}
        if cf:
            top_track = cf[0][0]
            for tid, score in self.content.recommend(top_track, k=k * 2):
                content_scores[tid] = score
        all_ids = set(cf_scores) | set(content_scores)
        blended = []
        for tid in all_ids:
            a = cf_scores.get(tid, 0.0)
            c = content_scores.get(tid, 0.0)
            blended.append((tid, self.alpha * a + (1 - self.alpha) * c))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:k]


def main() -> None:
    import argparse
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Tune hybrid recommender")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    if args.tune:
        print("Demo tune: load sample data and output top recommendations for user u1")
        events = pd.read_csv("data/sample_events.csv")
        als = ALSRecommender()
        als.fit(events)
        feats = pd.read_parquet("data/features.parquet")
        content = ContentRecommender(feats)
        hybrid = HybridRecommender(als, content)
        print(hybrid.recommend("u1", k=args.k))


if __name__ == "__main__":
    main()
