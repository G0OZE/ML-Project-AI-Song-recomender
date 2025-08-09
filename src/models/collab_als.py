"""Collaborative filtering with implicit ALS."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


WEIGHTS = {"play": 1.0, "like": 3.0, "skip": -1.0}


@dataclass
class ALSRecommender:
    factors: int = 20
    regularization: float = 0.1
    iterations: int = 20

    def __post_init__(self) -> None:
        self.model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization, iterations=self.iterations
        )

    def fit(self, events: pd.DataFrame) -> None:
        users = {u: i for i, u in enumerate(events["user_id"].unique())}
        items = {t: i for i, t in enumerate(events["track_id"].unique())}
        rows = events["user_id"].map(users)
        cols = events["track_id"].map(items)
        data = events["event_type"].map(WEIGHTS).clip(lower=0.0)
        mat = coo_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        self.model.fit(mat)
        self.mat = mat.tocsr()
        self.user_codes = users
        self.item_codes = items
        self.inv_item_codes = {i: t for t, i in items.items()}

    def recommend(self, user_id: str, k: int = 5) -> list[tuple[int, float]]:
        if user_id not in self.user_codes:
            return []
        idx = self.user_codes[user_id]
        recs = self.model.recommend(idx, self.mat, N=k)
        return [(self.inv_item_codes[i], float(score)) for i, score in recs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALS model")
    parser.add_argument("--events", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.events)
    rec = ALSRecommender()
    rec.fit(df)
    np.savez(args.model, user_factors=rec.model.user_factors, item_factors=rec.model.item_factors)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
