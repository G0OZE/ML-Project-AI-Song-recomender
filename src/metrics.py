"""Evaluation metrics for recommendation."""
from __future__ import annotations

import argparse
from typing import Iterable, List
import numpy as np
import pandas as pd


def precision_at_k(recommended: List[int], relevant: Iterable[int], k: int) -> float:
    if k == 0:
        return 0.0
    rel = set(relevant)
    hit = len(set(recommended[:k]) & rel)
    return hit / k


def recall_at_k(recommended: List[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hit = len(set(recommended[:k]) & rel)
    return hit / len(rel)


def apk(recommended: List[int], relevant: Iterable[int], k: int) -> float:
    score = 0.0
    hits = 0
    rel = set(relevant)
    for i, r in enumerate(recommended[:k], start=1):
        if r in rel:
            hits += 1
            score += hits / i
    return score / min(len(rel), k)


def map_at_k(recommended: List[List[int]], relevant: List[Iterable[int]], k: int) -> float:
    return float(np.mean([apk(r, rel, k) for r, rel in zip(recommended, relevant)]))


def evaluate(events_csv: str, k: int) -> dict[str, float]:
    df = pd.read_csv(events_csv).sort_values("timestamp")
    recs: List[List[int]] = []
    rels: List[List[int]] = []
    for _user, group in df.groupby("user_id"):
        history = list(group["track_id"])[:-1]
        test = group["track_id"].iloc[-1]
        recs.append(history[::-1])
        rels.append([test])
    return {
        "precision@k": precision_at_k(recs[0], rels[0], k),
        "recall@k": recall_at_k(recs[0], rels[0], k),
        "map@k": map_at_k(recs, rels, k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendations")
    parser.add_argument("--events", required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    metrics = evaluate(args.events, args.k)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
