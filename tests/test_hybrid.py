import pandas as pd
from src.features import build_features
from src.models.collab_als import ALSRecommender
from src.models.content_based import ContentRecommender
from src.models.hybrid import HybridRecommender


def test_hybrid(tmp_path):
    events = pd.read_csv("data/sample_events.csv")
    als = ALSRecommender(factors=2, iterations=5)
    als.fit(events)
    feats = build_features("data/sample_tracks.csv", tmp_path / "f.parquet")
    content = ContentRecommender(feats)
    hybrid = HybridRecommender(als, content)
    recs = hybrid.recommend("u1", k=3)
    assert recs
