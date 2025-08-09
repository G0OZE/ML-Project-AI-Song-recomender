from src.features import build_features
from src.models.content_based import ContentRecommender


def test_similarity(tmp_path):
    feats = build_features("data/sample_tracks.csv", tmp_path / "f.parquet")
    model = ContentRecommender(feats)
    recs = model.recommend(1, k=2)
    assert recs and recs[0][0] != 1
