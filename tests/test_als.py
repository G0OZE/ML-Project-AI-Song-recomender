import pandas as pd
from src.models.collab_als import ALSRecommender


def test_als_recommend():
    events = pd.read_csv("data/sample_events.csv")
    model = ALSRecommender(factors=2, iterations=5)
    model.fit(events)
    recs = model.recommend("u1", k=2)
    assert isinstance(recs, list)
