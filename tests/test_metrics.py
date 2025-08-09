from src.metrics import precision_at_k, recall_at_k, map_at_k, evaluate


def test_metrics_basic():
    rec = [1, 2, 3]
    rel = [2]
    assert precision_at_k(rec, rel, 2) == 0.5
    assert recall_at_k(rec, rel, 2) == 1.0
    assert map_at_k([rec], [rel], 3) > 0
    metrics = evaluate("data/sample_events.csv", k=2)
    assert "precision@k" in metrics
