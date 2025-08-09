import pandas as pd
from src.features import build_features


def test_build_features(tmp_path):
    out = tmp_path / "features.parquet"
    df = build_features("data/sample_tracks.csv", out)
    assert "energy" in df.columns
    assert len(df) == 4
