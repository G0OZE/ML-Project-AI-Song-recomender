"""Feature extraction utilities."""
from __future__ import annotations

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["energy", "danceability", "valence", "tempo"]


def build_features(input_csv: str, out_path: str) -> pd.DataFrame:
    """Load track metadata and compute z-scored feature matrix."""
    df = pd.read_csv(input_csv)
    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATURE_COLUMNS])
    feat_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    feat_df.insert(0, "track_id", df["track_id"])
    feat_df.to_parquet(out_path, index=False)
    return feat_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio/metadata features")
    parser.add_argument("--input", required=True, help="input metadata CSV")
    parser.add_argument("--out", required=True, help="output parquet path")
    args = parser.parse_args()
    build_features(args.input, args.out)
    print(f"Features written to {args.out}")


if __name__ == "__main__":
    main()
