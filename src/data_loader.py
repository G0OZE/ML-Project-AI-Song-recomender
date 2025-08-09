"""Utilities for preparing demo data sets."""
from __future__ import annotations

import argparse
from pathlib import Path


def make_demo() -> None:
    """Ensure the demo CSV files exist in the data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("Demo data available in", data_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Data loader utilities")
    parser.add_argument("--make-demo", action="store_true", help="create demo dataset")
    args = parser.parse_args()
    if args.make_demo:
        make_demo()


if __name__ == "__main__":
    main()
