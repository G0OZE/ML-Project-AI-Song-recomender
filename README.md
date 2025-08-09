# ML-Project-AI-Song-recomender

An educational demo showing how to build a tiny song recommender system that mixes
content based similarity with collaborative filtering.  The project exposes a
FastAPI service and a Streamlit UI for quick experimentation.

```
User events  -->  ALS (implicit)
                    \\
                     Hybrid ----> API/UI
Tracks ----> Feature extraction --/
```

## Features
- Demo dataset with track metadata and user events
- Feature extraction with `pandas`/`scikit-learn`
- Collaborative filtering using `implicit` ALS
- Hybrid recommender blending collaborative and content signals
- FastAPI endpoints and Streamlit UI
- Simple evaluation metrics and unit tests

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Commands
```bash
# Build demo features and train models
python -m src.data_loader --make-demo
python -m src.features --input data/sample_tracks.csv --out data/features.parquet
python -m src.models.collab_als --events data/sample_events.csv --model artifacts/als.npz
python -m src.models.hybrid --tune --k 20

# Evaluate
python -m src.metrics --events data/sample_events.csv --k 20

# API
uvicorn src.api.main:app --reload

# UI
streamlit run ui/streamlit_app.py
```

## Dataset
CSV files in `data/` provide a tiny reproducible dataset so the project runs
out of the box.  Replace these with your own data or connect to the Spotify API
by filling out `.env` based on `.env.example`.

## Tests
Run the unit test suite with:
```bash
pytest
```
