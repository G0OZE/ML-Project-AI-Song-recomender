"""FastAPI application exposing recommendation endpoints."""
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd

from ..features import build_features
from ..models.collab_als import ALSRecommender
from ..models.content_based import ContentRecommender
from ..models.hybrid import HybridRecommender
from ..schemas import EventSchema, RecommendRequest, TrackSchema
from ..storage import Event, Track, get_session, init_db

app = FastAPI(title="Song Recommender")
init_db()

tracks_df = pd.read_csv("data/sample_tracks.csv")
if Path("data/features.parquet").exists():
    features_df = pd.read_parquet("data/features.parquet")
else:
    features_df = build_features("data/sample_tracks.csv", "data/features.parquet")
content_model = ContentRecommender(features_df)

events_df = pd.read_csv("data/sample_events.csv")
als_model = ALSRecommender()
als_model.fit(events_df)

hybrid_model = HybridRecommender(als_model, content_model)

with get_session() as session:
    for row in tracks_df.to_dict(orient="records"):
        session.merge(Track(**row))
    for row in events_df.to_dict(orient="records"):
        session.add(Event(**row))
    session.commit()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, list[dict[str, float]]]:
    recs = hybrid_model.recommend(req.user_id, req.k)
    results = []
    for tid, score in recs:
        row = tracks_df[tracks_df["track_id"] == tid].iloc[0]
        results.append({"track_id": tid, "title": row["title"], "score": score})
    return {"tracks": results}


@app.post("/event")
def log_event(event: EventSchema) -> dict[str, str]:
    with get_session() as session:
        session.add(Event(**event.dict()))
        session.commit()
    return {"status": "logged"}


@app.get("/tracks/{track_id}")
def get_track(track_id: int) -> TrackSchema:
    row = tracks_df[tracks_df["track_id"] == track_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Track not found")
    r = row.iloc[0]
    return TrackSchema(track_id=int(r["track_id"]), title=r["title"], artist=r["artist"], genre=r["genre"])
