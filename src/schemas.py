"""Pydantic schemas for API requests/responses."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_id: str
    k: int = 20


class EventSchema(BaseModel):
    user_id: str
    track_id: int
    event_type: str
    timestamp: Optional[str] = None


class TrackSchema(BaseModel):
    track_id: int
    title: str
    artist: str
    genre: Optional[str] = None
