"""Database models and utilities."""
from __future__ import annotations

from typing import Optional
from sqlmodel import SQLModel, Field, Session, create_engine

from .config import settings


class Track(SQLModel, table=True):
    track_id: int = Field(primary_key=True)
    title: str
    artist: str
    genre: Optional[str] = None


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    track_id: int
    event_type: str
    timestamp: str


def get_engine():
    return create_engine(settings.database_url, echo=False)


def init_db() -> None:
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    engine = get_engine()
    return Session(engine)
