from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    database_url: str = "sqlite:///./demo.db"
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
