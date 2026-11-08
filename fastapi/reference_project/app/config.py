from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIB_", env_file=".env", extra="ignore")

    app_name: str = "Library API"
    database_url: str = "sqlite:///library.db"   # "memory" -> in-memory adapter


@lru_cache
def get_settings() -> Settings:
    return Settings()
