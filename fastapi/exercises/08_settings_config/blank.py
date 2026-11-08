"""
Exercise 08 - Configuration with pydantic-settings (env vars, .env, cached)  [config]

12-factor config: settings come from the environment, never hard-coded.
1. class Settings(BaseSettings):
       model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")
       app_name: str = "My API" ; debug: bool = False ; database_url: str = "sqlite:///app.db"
       secret_key: SecretStr          (required! no default; SecretStr hides it when printed)
       allowed_origins: list[str] = ["http://localhost:3000"]
2. @lru_cache def get_settings() -> Settings: return Settings()   (read env once)
3. app = FastAPI(); GET /info returns {"app_name": ..., "debug": ...} using Depends(get_settings)
4. Test: set os.environ["APP_SECRET_KEY"] = "s3cret", os.environ["APP_DEBUG"] = "true",
   get_settings.cache_clear(), then check values; secret_key.get_secret_value() == "s3cret"
   and str(settings.secret_key) shows "**********"
5. Missing APP_SECRET_KEY -> pydantic ValidationError at startup (fail fast = good).
Self-check expects these names: app, Settings, get_settings

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    import os
    from pydantic import ValidationError
    os.environ["APP_SECRET_KEY"] = "abc"
    os.environ["APP_DEBUG"] = "true"
    os.environ["APP_ALLOWED_ORIGINS"] = '["https://a.com", "https://b.com"]'
    get_settings.cache_clear()
    s = get_settings()
    assert s.debug is True and s.secret_key.get_secret_value() == "abc"
    assert "abc" not in str(s.secret_key)
    assert s.allowed_origins == ["https://a.com", "https://b.com"]
    assert get_settings() is s
    assert TestClient(app).get("/info").json() == {"app_name": "My API", "debug": True}
    del os.environ["APP_SECRET_KEY"]
    try:
        Settings(_env_file=None)
        raise AssertionError("should fail without secret")
    except ValidationError:
        pass
    print("All checks passed!")
