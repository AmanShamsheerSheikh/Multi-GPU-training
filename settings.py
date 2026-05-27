from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    PGBOUNCER_PORT: int
    PGBOUNCER_POOL_MODE: str
    PGBOUNCER_MAX_CLIENT_CONN: int
    PGBOUNCER_DEFAULT_POOL_SIZE: int
    RUNPOD_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()