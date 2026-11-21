from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rag"
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    COMPLETION_MODEL: str = "gpt-4o-mini"

    # Default retrieval & reranking parameters
    DEFAULT_RRF_K: int = 60
    DEFAULT_HALF_LIFE_DAYS: float = 30.0

    # Default weights: (V, T, C, S)
    WEIGHT_V: float = 0.50
    WEIGHT_T: float = 0.25
    WEIGHT_C: float = 0.15
    WEIGHT_S: float = 0.10


settings = Settings()
