"""Central settings. Secrets come from .env and are held as SecretStr so they never show up in reprs or logs."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Secrets
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    neo4j_password: SecretStr | None = None

    @field_validator("gemini_api_key", "openrouter_api_key", "neo4j_password", mode="before")
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        # "KEY=" in .env arrives as an empty string; treat it as not set so the error says so.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"

    # Models (pinned in docs/PLAN.md)
    extractor_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.8-flash"
    generator_model: str = "openai/gpt-oss-120b"
    generator_provider: str = "deepinfra/bf16"      # OpenRouter endpoint tag; fallbacks disabled
    generator_reasoning_effort: str = "medium"      # same for every system
    embedding_model: str = "intfloat/e5-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Paths
    cache_dir: Path = ROOT / ".cache"
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"


def get_settings() -> Settings:
    return Settings()
