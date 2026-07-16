from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE_PATH = _PROJECT_ROOT / ".env"
_SECRET_DIR_PATH = _PROJECT_ROOT / "secrets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_PATH,
        secrets_dir=_SECRET_DIR_PATH,
        env_file_encoding="utf_8",
        case_sensitive=False,
        extra="ignore",
    )

    # logging level
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG", description="Logging level"
    )

    # Model config
    token: Optional[SecretStr] = Field(
        default=None, description="Model provider API token"
    )
    embedding_provider: Literal["ollama", "github_openai"] = Field(
        default="ollama", description="Embedding provider"
    )
    generation_provider: Literal["ollama", "github_openai"] = Field(
        default="ollama", description="Response generation provider"
    )
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="HuggingFace sentence transformer model for embeddings",
    )
    generator_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="HuggingFace Causla-LM model for answer generation",
    )
    embedding_dim: int = Field(default=768, description="Embedding dimention")
    llm_provider_base_url: str = Field(
        default="https://models.inference.ai.azure.com",
        description="LLM providers base url",
    )

    # Redis config
    redis_url: str = Field(
        default="redis://medico-redis:6379",
        description="Redis connection URL (redis://host:port)",
    )
    embedding_cache_ttl: int = Field(
        default=86400,
        description="Embedding cache TTL in seconds (default: 24 h)",
        ge=60,
    )

    # Vector store
    vector_store_provider: Literal["qdrant"] = Field(
        default="qdrant", description="Vector store provider"
    )

    # Qdrant config
    qdrant_url: str = Field(
        default="http://qdrant-store:6333", description="QDrant base url"
    )
    qdrant_key: Optional[str] = Field(default=None, description="QDrant API key")
    qdrant_collection: str = Field(
        default="medquad_documents",
        description="Qdrant collection name",
    )

    # LangFuse
    langfuse_public_key: Optional[str] = Field(
        default=None, description="Langfuse public key"
    )
    langfuse_secret_key: Optional[SecretStr] = Field(
        default=None, description="Langfuse secret key"
    )
    langfuse_host: str = Field(
        default="http://langfuse:3000",
        description="Langfuse self-hosted instance URL",
    )

    # Application
    app_env: Literal["development", "staging", "test", "production"] = "development"

    # Chunking
    chunk_size: int = Field(
        default=512, description="Document chunk size in token", gt=100, le=1024
    )
    chunk_overlap: int = Field(
        default=65, description="chunk overlap in token", gt=0, le=256
    )

    # Retriever
    top_k: int = Field(
        default=5, description="Number of chunk to retrieve per query", gt=1, le=10
    )
    max_retrieval_iterations: int = Field(
        default=3, description="Maxumum allowed retrieval before generaton", gt=1
    )
    relevancy_threshold: float = Field(
        default=0.8,
        description="Minimum cosine similarity score to consider a chunk relevant",
        gt=0,
        le=1,
    )

    # Guardrails AI
    guardrails_api_key: Optional[str] = Field(
        default=None, description="Guardrails AI api key"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton (cached after first call).

    Returns:
        Settings: The application settings object.
    """

    return Settings()
