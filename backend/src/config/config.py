"""Core configuration and settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    koboldcpp_base_url: str = Field(
        default="http://127.0.0.1:5001/v1",
        alias="KOBOLDCPP_BASE_URL",
        description="KoboldCpp base URL",
    )
    koboldcpp_model: str = Field(
        default="qwen3-4b-instruct-2507",
        alias="KOBOLDCPP_MODEL",
        description="KoboldCpp model identifier",
    )
    koboldcpp_embedding_model: str = Field(
        default="all-MiniLM-L6-v2-Q8_0",
        alias="KOBOLDCPP_EMBEDDING_MODEL",
        description="KoboldCpp embedding model identifier (loaded via --embeddingsmodel)",
    )

    # Hugging Face Provider Configuration
    hf_api_key: str = Field(
        default="",
        alias="HF_API_KEY",
        description="Hugging Face User Access Token",
    )
    hf_model: str = Field(
        default="",
        alias="HF_MODEL",
        description="Hugging Face Model ID (e.g. meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    hf_base_url: str | None = Field(
        default=None,
        alias="HF_BASE_URL",
        description="Optional custom Hugging Face endpoint or TGI URL",
    )

    # Google Gemini Provider Configuration
    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        description="Google Gemini API Key",
    )
    gemini_model: str = Field(
        default="",
        alias="GEMINI_MODEL",
        description="Google Gemini Model ID (e.g. gemini-2.5-flash)",
    )

    postgres_dsn: str = Field(
        default="",
        alias="POSTGRES_DSN",
        description="PostgreSQL connection string DSN",
    )
    finops_dsn: str = Field(
        default="",
        alias="FINOPS_DSN",
        description=(
            "PostgreSQL DSN for the FinOps database. Shares the PostgreSQL engine "
            "with the demo database but is a separate target with its own schema "
            "and embeddings; empty means the target is not configured."
        ),
    )
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        alias="MONGODB_URI",
        description="MongoDB connection URI",
    )
    mongodb_database: str = Field(
        default="nl2anyquery_demo",
        alias="MONGODB_DATABASE",
        description="MongoDB database name",
    )
    mongo_sample_limit: int = Field(
        default=10,
        alias="MONGO_SAMPLE_LIMIT",
        description="Number of documents sampled per collection during MongoDB metadata extraction",
    )

    retrieval_top_k: int = Field(
        default=10,
        alias="RETRIEVAL_TOP_K",
        description="Default number of schema objects to retrieve via semantic search",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
        description="Embedding model name",
    )
    postgres_embeddings_path: str = Field(
        default="backend/src/data/embeddings/postgres_embeddings.json",
        alias="POSTGRES_EMBEDDINGS_PATH",
        description="Path to precomputed table description embeddings for PostgreSQL",
    )
    mongo_embeddings_path: str = Field(
        default="backend/src/data/embeddings/mongo_embeddings.json",
        alias="MONGO_EMBEDDINGS_PATH",
        description="Path to precomputed collection description embeddings for MongoDB",
    )
    finops_embeddings_path: str = Field(
        default="backend/src/data/embeddings/finops_embeddings.json",
        alias="FINOPS_EMBEDDINGS_PATH",
        description="Path to precomputed table description embeddings for the FinOps database",
    )
    similarity_threshold: float = Field(
        default=0.25,
        alias="SIMILARITY_THRESHOLD",
        description="Cosine similarity threshold for candidate table retrieval (strictly > threshold)",
    )
    similarity_top_k: int = Field(
        default=10,
        alias="SIMILARITY_TOP_K",
        description="Number of top candidate tables to retrieve via similarity matching",
    )
    max_candidate_tables: int = Field(
        default=10,
        alias="MAX_CANDIDATE_TABLES",
        description="Maximum number of candidate tables to retrieve (alias for SIMILARITY_TOP_K)",
    )

    @property
    def effective_similarity_top_k(self) -> int:
        """Effective top-K limit for candidate table similarity matching from settings."""
        if self.similarity_top_k != 10:
            return self.similarity_top_k
        return self.max_candidate_tables
    max_retries: int = Field(
        default=3,
        alias="MAX_RETRIES",
        description="Maximum retry attempts for table selection and query generation",
    )
    model_temperature: float = Field(
        default=0.1,
        alias="MODEL_TEMPERATURE",
        description="Model sampling temperature",
    )
    model_max_tokens: int = Field(
        default=1500,
        alias="MODEL_MAX_TOKENS",
        description="Maximum tokens generated by model",
    )
    table_selector_max_tokens: int = Field(
        default=4000,
        alias="TABLE_SELECTOR_MAX_TOKENS",
        description="Maximum tokens generated by table selector model",
    )
    validator_max_tokens: int = Field(
        default=1200,
        alias="VALIDATOR_MAX_TOKENS",
        description="Maximum tokens generated by validator model",
    )
    guardrail_max_tokens: int = Field(
        default=600,
        alias="GUARDRAIL_MAX_TOKENS",
        description="Maximum tokens generated by guardrail model",
    )
    query_timeout_seconds: int = Field(
        default=10,
        alias="QUERY_TIMEOUT_SECONDS",
        description="Timeout for database query execution",
    )
    model_timeout_seconds: int | None = Field(
        default=None,
        alias="MODEL_TIMEOUT_SECONDS",
        description=(
            "Timeout in seconds for HTTP calls to KoboldCpp (generation and embeddings). "
            "Set to None or 0 to completely disable timeouts for local SLM operations."
        ),
    )
    max_result_rows: int = Field(
        default=1000,
        alias="MAX_RESULT_ROWS",
        description="Maximum result rows allowed",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
