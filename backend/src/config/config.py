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
    embeddings_dir: str = Field(
        default="backend/src/data/embeddings",
        alias="EMBEDDINGS_DIR",
        description=(
            "Directory holding the embeddings of databases added at runtime from a "
            "connection string. The three built-in targets keep their own explicit "
            "path settings above; a database supplied through the frontend has no "
            "settings field of its own, so its vectors are named after its "
            "generated key inside this directory."
        ),
    )
    connections_path: str = Field(
        default="backend/src/data/schemas/connections.json",
        alias="CONNECTIONS_PATH",
        description=(
            "Where connection strings supplied through the frontend are stored so "
            "they survive a restart. Holds credentials in plain text -- it is "
            "gitignored, and should be treated like the .env file."
        ),
    )
    similarity_threshold: float = Field(
        default=0.25,
        alias="SIMILARITY_THRESHOLD",
        description="Cosine similarity threshold for candidate table retrieval (strictly > threshold)",
    )
    max_candidate_tables: int = Field(
        default=15,
        alias="MAX_CANDIDATE_TABLES",
        description="Maximum number of candidate tables to retrieve",
    )
    max_retries: int = Field(
        default=3,
        alias="MAX_RETRIES",
        description="Maximum retry attempts for table selection and query generation",
    )
    model_disable_thinking: bool = Field(
        default=True,
        alias="MODEL_DISABLE_THINKING",
        description=(
            "Append Qwen3's '/no_think' soft switch to each prompt. Qwen3-4B is a "
            "hybrid reasoning model: left to itself it opens a <think> block whose "
            "length is unbounded, and the reasoning is charged to the same "
            "completion budget as the answer. On ingestion's describe stage, which "
            "asks for one line per column across a batch of tables, the reasoning "
            "leaves too little budget for the JSON and the object is cut off "
            "mid-string. Turning thinking off reclaims the whole budget for the "
            "answer. Set false for a model with no thinking mode to suppress."
        ),
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
    query_timeout_seconds: int = Field(
        default=10,
        alias="QUERY_TIMEOUT_SECONDS",
        description="Timeout for database query execution",
    )
    model_timeout_seconds: int = Field(
        default=300,
        alias="MODEL_TIMEOUT_SECONDS",
        description=(
            "Timeout for HTTP calls to KoboldCpp (generation and embeddings). Kept "
            "separate from QUERY_TIMEOUT_SECONDS because a local SLM can legitimately "
            "take minutes -- the ingestion describe stage asks for a table description "
            "plus one per column for a whole batch -- while a database query that "
            "hasn't returned in 10s should be abandoned."
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
