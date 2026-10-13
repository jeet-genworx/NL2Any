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
        description=(
            "Gemini model ID. Bare (e.g. 'gemini-2.5-flash') when talking to "
            "Google directly; whatever name the gateway exposes when "
            "GEMINI_BASE_URL is set, which for LiteLLM means the provider "
            "prefix too (e.g. 'gemini/gemini-2.5-flash')"
        ),
    )
    gemini_base_url: str | None = Field(
        default=None,
        alias="GEMINI_BASE_URL",
        description=(
            "Optional OpenAI-compatible gateway to reach Gemini through, such as "
            "a LiteLLM proxy. Set it and GEMINI_API_KEY becomes the gateway's "
            "key (LiteLLM issues 'sk-...' virtual keys) rather than a Google "
            "'AIza...' key. Left unset, the provider calls Google directly, so "
            "the presence of this URL is the only switch between the two."
        ),
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
    embedding_dimension: int = Field(
        default=768,
        alias="EMBEDDING_DIMENSION",
        description=(
            "Length of the vectors the embedding model produces. Stored embeddings "
            "and query embeddings are both checked against it, so changing the "
            "model to one with a different size means changing this and re-ingesting."
        ),
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
    def soft_delete_column_names(self) -> tuple[str, ...]:
        """Configured deletion-marker column names, lowercased, in priority order."""
        return tuple(
            part.strip().lower() for part in self.soft_delete_columns.split(",") if part.strip()
        )

    @property
    def row_scope_column_names(self) -> tuple[str, ...]:
        """Configured tenant/organisation scope column names, lowercased."""
        return tuple(
            part.strip().lower() for part in self.row_scope_columns.split(",") if part.strip()
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
    table_approval_enabled: bool = Field(
        default=True,
        alias="TABLE_APPROVAL_ENABLED",
        description=(
            "Whether a run may pause for a human to settle its tables when the "
            "selector is not certain. A request still has to ask for the gate "
            "with allow_table_approval, so this is the deployment-wide switch "
            "that turns the pause off for every caller at once -- useful when "
            "the API is being driven by something that cannot answer."
        ),
    )
    table_approval_ttl_seconds: int = Field(
        default=900,
        alias="TABLE_APPROVAL_TTL_SECONDS",
        description=(
            "How long a run paused for table approval is kept before it is "
            "discarded. A paused run holds its neighborhood and analysis in "
            "memory, and a human who closes the tab never comes back, so it "
            "expires rather than waiting forever."
        ),
    )
    soft_delete_columns: str = Field(
        default="is_deleted,deleted_at,is_archived,archived_at",
        alias="SOFT_DELETE_COLUMNS",
        description=(
            "Comma-separated column names, in priority order, that a database "
            "uses to mark a row deleted rather than removing it. Which name a "
            "schema uses is a house convention, not a standard -- is_deleted, "
            "deleted_at and is_archived are all common, and plenty of schemas "
            "have none. The generator is told which of these each table in the "
            "slice actually has, so it never has to guess: told merely to "
            "'exclude deleted rows', a small model adds is_deleted to tables "
            "that do not have the column. Set empty to disable the hint."
        ),
    )
    row_scope_columns: str = Field(
        default="tenant_id,org_id,organisation_id,organization_id,account_id,workspace_id",
        alias="ROW_SCOPE_COLUMNS",
        description=(
            "Comma-separated column names that partition rows between tenants, "
            "organisations or accounts. When both sides of a join carry one, it "
            "belongs in the ON clause: joining on the key alone can pair rows "
            "from different tenants. Single-tenant databases should set this "
            "empty."
        ),
    )
    table_neighborhood_max_levels: int = Field(
        default=3,
        alias="TABLE_NEIGHBORHOOD_MAX_LEVELS",
        description=(
            "How many foreign-key hops the breadth-first walk takes out from "
            "each embedding match when assembling the tables the selector "
            "chooses from. Embedding matching finds the tables a question "
            "names; the tables a query must touch are often one or two hops "
            "past them -- the bridge table nobody mentions, and the entity "
            "table on its far side. 3 reaches those while staying well short "
            "of the whole schema."
        ),
    )
    table_neighborhood_max_tables: int = Field(
        default=40,
        alias="TABLE_NEIGHBORHOOD_MAX_TABLES",
        description=(
            "Ceiling on unique tables handed to the table selector after the "
            "walk. Three hops through a densely normalized schema can reach "
            "most of the database, and the selector's prompt has to fit a "
            "context window. Embedding matches are never dropped; past them, "
            "nearer tables survive truncation, so what is discarded is always "
            "the weakest-related end of the walk."
        ),
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
    model_context_tokens: int = Field(
        default=16384,
        alias="MODEL_CONTEXT_TOKENS",
        description=(
            "Context window of the generation model, in tokens, shared by the "
            "prompt and the completion. The server reserves the completion "
            "budget first and truncates the PROMPT to fit, silently dropping "
            "its beginning -- which is where the instructions are. Stages that "
            "render schema into their prompt check against this to degrade "
            "what they include instead of overflowing. Must match the context "
            "KoboldCpp was started with (--contextsize)."
        ),
    )
    prompt_chars_per_token: float = Field(
        default=4.5,
        alias="PROMPT_CHARS_PER_TOKEN",
        description=(
            "Characters-per-token ratio used to size prompts without a "
            "tokenizer round-trip. Measured against KoboldCpp's own tokenizer "
            "on real selector prompts, schema text runs 4.8-5.1 chars/token -- "
            "better than ordinary prose, because repeated table and column "
            "identifiers compress well under BPE. 4.5 keeps a margin under the "
            "measured floor, and the caller holds back a further 10%. Lower it "
            "if prompts still overflow on a different model's tokenizer."
        ),
    )
    table_selector_max_tokens: int = Field(
        default=1500,
        alias="TABLE_SELECTOR_MAX_TOKENS",
        description=(
            "Maximum tokens generated by the table selector model. This is "
            "subtracted from the model's context window before the prompt is "
            "read, so an oversized value silently truncates the prompt instead "
            "of the answer: at 4000 against an 8192-token context, the whole "
            "instruction block was cut off and the model replied with a prose "
            "schema summary rather than JSON. The selector's answer is a short "
            "JSON object, so 1000 is ample."
        ),
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
