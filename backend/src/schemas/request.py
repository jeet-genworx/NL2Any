from typing import Literal
from pydantic import BaseModel, Field


class StageProviders(BaseModel):
    """Per-stage model provider selection for NL2AnyQuery pipeline."""

    table_selection: Literal["koboldcpp", "huggingface", "gemini"] = Field(
        default="koboldcpp",
        description="Provider for Stage 1 (Guardrail & Table Selection)",
    )
    query_generation: Literal["koboldcpp", "huggingface", "gemini"] = Field(
        default="koboldcpp",
        description="Provider for Stage 2 (Query Generation)",
    )
    validation: Literal["koboldcpp", "huggingface", "gemini"] = Field(
        default="koboldcpp",
        description="Provider for Stage 3 (Query Validation)",
    )


class QueryRequest(BaseModel):
    """Request model for the natural language query endpoint."""

    database: str = Field(
        default="postgres",
        description="Database target key: a built-in ('postgres', 'mongo', 'finops') or a saved connection's key",
    )
    question: str = Field(
        ...,
        description="Natural language question to query",
    )
    provider: Literal["koboldcpp", "huggingface", "gemini"] = Field(
        default="koboldcpp",
        description="Fallback/global LLM provider: 'koboldcpp', 'huggingface', or 'gemini'",
    )
    stages: StageProviders | None = Field(
        default=None,
        description="Optional independent per-stage provider configuration",
    )
    jargons: list[str] = Field(
        default_factory=list,
        description="Optional list of domain-specific jargon terms for spelling check",
    )
    table_selection_strategy: Literal["whole_schema", "bfs"] | None = Field(
        default=None,
        description=(
            "What the table selector is shown. 'whole_schema' sends every "
            "table and keeps the prompt prefix cacheable by the KoboldCpp "
            "KV slots; 'bfs' retrieves candidates by embedding similarity "
            "and widens along foreign keys, giving a much smaller but "
            "question-specific prompt with no prefix cache. PostgreSQL only: "
            "requesting 'bfs' for MongoDB falls back to 'whole_schema'. "
            "Unset uses the TABLE_SELECTION_STRATEGY setting."
        ),
    )


class RerunRequest(BaseModel):
    """Request model for partial pipeline rerun with per-stage feedback."""

    run_id: str = Field(
        ...,
        description="32-character hexadecimal ID of the original run to rerun from",
    )
    starting_stage: Literal["table_selection", "query_generation", "validation"] | None = Field(
        default=None,
        description="Stage to rerun from onward. If omitted, earliest stage with feedback is selected.",
    )
    feedback: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of stage name to user feedback text",
    )


class ConnectionRequest(BaseModel):
    """A connection string supplied through the frontend."""

    connection_string: str = Field(
        ...,
        min_length=1,
        description=(
            "PostgreSQL DSN, MongoDB URI, or libpq 'host=... dbname=...' string. "
            "Passed to the driver untouched; only the engine, database name and "
            "host are read from it."
        ),
    )
    label: str | None = Field(
        default=None,
        description="Display name for the picker. Derived from the database name when omitted.",
    )


class ConnectionResponse(BaseModel):
    """A saved database, as the picker sees it.

    Carries no connection string: it is written once, travels to the API once,
    and stays there. The frontend identifies the database by `key` afterwards.
    """

    key: str = Field(..., description="Stable key identifying this database in /ingest and /query")
    label: str
    database_type: str
    database_name: str = Field(default="", description="Database name read from the connection string")
    ingested: bool = Field(
        default=False,
        description="Whether this database already has embeddings on disk and can be queried without ingesting",
    )
    already_saved: bool = Field(
        default=False,
        description="True when this connection string was already known, and an existing entry was returned",
    )
    source: str = Field(default="user", description="'builtin' for a configured target, 'user' for a saved connection")
