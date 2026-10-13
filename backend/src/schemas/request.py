from typing import Literal
from pydantic import BaseModel, Field

ProviderName = Literal["koboldcpp", "huggingface", "gemini"]


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
    provider: ProviderName = Field(
        default="koboldcpp",
        description=(
            "Default LLM provider for every stage: 'koboldcpp', 'huggingface', or "
            "'gemini'. The stage-specific fields below override it where set."
        ),
    )
    selector_provider: ProviderName | None = Field(
        default=None,
        description=(
            "Provider for table selection. Falls back to `provider` when unset. "
            "Separate because selection reads a whole schema neighborhood and "
            "generation writes one query, and the two do not need the same model."
        ),
    )
    generator_provider: ProviderName | None = Field(
        default=None,
        description=(
            "Provider for query planning, generation and validation -- the loop "
            "that turns settled tables into a query. Falls back to `provider` "
            "when unset. Planning and validation ride with generation because "
            "they feed back into it on each retry."
        ),
    )
    jargons: list[str] = Field(
        default_factory=list,
        description="Optional list of domain-specific jargon terms for spelling check",
    )
    allow_table_approval: bool = Field(
        default=False,
        description=(
            "Whether this caller can answer a question. With it set, a run whose "
            "table selection is uncertain comes back with status "
            "'awaiting_table_approval' and a run id instead of a result, and is "
            "finished by POSTing the chosen tables to /query/{run_id}/tables. "
            "Left false, such a run proceeds as before and reports its doubt on "
            "selection_notice."
        ),
    )


class TableApprovalDecision(BaseModel):
    """The tables a human settled on for a run paused at the table gate."""

    approved_objects: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Final table list for the query. May only name tables from the "
            "paused run's candidate set; bridge tables needed to join them are "
            "still added automatically, and they need not be listed."
        ),
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
