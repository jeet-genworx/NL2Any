"""API request and input schemas."""

from pydantic import BaseModel, Field


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
