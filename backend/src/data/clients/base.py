"""Base protocol for database adapters focused on schema discovery."""

from typing import Protocol, runtime_checkable
from backend.src.data.models.schema import DatabaseSchema


@runtime_checkable
class DatabaseAdapter(Protocol):
    """Protocol for database schema and metadata discovery.

    Focused strictly on metadata and schema extraction in Part 1.
    """

    def get_metadata(self) -> DatabaseSchema:
        """Extract authoritative database metadata and return a normalized DatabaseSchema."""
        ...

    def ping(self) -> None:
        """Prove the connection works, raising if it does not.

        Cheaper than `get_metadata` and used to validate a connection string the
        moment a user supplies one, rather than letting a bad host surface
        minutes later as an ingestion failure.
        """
        ...

    def close(self) -> None:
        """Close database connections and release resources."""
        ...
