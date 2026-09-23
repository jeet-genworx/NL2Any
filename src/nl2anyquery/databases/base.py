"""Base protocol for database adapters focused on schema discovery."""

from typing import Protocol, runtime_checkable
from nl2anyquery.models.schema import DatabaseSchema


@runtime_checkable
class DatabaseAdapter(Protocol):
    """Protocol for database schema and metadata discovery.

    Focused strictly on metadata and schema extraction in Part 1.
    """

    def get_metadata(self) -> DatabaseSchema:
        """Extract authoritative database metadata and return a normalized DatabaseSchema."""
        ...

    def close(self) -> None:
        """Close database connections and release resources."""
        ...
