"""PostgreSQL database adapter."""

from typing import Any
import psycopg

from nl2anyquery.core.config import settings
from nl2anyquery.databases.base import DatabaseAdapter
from nl2anyquery.databases.postgres.metadata import PostgreSQLMetadataExtractor
from nl2anyquery.models.schema import DatabaseSchema


class PostgreSQLAdapter(DatabaseAdapter):
    """Adapter for connecting to PostgreSQL and discovering schema metadata."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.postgres_dsn
        if not self.dsn:
            raise ValueError(
                "PostgreSQL connection string (DSN) is required. Set POSTGRES_DSN in .env."
            )
        self._conn: psycopg.Connection[Any] | None = None

    def _get_connection(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def get_metadata(self) -> DatabaseSchema:
        """Extract authoritative PostgreSQL schema metadata."""
        conn = self._get_connection()
        extractor = PostgreSQLMetadataExtractor(conn)
        return extractor.extract_schema()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PostgreSQLAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
