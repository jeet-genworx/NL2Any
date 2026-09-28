"""PostgreSQL client and metadata package."""

from backend.src.data.clients.postgres.adapter import PostgreSQLAdapter
from backend.src.data.clients.postgres.metadata import PostgreSQLMetadataExtractor

__all__ = ["PostgreSQLAdapter", "PostgreSQLMetadataExtractor"]
