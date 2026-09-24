"""PostgreSQL database package."""

from ingestion.databases.postgres.adapter import PostgreSQLAdapter
from ingestion.databases.postgres.metadata import PostgreSQLMetadataExtractor

__all__ = ["PostgreSQLAdapter", "PostgreSQLMetadataExtractor"]
