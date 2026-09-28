"""Builds the database adapter for a database type.

Lives beside the adapters rather than in the ingestion pipeline: choosing an
adapter and wiring it to its connection settings is a data-layer concern, and
anything needing metadata extraction can now ask for one without importing
ingestion.
"""

from backend.src.config import settings
from backend.src.data.clients.base import DatabaseAdapter
from backend.src.data.clients.mongo.adapter import MongoDBAdapter
from backend.src.data.clients.postgres.adapter import PostgreSQLAdapter
from backend.src.data.models.schema import DatabaseType


def get_adapter(database_type: DatabaseType) -> DatabaseAdapter:
    """Build the adapter for a database type, using connection settings from .env."""
    if database_type == DatabaseType.POSTGRESQL:
        return PostgreSQLAdapter(dsn=settings.postgres_dsn)
    return MongoDBAdapter(uri=settings.mongodb_uri, database=settings.mongodb_database)
