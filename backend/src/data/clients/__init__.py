"""Database clients and adapters package."""

from backend.src.data.clients.base import DatabaseAdapter
from backend.src.data.clients.detector import detect_database_type
from backend.src.data.clients.mongo.adapter import MongoDBAdapter
from backend.src.data.clients.postgres.adapter import PostgreSQLAdapter

__all__ = [
    "DatabaseAdapter",
    "MongoDBAdapter",
    "PostgreSQLAdapter",
    "detect_database_type",
]
