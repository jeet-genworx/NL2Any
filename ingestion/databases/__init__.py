"""Database abstraction and detection module."""

from ingestion.databases.base import DatabaseAdapter
from ingestion.databases.detector import detect_database_type

__all__ = ["DatabaseAdapter", "detect_database_type"]
