"""Database abstraction and detection module."""

from nl2anyquery.databases.base import DatabaseAdapter
from nl2anyquery.databases.detector import detect_database_type

__all__ = ["DatabaseAdapter", "detect_database_type"]
