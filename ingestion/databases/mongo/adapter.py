"""MongoDB database adapter."""

from typing import Any
from pymongo import MongoClient

from query_processing.core.config import settings
from ingestion.databases.base import DatabaseAdapter
from ingestion.databases.mongo.metadata import MongoDBMetadataExtractor
from query_processing.models.schema import DatabaseSchema


class MongoDBAdapter(DatabaseAdapter):
    """Adapter for connecting to MongoDB and extracting schema metadata."""

    def __init__(
        self,
        uri: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.mongodb_uri
        self.database_name = database or settings.mongodb_database
        if not self.uri:
            raise ValueError("MongoDB URI is required. Set MONGODB_URI in .env.")
        if not self.database_name:
            raise ValueError("MongoDB database name is required. Set MONGODB_DATABASE in .env.")
        self._client: MongoClient[dict[str, Any]] | None = None

    def _get_client(self) -> MongoClient[dict[str, Any]]:
        if self._client is None:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        return self._client

    def get_metadata(self) -> DatabaseSchema:
        """Extract authoritative MongoDB schema metadata."""
        client = self._get_client()
        db = client[self.database_name]
        extractor = MongoDBMetadataExtractor(db)
        return extractor.extract_schema()

    def close(self) -> None:
        """Close MongoDB connection client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "MongoDBAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
