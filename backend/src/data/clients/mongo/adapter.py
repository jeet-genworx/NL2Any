"""MongoDB database adapter."""

from typing import Any
from pymongo import MongoClient

from backend.src.config import settings
from backend.src.data.clients.base import DatabaseAdapter
from backend.src.data.clients.mongo.metadata import MongoDBMetadataExtractor
from backend.src.data.models.schema import DatabaseSchema


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
        extractor = MongoDBMetadataExtractor(db, sample_limit=settings.mongo_sample_limit)
        return extractor.extract_schema()

    def ping(self) -> None:
        """Round-trip the server, raising if the URI does not reach one.

        MongoClient construction is lazy and never fails on a bad host, so a
        command has to be issued for the URI to be proven at all.
        """
        self._get_client().admin.command("ping")

    def close(self) -> None:
        """Close MongoDB connection client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "MongoDBAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
