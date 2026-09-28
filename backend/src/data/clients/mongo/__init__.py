"""MongoDB client and metadata package."""

from backend.src.data.clients.mongo.adapter import MongoDBAdapter
from backend.src.data.clients.mongo.metadata import MongoDBMetadataExtractor

__all__ = ["MongoDBAdapter", "MongoDBMetadataExtractor"]
