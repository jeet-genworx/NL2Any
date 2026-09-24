"""MongoDB database package."""

from ingestion.databases.mongo.adapter import MongoDBAdapter
from ingestion.databases.mongo.metadata import MongoDBMetadataExtractor

__all__ = ["MongoDBAdapter", "MongoDBMetadataExtractor"]
