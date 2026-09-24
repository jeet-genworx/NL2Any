"""Database seeding utilities."""

from ingestion.databases.seed.mongo import seed_mongo
from ingestion.databases.seed.postgres import seed_postgres

__all__ = ["seed_postgres", "seed_mongo"]
