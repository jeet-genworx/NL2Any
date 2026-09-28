"""Database seeders package."""

from backend.src.data.seed.mongo import seed_mongo
from backend.src.data.seed.postgres import seed_postgres

__all__ = ["seed_mongo", "seed_postgres"]
