"""Database seeding utilities."""

from nl2anyquery.databases.seed.mongo import seed_mongo
from nl2anyquery.databases.seed.postgres import seed_postgres

__all__ = ["seed_postgres", "seed_mongo"]
