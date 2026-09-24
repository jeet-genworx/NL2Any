"""Standalone entrypoint for seeding MongoDB with synthetic demo data."""

from ingestion.databases.seed.mongo import seed_mongo

if __name__ == "__main__":
    seed_mongo()
