"""Standalone entrypoint for seeding PostgreSQL with synthetic demo data."""

from ingestion.databases.seed.postgres import seed_postgres

if __name__ == "__main__":
    seed_postgres()
