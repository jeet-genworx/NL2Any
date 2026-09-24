#!/bin/sh
# Seeds Postgres and MongoDB with synthetic demo data, then extracts the
# canonical schema TOML that query processing reads from. Safe to re-run:
# seeders drop/recreate their own tables/collections, and init-schema
# overwrites the TOML files from live adapter metadata each time.
set -e

echo "Seeding PostgreSQL..."
uv run seed-postgres

echo "Seeding MongoDB..."
uv run seed-mongo

echo "Extracting PostgreSQL schema..."
uv run init-schema --database postgres

echo "Extracting MongoDB schema..."
uv run init-schema --database mongo

echo "Seed complete."
