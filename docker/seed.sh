#!/bin/sh
# Seeds Postgres and MongoDB with synthetic demo data. Safe to re-run: both
# seeders drop/recreate their own tables/collections before inserting.
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
