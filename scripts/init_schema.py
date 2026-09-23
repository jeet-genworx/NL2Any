"""Script to initialize database schema and write canonical TOML file."""

import argparse
import asyncio
from pathlib import Path

from nl2anyquery.core.config import settings
from nl2anyquery.databases.mongo.adapter import MongoDBAdapter
from nl2anyquery.databases.postgres.adapter import PostgreSQLAdapter
from nl2anyquery.models.schema import DatabaseType
from nl2anyquery.schema.manager import SchemaManager, get_default_schema_path


async def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize database schema TOML.")
    parser.add_argument(
        "--database",
        "-d",
        choices=["postgres", "postgresql", "mongo", "mongodb"],
        required=True,
        help="Target database type",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output TOML file path (defaults to schemas/<db>.toml)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Enrich schema with table and column descriptions via KoboldCpp SLM",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration of existing descriptions",
    )
    args = parser.parse_args()

    db_arg = args.database.lower()
    if db_arg in ("postgres", "postgresql"):
        db_type = DatabaseType.POSTGRESQL
        adapter = PostgreSQLAdapter(dsn=settings.postgres_dsn)
    else:
        db_type = DatabaseType.MONGODB
        adapter = MongoDBAdapter(uri=settings.mongodb_uri, database=settings.mongodb_database)

    out_path = Path(args.output) if args.output else get_default_schema_path(db_type)
    print(f"Extracting metadata for {db_type.value}...")
    manager = SchemaManager()

    with adapter:
        schema = await manager.initialize_schema(
            adapter=adapter,
            output_path=out_path,
            describe_with_slm=args.describe,
            refresh=args.refresh,
        )

    print(f"Successfully generated schema: {out_path}")
    print(f"Total objects: {len(schema.objects)}")
    for obj in schema.objects:
        desc_preview = f" - '{obj.description}'" if obj.description else " (no description)"
        print(f"  [{obj.kind.value}] {obj.name} ({len(obj.fields)} fields){desc_preview}")
    print(f"Total relationships: {len(schema.relationships)}")


if __name__ == "__main__":
    asyncio.run(main())
