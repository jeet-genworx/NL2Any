"""High-level schema manager coordinating discovery, enrichment, and TOML storage."""

from pathlib import Path
from ingestion.databases.base import DatabaseAdapter
from ingestion.databases.detector import detect_database_type
from ingestion.databases.mongo.adapter import MongoDBAdapter
from ingestion.databases.postgres.adapter import PostgreSQLAdapter
from query_processing.models.schema import DatabaseSchema, DatabaseType
from ingestion.schema.profiler import SchemaProfiler
from ingestion.schema.toml_store import load_schema_file, save_schema_file


def get_default_schema_path(database_type: DatabaseType) -> Path:
    """Return canonical schema TOML path for the database type."""
    filename = "postgres.toml" if database_type == DatabaseType.POSTGRESQL else "mongo.toml"
    return Path("ingestion/schemas") / filename


class SchemaManager:
    """Coordinates schema extraction, enrichment, and persistence."""

    def __init__(self, profiler: SchemaProfiler | None = None) -> None:
        self.profiler = profiler or SchemaProfiler()

    def get_adapter_for_connection(self, connection_string: str) -> DatabaseAdapter:
        """Create the appropriate DatabaseAdapter for a given connection string."""
        db_type = detect_database_type(connection_string)
        if db_type == DatabaseType.POSTGRESQL:
            return PostgreSQLAdapter(dsn=connection_string)
        elif db_type == DatabaseType.MONGODB:
            return MongoDBAdapter(uri=connection_string)
        raise ValueError(f"No adapter available for database type: {db_type}")

    async def initialize_schema(
        self,
        adapter: DatabaseAdapter,
        output_path: Path | str | None = None,
        *,
        describe_with_slm: bool = False,
        refresh: bool = False,
    ) -> DatabaseSchema:
        """Extract schema from database adapter, optionally enrich, and save to TOML."""
        # 1. Authoritative deterministic metadata extraction
        schema = adapter.get_metadata()

        # 2. Determine target file path
        if output_path is None:
            output_path = get_default_schema_path(schema.database_type)
        out_path = Path(output_path)

        # 3. If file already exists and not refreshing, preserve existing descriptions
        if out_path.exists() and not refresh:
            try:
                existing_schema = load_schema_file(out_path)
                for obj in schema.objects:
                    existing_obj = existing_schema.get_object(obj.name)
                    if existing_obj:
                        if existing_obj.description:
                            obj.description = existing_obj.description
                        for field in obj.fields:
                            existing_field = existing_obj.get_field(field.name)
                            if existing_field and existing_field.description:
                                field.description = existing_field.description
                schema.description_generated_at = existing_schema.description_generated_at
            except Exception:
                pass

        # 4. Optional SLM enrichment
        if describe_with_slm:
            schema = await self.profiler.enrich_schema(schema, refresh=refresh)

        # 5. Save canonical semantic TOML
        save_schema_file(schema, out_path)
        return schema
