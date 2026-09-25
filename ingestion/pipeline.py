"""Orchestrates the full ingestion flow for a database.

extract metadata -> save schema/graph (+ MST, for Postgres) -> generate SLM
table and column descriptions -> embed the table descriptions. This is the single entrypoint
the API (and therefore the frontend, when a user selects a database) calls;
the granular CLI commands (init-schema, describe-schema, embed-schema) call
the same building blocks individually for manual/inspectable use.
"""

from pathlib import Path
from typing import Any

from query_processing.core.config import settings
from query_processing.models.schema import DatabaseSchema, DatabaseType
from ingestion.databases.base import DatabaseAdapter
from ingestion.databases.mongo.adapter import MongoDBAdapter
from ingestion.databases.postgres.adapter import PostgreSQLAdapter
from ingestion.schema.manager import get_default_schema_path
from ingestion.schema.toml_store import save_schema_file
from ingestion.schema.graph import get_default_graph_path, save_schema_graph
from ingestion.schema.mst import get_default_mst_path, save_minimum_spanning_tree
from ingestion.schema.describe import (
    DEFAULT_BATCH_SIZE,
    apply_descriptions,
    describe_tables,
    get_default_descriptions_path,
    save_descriptions,
    table_descriptions,
)
from ingestion.schema.embed import (
    embed_descriptions,
    get_default_embeddings_path,
    save_embeddings,
)


def get_adapter(db_type: DatabaseType) -> DatabaseAdapter:
    """Build the appropriate database adapter, using connection settings from .env."""
    if db_type == DatabaseType.POSTGRESQL:
        return PostgreSQLAdapter(dsn=settings.postgres_dsn)
    return MongoDBAdapter(uri=settings.mongodb_uri, database=settings.mongodb_database)


def extract_and_save_schema(
    db_type: DatabaseType,
) -> tuple[DatabaseSchema, Path, Path, Path | None]:
    """Extract metadata and save the canonical schema TOML, graph TOML, and
    (Postgres only, since MongoDB has no foreign keys) the MST TOML.

    Returns (schema, schema_path, graph_path, mst_path); mst_path is None for MongoDB.
    """
    adapter = get_adapter(db_type)
    with adapter:
        schema = adapter.get_metadata()

    schema_path = get_default_schema_path(db_type)
    save_schema_file(schema, schema_path)

    graph_path = get_default_graph_path(db_type)
    save_schema_graph(schema, graph_path)

    mst_path: Path | None = None
    if db_type == DatabaseType.POSTGRESQL:
        mst_path = get_default_mst_path(db_type)
        save_minimum_spanning_tree(schema, mst_path)

    return schema, schema_path, graph_path, mst_path


async def run_ingestion_pipeline(
    db_type: DatabaseType,
    use_mst: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the full ingestion flow for a database.

    1. Extract metadata via the database adapter, save the schema + graph TOML
       (and MST TOML, for Postgres).
    2. Generate an SLM description for every table *and* for every one of its
       columns, batched with relationship context. Reads from the MST (tree
       order, reduced edges) when `use_mst` is true and the database has one;
       otherwise reads straight from the plain graph (extraction order, full
       edge set). MongoDB always uses the graph, since it has no foreign-key
       concept and therefore no MST. Both the table and the column descriptions
       are written to the descriptions JSON and into the canonical schema TOML.
    3. Embed the table descriptions via the local sentence-transformer model.
       Column descriptions are not embedded.

    This is exactly the flow triggered when a user selects a database in the
    frontend and clicks "Initialize Database".
    """
    schema, schema_path, graph_path, mst_path = extract_and_save_schema(db_type)

    used_mst = use_mst and mst_path is not None
    source_path = mst_path if used_mst else graph_path

    descriptions = await describe_tables(source_path, batch_size=batch_size)
    table_only = table_descriptions(descriptions)

    # The descriptions JSON holds the full record: each table's description and
    # every column description.
    descriptions_path = get_default_descriptions_path(db_type)
    save_descriptions(descriptions, descriptions_path)

    # Fold both the table and the column descriptions back into the canonical
    # schema TOML -- the file query processing reads -- and re-save it. Stage 1
    # wrote that file before any description existed, so this is what actually
    # populates the `description` fields and `description_generated_at`.
    column_description_count = apply_descriptions(schema, descriptions)
    save_schema_file(schema, schema_path)

    # Only the table description is embedded. Column descriptions are
    # documentation in the schema TOML and stay out of the retrieval vectors.
    embeddings = await embed_descriptions(table_only)
    embeddings_path = get_default_embeddings_path(db_type)
    save_embeddings(embeddings, embeddings_path)

    return {
        "database_type": db_type.value,
        "database_name": schema.database_name,
        "table_count": len(schema.objects),
        "relationship_count": len(schema.relationships),
        "used_mst": used_mst,
        "description_count": len(descriptions),
        "column_description_count": column_description_count,
        "embedding_count": len(embeddings),
        "embedding_dimensions": len(next(iter(embeddings.values()))) if embeddings else 0,
        "schema_path": str(schema_path),
        "graph_path": str(graph_path),
        "mst_path": str(mst_path) if mst_path else None,
        "descriptions_path": str(descriptions_path),
        "embeddings_path": str(embeddings_path),
    }
