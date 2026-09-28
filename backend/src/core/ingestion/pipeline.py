"""Orchestrates the full ingestion flow for a database.

extract metadata -> save schema/graph (+ MST, for PostgreSQL) -> generate model
descriptions for every table and column -> embed the table descriptions.

This is the single entrypoint the API calls when a user selects a database in
the frontend. The granular CLI commands (init-schema, describe-schema,
embed-schema) drive the same building blocks individually, for manual and
inspectable runs.

Orchestration only: each stage's logic lives in `schema/`, every path in
`data.repositories.paths`, and every write in the matching repository.
"""

from pathlib import Path
from typing import Any, NamedTuple

from backend.src.core.ingestion.schema.describe import (
    DEFAULT_BATCH_SIZE,
    apply_descriptions,
    describe_tables,
    table_descriptions,
)
from backend.src.core.ingestion.schema.embed import embed_descriptions
from backend.src.core.ingestion.schema.graph import build_schema_graph
from backend.src.core.ingestion.schema.mst import compute_minimum_spanning_tree
from backend.src.data.clients.factory import get_adapter
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.repositories import (
    description_repository,
    embedding_repository,
    graph_repository,
    paths,
    schema_repository,
)


class SchemaArtifacts(NamedTuple):
    """What stage 1 extracted and where it was written.

    `mst_path` is None for MongoDB, which has no foreign keys to reduce.
    """

    schema: DatabaseSchema
    schema_path: Path
    graph_path: Path
    mst_path: Path | None


def extract_and_save_schema(db_type: DatabaseType) -> SchemaArtifacts:
    """Extract metadata and save the schema, graph and (PostgreSQL only) MST files."""
    adapter = get_adapter(db_type)
    with adapter:
        schema = adapter.get_metadata()

    schema_path = schema_repository.save_schema(schema, paths.schema_path(db_type))
    graph_path = graph_repository.save_graph_document(
        build_schema_graph(schema), paths.graph_path(db_type)
    )

    mst_path: Path | None = None
    if db_type == DatabaseType.POSTGRESQL:
        mst_path = graph_repository.save_graph_document(
            compute_minimum_spanning_tree(schema), paths.mst_path(db_type)
        )

    return SchemaArtifacts(schema, schema_path, graph_path, mst_path)


async def run_ingestion_pipeline(
    db_type: DatabaseType,
    use_mst: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the full ingestion flow for a database and return a summary.

    1. Extract metadata and save the schema, graph and MST files.
    2. Describe every table and column, batched with relationship context. Reads
       from the MST (tree order, reduced edges) when `use_mst` is set and the
       database has one, otherwise from the plain graph (extraction order, full
       edge set). MongoDB always uses the graph.
    3. Embed the table descriptions. Column descriptions are not embedded.

    The descriptions are written twice: in full to the documentation TOML, and
    folded back into the canonical schema TOML, which is the file query
    processing reads.
    """
    artifacts = extract_and_save_schema(db_type)

    used_mst = use_mst and artifacts.mst_path is not None
    source_path = artifacts.mst_path if used_mst else artifacts.graph_path

    descriptions = await describe_tables(source_path, batch_size=batch_size)

    # Stage 1 wrote the schema TOML before any description existed, so this is
    # what actually populates its `description` fields.
    column_description_count = apply_descriptions(artifacts.schema, descriptions)
    schema_repository.save_schema(artifacts.schema, artifacts.schema_path)

    # Written from the completed schema, so the documentation file lists every
    # column and can answer a request on its own.
    descriptions_path = description_repository.save_descriptions(
        descriptions,
        paths.descriptions_path(db_type),
        database_type=db_type,
        database_name=artifacts.schema.database_name,
        schema=artifacts.schema,
    )

    embeddings = await embed_descriptions(table_descriptions(descriptions))
    embeddings_path = embedding_repository.save_embeddings(
        embeddings, paths.embeddings_path(db_type)
    )

    return {
        "database_type": db_type.value,
        "database_name": artifacts.schema.database_name,
        "table_count": len(artifacts.schema.objects),
        "relationship_count": len(artifacts.schema.relationships),
        "used_mst": used_mst,
        "description_count": len(descriptions),
        "column_description_count": column_description_count,
        "embedding_count": len(embeddings),
        "embedding_dimensions": len(next(iter(embeddings.values()))) if embeddings else 0,
        "schema_path": str(artifacts.schema_path),
        "graph_path": str(artifacts.graph_path),
        "mst_path": str(artifacts.mst_path) if artifacts.mst_path else None,
        "descriptions_path": str(descriptions_path),
        "embeddings_path": str(embeddings_path),
    }
