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

import logging
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
from backend.src.data.models.targets import DatabaseTarget, as_target
from backend.src.data.repositories import (
    description_repository,
    embedding_repository,
    graph_repository,
    paths,
    schema_repository,
)

logger = logging.getLogger(__name__)


class SchemaArtifacts(NamedTuple):
    """What stage 1 extracted and where it was written.

    `mst_path` is None for MongoDB, which has no foreign keys to reduce.
    """

    schema: DatabaseSchema
    schema_path: Path
    graph_path: Path
    mst_path: Path | None


def extract_and_save_schema(database: DatabaseTarget | DatabaseType) -> SchemaArtifacts:
    """Extract metadata and save the schema, graph and (PostgreSQL only) MST files."""
    target = as_target(database)
    adapter = get_adapter(target)
    with adapter:
        schema = adapter.get_metadata()

    schema_path = schema_repository.save_schema(schema, paths.schema_path(target))
    graph_path = graph_repository.save_graph_document(
        build_schema_graph(schema), paths.graph_path(target)
    )

    mst_path: Path | None = None
    if target.has_mst:
        mst_path = graph_repository.save_graph_document(
            compute_minimum_spanning_tree(schema), paths.mst_path(target)
        )

    return SchemaArtifacts(schema, schema_path, graph_path, mst_path)


def existing_ingestion(
    database: DatabaseTarget | DatabaseType, use_mst: bool = True
) -> dict[str, Any] | None:
    """Summarize a previous ingestion from the artifacts already on disk.

    Returns None when there is nothing usable to reuse -- which is the signal to
    run the pipeline for real. The embeddings file is the marker, since it is
    written last and is what query processing actually retrieves against: if it
    is present and non-empty, the run that produced it got all the way through.
    A schema TOML must be there too, otherwise the artifacts are half-written
    and re-running is the right move.
    """
    target = as_target(database)
    embeddings_path = paths.embeddings_path(target)
    try:
        embeddings = embedding_repository.load_embeddings(embeddings_path)
    except (OSError, ValueError):
        # Unreadable for any reason -- absent, a directory left by a bind mount,
        # malformed JSON, no permission -- means there is nothing to reuse. The
        # caller re-ingests rather than failing, so one broken target cannot
        # take down a listing of all of them.
        logger.warning("Ignoring unusable embeddings at %s.", embeddings_path, exc_info=True)
        return None
    if not embeddings:
        return None

    schema_path = paths.schema_path(target)
    if not schema_path.exists():
        return None
    schema = schema_repository.load_schema(schema_path)

    descriptions_path = paths.descriptions_path(target)
    descriptions = (
        description_repository.load_descriptions(descriptions_path)
        if descriptions_path.exists()
        else {}
    )

    mst_path = paths.mst_path(target)
    has_mst = target.has_mst and mst_path.exists()

    return {
        "database": target.key,
        "database_type": target.db_type.value,
        "database_name": schema.database_name,
        "table_count": len(schema.objects),
        "relationship_count": len(schema.relationships),
        # The setting that would apply to a fresh run. The artifacts on disk may
        # have been generated with a different one -- it is not recorded.
        "used_mst": use_mst and has_mst,
        "description_count": len(descriptions),
        "column_description_count": sum(len(t.columns) for t in descriptions.values()),
        "embedding_count": len(embeddings),
        "embedding_dimensions": len(next(iter(embeddings.values()))),
        "schema_path": str(schema_path),
        "graph_path": str(paths.graph_path(target)),
        "mst_path": str(mst_path) if has_mst else None,
        "descriptions_path": str(descriptions_path) if descriptions_path.exists() else None,
        "embeddings_path": str(embeddings_path),
        "skipped": True,
    }


async def run_ingestion_pipeline(
    database: DatabaseTarget | DatabaseType,
    use_mst: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
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

    Skipped entirely when this database already has embeddings on disk: the
    whole flow costs a database round-trip plus one model call per batch, and
    selecting a database in the frontend triggers it every time. Pass
    `force=True` to re-ingest anyway, which is what the frontend's explicit
    "Re-run Ingestion" button does and what you want after the schema changes.
    """
    target = as_target(database)

    if not force:
        reused = existing_ingestion(target, use_mst=use_mst)
        if reused is not None:
            logger.info(
                "Skipping ingestion for %s: %d embeddings already at %s.",
                target.key,
                reused["embedding_count"],
                reused["embeddings_path"],
            )
            return reused

    artifacts = extract_and_save_schema(target)

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
        paths.descriptions_path(target),
        database_type=target.db_type,
        database_name=artifacts.schema.database_name,
        schema=artifacts.schema,
    )

    embeddings = await embed_descriptions(table_descriptions(descriptions))
    embeddings_path = embedding_repository.save_embeddings(
        embeddings, paths.embeddings_path(target)
    )

    return {
        "database": target.key,
        "database_type": target.db_type.value,
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
        "skipped": False,
    }
