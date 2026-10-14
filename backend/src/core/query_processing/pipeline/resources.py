"""Per-database artifacts the pipeline reads: schema, embeddings, join graph.

All three are derived from files written by ingestion and are expensive enough
to load that they are cached for the life of the process. They are grouped here
because they share that lifecycle -- and the invalidation that goes with it.
"""

from __future__ import annotations

from typing import Any

from backend.src.config import settings
from backend.src.core.ingestion.schema.manager import get_default_schema_path
from backend.src.core.query_processing.pipeline.join_graph import JoinGraph
from backend.src.core.query_processing.retrieval.store import EmbeddingStore
from backend.src.core.query_processing.retrieval.vector import VectorRetriever
from backend.src.data.clients.detector import detect_database_type
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories import paths, schema_repository


def resolve_database_target(
    database: str | DatabaseType | DatabaseTarget,
) -> tuple[DatabaseTarget, DatabaseType]:
    """Turn a request's `database` argument into (target, engine type).

    A name that is not a configured target is still accepted: the engine is
    sniffed from the string, so "postgres" works before any target is defined.
    """
    if isinstance(database, DatabaseTarget):
        return database, database.db_type
    if isinstance(database, DatabaseType):
        return as_target(database), database
    if isinstance(database, str):
        try:
            target = resolve_target(database)
            return target, target.db_type
        except ValueError:
            db_type = detect_database_type(database)
            return as_target(db_type), db_type
    return as_target(DatabaseType.POSTGRESQL), DatabaseType.POSTGRESQL


class SchemaResources:
    """Schemas, vector retrievers and join graphs, each loaded once.

    An injected `vector_retriever` or `embedding_store` overrides for every
    database; otherwise each target lazily loads its own embeddings file.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore | None = None,
        vector_retriever: VectorRetriever | None = None,
    ) -> None:
        self.embedding_store = embedding_store
        self.injected_retriever = vector_retriever
        self.schemas: dict[Any, DatabaseSchema] = {}
        self.vector_retrievers: dict[Any, VectorRetriever] = {}
        # Join graphs are derived purely from a schema, so one per schema object
        # is reused across every question asked against that database. The
        # schema is kept beside its graph so a recycled id cannot serve a graph
        # built from a different schema.
        self.join_graphs: dict[int, tuple[DatabaseSchema, JoinGraph]] = {}

    def invalidate(self) -> None:
        """Drop every cached artifact, so the next question re-reads from disk.

        Ingestion rewrites the files these were read from, so without this a
        re-ingested database keeps being queried through the artifacts that
        existed before it ran -- the run reports success and nothing it produced
        is ever used.

        Every target is dropped rather than just the one re-ingested: the caches
        are keyed by whatever the caller passed (a target, a bare engine type or
        a name), so there is no single key to remove, and ingestion is rare
        enough that the cost is one lazy reload per target still in use.
        """
        self.schemas.clear()
        self.vector_retrievers.clear()
        self.join_graphs.clear()

    def schema(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file."""
        if database in self.schemas:
            return self.schemas[database]

        try:
            target = (
                as_target(database) if not isinstance(database, str) else resolve_target(database)
            )
            if target.key in self.schemas:
                return self.schemas[target.key]
            schema_path = paths.schema_path(target)
        except Exception:
            schema_path = get_default_schema_path(database)
            target = None

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. Run 'init-schema' first."
            )
        schema = schema_repository.load_schema(schema_path)
        key = target.key if target else database
        self.schemas[key] = schema
        return schema

    def vector_retriever(
        self, database: DatabaseTarget | DatabaseType | str
    ) -> VectorRetriever:
        """Get or initialize the vector retriever for this database's embeddings."""
        if self.injected_retriever is not None:
            return self.injected_retriever

        if database in self.vector_retrievers:
            return self.vector_retrievers[database]

        try:
            target = (
                as_target(database) if not isinstance(database, str) else resolve_target(database)
            )
            key = target.key
            emb_path = str(paths.embeddings_path(target))
        except Exception:
            key = database
            emb_path = (
                settings.postgres_embeddings_path
                if database == DatabaseType.POSTGRESQL
                else settings.mongo_embeddings_path
            )

        if key not in self.vector_retrievers:
            store = self.embedding_store or EmbeddingStore(emb_path)
            self.vector_retrievers[key] = VectorRetriever(store)
        return self.vector_retrievers[key]

    def join_graph(self, schema: DatabaseSchema) -> JoinGraph:
        """Join graph for a schema, built once and reused.

        Keyed by schema identity rather than by database name, because a
        reloaded schema is a different object that must get a freshly built
        graph. The cached schema is compared by identity on every hit: CPython
        reuses the id of a collected object, and a graph from a stale schema
        would hand the selector tables from another database entirely.
        """
        key = id(schema)
        cached = self.join_graphs.get(key)
        if cached is not None and cached[0] is schema:
            return cached[1]
        graph = JoinGraph(schema)
        self.join_graphs[key] = (schema, graph)
        return graph
