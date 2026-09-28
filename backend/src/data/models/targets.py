"""Selectable database targets.

`DatabaseType` says which *engine* a query is written for -- it drives SQL
dialect, the generator, the validator and the executor. It is not enough to
identify *which database* to talk to, because more than one target can share an
engine: the demo `postgres` database and the `finops` database are both
PostgreSQL, with different DSNs, different schemas and different embeddings.

A `DatabaseTarget` is that missing identity. Everything that varies per database
-- connection, the stem its schema/graph/MST/description files are named after,
and its embeddings file -- hangs off the target, while `db_type` keeps driving
the engine-specific behaviour.

Connection strings and paths are read live from settings through properties
rather than captured at import, so overriding an environment variable in a test
or a container takes effect without rebuilding the registry.
"""

from dataclasses import dataclass

from backend.src.config import settings
from backend.src.data.models.schema import DatabaseType


@dataclass(frozen=True)
class DatabaseTarget:
    """One selectable database: its engine, how to reach it, and where its artifacts live."""

    key: str
    label: str
    db_type: DatabaseType
    file_stem: str

    @property
    def connection(self) -> str:
        """DSN (PostgreSQL) or URI (MongoDB) for this target, read live from settings."""
        if self.key == "finops":
            return settings.finops_dsn
        if self.db_type == DatabaseType.POSTGRESQL:
            return settings.postgres_dsn
        return settings.mongodb_uri

    @property
    def embeddings_path(self) -> str:
        """Path to this target's embeddings JSON, read live from settings."""
        if self.key == "finops":
            return settings.finops_embeddings_path
        if self.db_type == DatabaseType.POSTGRESQL:
            return settings.postgres_embeddings_path
        return settings.mongo_embeddings_path

    @property
    def has_mst(self) -> bool:
        """Only PostgreSQL targets have foreign keys, and so an MST to reduce."""
        return self.db_type == DatabaseType.POSTGRESQL

    @property
    def configured(self) -> bool:
        """Whether this target has a connection string set.

        Surfaced in the picker so an unusable target says why, rather than
        failing only once someone selects it.
        """
        return bool(self.connection.strip())


POSTGRES_TARGET = DatabaseTarget(
    key="postgres",
    label="PostgreSQL (Relational)",
    db_type=DatabaseType.POSTGRESQL,
    file_stem="postgres",
)
MONGO_TARGET = DatabaseTarget(
    key="mongo",
    label="MongoDB (Document)",
    db_type=DatabaseType.MONGODB,
    file_stem="mongo",
)
FINOPS_TARGET = DatabaseTarget(
    key="finops",
    label="FinOps (PostgreSQL)",
    db_type=DatabaseType.POSTGRESQL,
    file_stem="finops",
)

TARGETS: dict[str, DatabaseTarget] = {
    target.key: target for target in (POSTGRES_TARGET, MONGO_TARGET, FINOPS_TARGET)
}

# Spellings accepted from the CLI, the API path and the frontend.
_ALIASES: dict[str, str] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "mongo": "mongo",
    "mongodb": "mongo",
    "finops": "finops",
    "finopsiq": "finops",
    "finopsiq_be": "finops",
}


def resolve_target(database: str) -> DatabaseTarget:
    """Resolve a user-supplied database name to a target.

    Raises ValueError naming the valid keys, so callers can surface it directly.
    """
    key = _ALIASES.get(str(database).strip().lower())
    if key is None:
        valid = ", ".join(sorted(TARGETS))
        raise ValueError(f"Unsupported database '{database}'. Use one of: {valid}.")
    return TARGETS[key]


def default_target_for_type(db_type: DatabaseType) -> DatabaseTarget:
    """The target a bare DatabaseType refers to.

    DatabaseType alone is ambiguous now that two targets are PostgreSQL; this
    keeps type-keyed call sites (CLI flags, tests, query processing) pointing at
    the original demo databases.
    """
    return POSTGRES_TARGET if db_type == DatabaseType.POSTGRESQL else MONGO_TARGET


def as_target(value: "DatabaseTarget | DatabaseType") -> DatabaseTarget:
    """Accept either a target or a bare engine type, and return a target.

    Lets every path, adapter and pipeline entry point take the richer target
    while older type-keyed callers keep working unchanged.
    """
    return value if isinstance(value, DatabaseTarget) else default_target_for_type(value)
