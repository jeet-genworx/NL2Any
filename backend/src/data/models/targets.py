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

Three targets are built in, configured by environment variable. Beyond those, a
target can be built at runtime from a connection string a user supplies
(`target_from_connection`) and added to the registry (`register_target`), which
is what lets the frontend accept a database nobody configured in advance. A
user-supplied target differs only in where its connection and paths come from --
carried on the instance instead of looked up in settings -- so every consumer
downstream, from the adapter factory to the ingestion pipeline, treats it
identically to a built-in one.
"""

import hashlib
import re
from dataclasses import dataclass

from backend.src.config import settings
from backend.src.data.clients.detector import (
    ConnectionInfo,
    driver_connection,
    identity_of,
    parse_connection,
)
from backend.src.data.models.schema import DatabaseType


@dataclass(frozen=True)
class DatabaseTarget:
    """One selectable database: its engine, how to reach it, and where its artifacts live."""

    key: str
    label: str
    db_type: DatabaseType
    file_stem: str
    # Set on user-supplied targets, which carry their own connection and
    # artifact location rather than reading them from a named settings field.
    connection_override: str | None = None
    embeddings_path_override: str | None = None
    # The database within a MongoDB server, taken from its URI path. None for
    # PostgreSQL, whose database is part of the DSN the driver parses itself.
    mongo_database: str | None = None
    source: str = "builtin"

    @property
    def connection(self) -> str:
        """DSN (PostgreSQL) or URI (MongoDB) for this target, read live from settings."""
        if self.connection_override is not None:
            return self.connection_override
        if self.key == "finops":
            return settings.finops_dsn
        if self.db_type == DatabaseType.POSTGRESQL:
            return settings.postgres_dsn
        return settings.mongodb_uri

    @property
    def embeddings_path(self) -> str:
        """Path to this target's embeddings JSON, read live from settings."""
        if self.embeddings_path_override is not None:
            return self.embeddings_path_override
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


# Targets built at runtime from a user-supplied connection string, keyed the
# same way as the built-ins so every consumer resolves them identically. The
# process holds them in memory; `data.repositories.connection_repository` is
# what makes them outlive a restart, by re-registering them at startup.
_DYNAMIC: dict[str, DatabaseTarget] = {}


def _slug(info: ConnectionInfo, connection: str) -> str:
    """A stable, filesystem-safe key for the database a connection string reaches.

    Readable half first so artifacts on disk can be recognised at a glance
    (`pg_finopsiq_be_3f9c1d2a`), then a digest of the credential-free identity,
    which is what actually guarantees uniqueness and, more importantly,
    *stability*: the same database supplied twice derives the same key, finds
    its existing schema and embeddings on disk, and is not ingested again.
    """
    prefix = "pg" if info.db_type == DatabaseType.POSTGRESQL else "mg"
    readable = re.sub(r"[^a-z0-9]+", "_", (info.database or info.host).lower()).strip("_")
    digest = hashlib.sha256(identity_of(connection).encode("utf-8")).hexdigest()[:8]
    return "_".join(part for part in (prefix, readable, digest) if part)


def target_from_connection(connection: str, label: str | None = None) -> DatabaseTarget:
    """Build a target for a database named by a connection string.

    Does not register it and does not touch the network; it only works out the
    identity the rest of the system needs. Raises ValueError when the engine
    cannot be determined, which is the one input error worth refusing outright.

    What the target carries is the driver-ready form of the string, so every
    consumer downstream can hand it straight to psycopg or pymongo. In practice
    that is the string exactly as supplied; the one case it differs is a
    SQLAlchemy URL such as `postgresql+asyncpg://`, whose driver suffix libpq
    would reject.
    """
    cleaned = driver_connection(connection)
    info = parse_connection(cleaned)
    key = _slug(info, cleaned)

    engine = "PostgreSQL" if info.db_type == DatabaseType.POSTGRESQL else "MongoDB"
    described = info.database or info.host or key
    return DatabaseTarget(
        key=key,
        label=label.strip() if label and label.strip() else f"{described} ({engine})",
        db_type=info.db_type,
        file_stem=key,
        connection_override=cleaned,
        embeddings_path_override=f"{settings.embeddings_dir.rstrip('/')}/{key}_embeddings.json",
        # PostgreSQL's database is inside the DSN, which psycopg parses itself;
        # MongoDB's has to be pulled out, because pymongo needs it separately to
        # pick a database off the client.
        mongo_database=(info.database or None) if info.db_type == DatabaseType.MONGODB else None,
        source="user",
    )


def register_target(target: DatabaseTarget) -> DatabaseTarget:
    """Add a user-supplied target to the registry, or return the one already there.

    Registration is idempotent on the key, and the key is derived from the
    database's identity, so re-supplying a connection string that is already
    known returns the existing target -- with its label, and with whatever it
    has already ingested -- rather than creating a second one.
    """
    existing = _DYNAMIC.get(target.key)
    if existing is not None:
        return existing
    _DYNAMIC[target.key] = target
    return target


def unregister_target(key: str) -> DatabaseTarget | None:
    """Remove a user-supplied target, returning it, or None if it was not registered.

    Built-in targets cannot be removed; passing one of their keys raises.
    """
    if key in TARGETS:
        raise ValueError(f"'{key}' is a built-in database and cannot be removed.")
    return _DYNAMIC.pop(key, None)


def all_targets() -> dict[str, DatabaseTarget]:
    """Every selectable target: the built-ins first, then user-supplied ones."""
    return {**TARGETS, **_DYNAMIC}


def find_by_connection(connection: str) -> DatabaseTarget | None:
    """The existing target reaching the same database as `connection`, if any.

    Compares credential-free identities, so a connection string differing only
    by user or password still matches. Built-in targets are included: pasting
    the DSN that `POSTGRES_DSN` already points at should select that database,
    not save a second copy of it under a generated key and ingest it again.

    Returns None when nothing matches, or when `connection` cannot be parsed --
    the caller then treats it as new.
    """
    try:
        wanted = identity_of(connection)
    except ValueError:
        return None

    for target in all_targets().values():
        existing = target.connection
        if not existing.strip():
            continue
        try:
            if identity_of(existing) == wanted:
                return target
        except ValueError:
            # A configured target whose connection string this cannot parse is
            # simply not a match; it is still perfectly usable by its driver.
            continue
    return None


def dynamic_targets() -> dict[str, DatabaseTarget]:
    """Only the user-supplied targets, in registration order."""
    return dict(_DYNAMIC)


def clear_dynamic_targets() -> None:
    """Drop every user-supplied target. Used by tests and when reloading the store."""
    _DYNAMIC.clear()


def resolve_target(database: str) -> DatabaseTarget:
    """Resolve a user-supplied database name to a target.

    Checks the built-in aliases first, then the targets registered from
    connection strings, so `/ingest/{key}` and `/query` reach a user-supplied
    database through exactly the same path as a configured one.

    Raises ValueError naming the valid keys, so callers can surface it directly.
    """
    name = str(database).strip()
    key = _ALIASES.get(name.lower())
    if key is not None:
        return TARGETS[key]

    # Dynamic keys are generated, not typed, so they are matched exactly rather
    # than case-folded through the alias table.
    dynamic = _DYNAMIC.get(name)
    if dynamic is not None:
        return dynamic

    valid = ", ".join(sorted(all_targets()))
    raise ValueError(f"Unsupported database '{database}'. Use one of: {valid}.")


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
