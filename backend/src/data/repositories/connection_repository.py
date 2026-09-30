"""Persistence for the databases a user added from a connection string.

The other repositories in this package store what ingestion *produced*. This
one stores what it was pointed at: the connection strings supplied through the
frontend, so the picker still lists those databases after the API restarts and
their already-ingested schema and embeddings are found rather than rebuilt.

The file is keyed by the target key, which is derived from the database's
identity, so re-saving a connection string that is already known overwrites its
entry instead of adding a second one.

This file contains credentials in plain text, exactly as `.env` does. It is
gitignored, and `settings.connections_path` moves it somewhere better for a
deployment that wants that.
"""

import logging
from pathlib import Path
from typing import Any

from backend.src.config import settings
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import (
    DatabaseTarget,
    clear_dynamic_targets,
    dynamic_targets,
    register_target,
)
from backend.src.data.repositories import files

logger = logging.getLogger(__name__)


def connections_path() -> Path:
    """Where the saved connections are stored."""
    return Path(settings.connections_path)


def _to_record(target: DatabaseTarget) -> dict[str, Any]:
    """The stored shape of one target."""
    return {
        "key": target.key,
        "label": target.label,
        "database_type": target.db_type.value,
        "file_stem": target.file_stem,
        "connection": target.connection,
        "embeddings_path": target.embeddings_path,
        "mongo_database": target.mongo_database,
    }


def _from_record(record: dict[str, Any]) -> DatabaseTarget:
    """Rebuild a target from its stored shape.

    Deliberately reconstructs from the stored fields rather than re-deriving
    from the connection string: a target that was saved keeps the key its
    artifacts are named after, even if the key derivation later changes.
    """
    return DatabaseTarget(
        key=record["key"],
        label=record.get("label") or record["key"],
        db_type=DatabaseType(record["database_type"]),
        file_stem=record.get("file_stem") or record["key"],
        connection_override=record["connection"],
        embeddings_path_override=record.get("embeddings_path"),
        mongo_database=record.get("mongo_database"),
        source="user",
    )


def load_connections() -> list[DatabaseTarget]:
    """Read the saved connections and register every one of them.

    Called once at API startup. A missing file is the normal first-run state and
    yields an empty list; a corrupt or partially-readable file is logged and
    treated the same way, because failing to parse one saved connection should
    not stop the service from starting with its built-in databases.
    """
    path = connections_path()
    if not path.is_file():
        return []

    try:
        records = files.read_json(path)
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable connections file at %s.", path, exc_info=True)
        return []

    if not isinstance(records, list):
        logger.warning("Connections file at %s is not a list; ignoring it.", path)
        return []

    clear_dynamic_targets()
    loaded: list[DatabaseTarget] = []
    for record in records:
        try:
            loaded.append(register_target(_from_record(record)))
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed connection record in %s.", path, exc_info=True)
    return loaded


def save_connections() -> Path:
    """Write every currently registered user-supplied target to disk."""
    return files.write_json(
        connections_path(),
        [_to_record(target) for target in dynamic_targets().values()],
        indent=2,
    )
