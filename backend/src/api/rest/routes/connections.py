"""Routes for adding a database from a connection string.

Adding a database is deliberately separate from ingesting it. This module's job
ends once a connection string has been understood, proven to reach something,
and saved under a key; `POST /ingest/{key}` then runs the unchanged ingestion
pipeline against it, exactly as it does for a database configured by
environment variable.

The one behaviour worth knowing about is on `POST /connections`: because a
target's key is derived from the database's identity rather than from the
string that named it, supplying a connection string for a database that is
already saved returns the existing entry with `already_saved: true`. Its schema
and embeddings are still on disk, so selecting it costs nothing and the
ingestion pipeline is never asked to rebuild them.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.src.api.rest.dependencies import is_ingested
from backend.src.data.clients.detector import parse_connection, redact
from backend.src.data.clients.factory import get_adapter
from backend.src.data.models.targets import (
    DatabaseTarget,
    dynamic_targets,
    find_by_connection,
    register_target,
    target_from_connection,
    unregister_target,
)
from backend.src.data.repositories import connection_repository, paths
from backend.src.schemas.request import ConnectionRequest, ConnectionResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Connections"])


def _describe(target: DatabaseTarget, *, already_saved: bool = False) -> ConnectionResponse:
    """Build the picker's view of a target, without disclosing its connection string."""
    try:
        database_name = parse_connection(target.connection).database
    except ValueError:
        database_name = ""

    return ConnectionResponse(
        key=target.key,
        label=target.label,
        database_type=target.db_type.value,
        database_name=database_name,
        # The same test the ingestion pipeline uses to decide whether to skip,
        # so the frontend never reports a database as ready when selecting it
        # would still trigger a full ingestion run.
        ingested=is_ingested(target),
        already_saved=already_saved,
        source=target.source,
    )


def _probe(target: DatabaseTarget) -> None:
    """Open the database once to prove the connection string works.

    Worth the round-trip: without it a typo in a host or password surfaces
    several minutes into ingestion, as a failure that looks like the pipeline's
    fault. Connection errors are reported as 502 -- the request was well-formed,
    the database it named did not answer.
    """
    try:
        with get_adapter(target) as adapter:
            adapter.ping()
    except Exception as err:
        logger.warning("Probe failed for %s: %s", redact(target.connection), err)
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to the database: {err}",
        )


@router.get("/connections", response_model=list[ConnectionResponse])
def list_connections_endpoint() -> list[ConnectionResponse]:
    """List the databases saved from a connection string.

    The built-in, environment-configured databases are not listed here; the
    picker gets everything selectable from `/databases`.
    """
    return [_describe(target) for target in dynamic_targets().values()]


@router.post("/connections", response_model=ConnectionResponse)
def create_connection_endpoint(request: ConnectionRequest) -> ConnectionResponse:
    """Save a database named by a connection string, and return its key.

    Returns the existing entry, with `already_saved: true`, when this database
    is already known -- rather than saving a duplicate that would re-ingest a
    schema already on disk. "Already known" covers the built-in, configured
    databases too: pasting the DSN behind `POSTGRES_DSN` selects that database
    rather than adding a second copy of it.
    """
    try:
        target = target_from_connection(request.connection_string, label=request.label)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    known = find_by_connection(request.connection_string)
    if known is not None:
        return _describe(known, already_saved=True)

    _probe(target)

    registered = register_target(target)
    connection_repository.save_connections()
    logger.info("Saved connection %s as '%s'.", redact(target.connection), registered.key)
    return _describe(registered)


@router.delete("/connections/{key}", response_model=ConnectionResponse)
def delete_connection_endpoint(key: str, delete_artifacts: bool = False) -> ConnectionResponse:
    """Forget a saved database.

    Its ingested artifacts are left on disk by default, so re-adding the same
    connection string later finds them again and skips ingestion. Pass
    `delete_artifacts=true` to remove them as well.
    """
    try:
        removed = unregister_target(key)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if removed is None:
        raise HTTPException(status_code=404, detail=f"No saved connection with key '{key}'.")

    connection_repository.save_connections()

    if delete_artifacts:
        for path in (
            paths.schema_path(removed),
            paths.graph_path(removed),
            paths.mst_path(removed),
            paths.descriptions_path(removed),
            paths.embeddings_path(removed),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete %s.", path, exc_info=True)

    return _describe(removed)
