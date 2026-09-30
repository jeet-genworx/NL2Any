"""FastAPI dependency injection and parameter resolution."""

from fastapi import HTTPException
from backend.src.core.ingestion.pipeline import existing_ingestion
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import DatabaseTarget, resolve_target
from backend.src.data.repositories import paths

_orchestrator_instance: NL2AnyQueryOrchestrator | None = None


def is_ingested(target: DatabaseTarget) -> bool:
    """Whether this database can be queried without ingesting it first.

    Defers to the same check the ingestion pipeline uses to decide whether to
    skip, so the picker never reports a database as ready when selecting it
    would still trigger a full run.

    The existence check in front of it is only to keep the listings quiet. A
    database that has never been ingested has no embeddings file, and asking the
    pipeline about it logs the resulting FileNotFoundError with a traceback --
    which is useful when a file was expected and is not there, and pure noise on
    every listing of a database nobody has ingested yet.
    """
    if not paths.embeddings_path(target).exists():
        return False
    return existing_ingestion(target) is not None


def get_orchestrator() -> NL2AnyQueryOrchestrator:
    """Retrieve or initialize the shared orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = NL2AnyQueryOrchestrator()
    return _orchestrator_instance


def resolve_db_target(database: str) -> DatabaseTarget:
    """Validate and resolve a database string into a DatabaseTarget.

    Targets, not engine types: `postgres` and `finops` are both PostgreSQL but
    are different databases with their own schemas and embeddings.
    """
    try:
        return resolve_target(database)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


def resolve_db_type(database_type: str) -> DatabaseType:
    """Resolve a database string into its engine type."""
    return resolve_db_target(database_type).db_type
