"""FastAPI dependency injection and parameter resolution."""

from fastapi import HTTPException
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.data.models.schema import DatabaseType

_orchestrator_instance: NL2AnyQueryOrchestrator | None = None


def get_orchestrator() -> NL2AnyQueryOrchestrator:
    """Retrieve or initialize the shared orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = NL2AnyQueryOrchestrator()
    return _orchestrator_instance


def resolve_db_type(database_type: str) -> DatabaseType:
    """Validate and resolve a database string into DatabaseType."""
    db_clean = database_type.lower()
    if db_clean in ("postgres", "postgresql"):
        return DatabaseType.POSTGRESQL
    elif db_clean in ("mongo", "mongodb"):
        return DatabaseType.MONGODB
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported database '{database_type}'. Use 'postgres' or 'mongo'.",
    )
