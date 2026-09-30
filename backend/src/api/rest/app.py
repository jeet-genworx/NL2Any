"""FastAPI application initialization and setup."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.src.api.rest.routes import (
    connections_router,
    health_router,
    query_router,
    schema_router,
)
from backend.src.data.repositories import connection_repository

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Re-register the databases a user saved from a connection string.

    Their schema and embeddings are already on disk; without this they would be
    missing from the picker after a restart, and re-adding one would ingest a
    database that had already been ingested.
    """
    restored = connection_repository.load_connections()
    if restored:
        logger.info("Restored %d saved database connection(s).", len(restored))
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NL2AnyQuery API",
        description="Natural Language to Database Query Execution API",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Register routers (no middleware layer as per architecture specification)
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(schema_router)
    app.include_router(connections_router)

    return app


app = create_app()
