"""FastAPI application initialization and setup."""

from fastapi import FastAPI
from backend.src.api.rest.routes import health_router, query_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NL2AnyQuery API",
        description="Natural Language to Database Query Execution API",
        version="0.2.0",
    )

    # Register routers (no middleware layer as per architecture specification)
    app.include_router(health_router)
    app.include_router(query_router)

    return app


app = create_app()
