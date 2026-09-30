"""REST API routes package."""

from backend.src.api.rest.routes.connections import router as connections_router
from backend.src.api.rest.routes.health import router as health_router
from backend.src.api.rest.routes.query import router as query_router
from backend.src.api.rest.routes.schema import router as schema_router

__all__ = ["connections_router", "health_router", "query_router", "schema_router"]
