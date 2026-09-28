"""REST API routes package."""

from backend.src.api.rest.routes.health import router as health_router
from backend.src.api.rest.routes.query import router as query_router

__all__ = ["health_router", "query_router"]
