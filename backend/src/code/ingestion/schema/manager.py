"""Canonical schema TOML path resolution, used by both ingestion and query processing."""

from pathlib import Path
from backend.src.data.models.schema import DatabaseType


def _get_schemas_dir() -> Path:
    p = Path("backend/src/data/schemas")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[3] / "data" / "schemas"
    if alt.exists():
        return alt
    return p


def get_default_schema_path(database_type: DatabaseType) -> Path:
    """Return canonical schema TOML path for the database type."""
    filename = "postgres.toml" if database_type == DatabaseType.POSTGRESQL else "mongo.toml"
    return _get_schemas_dir() / filename
